import os
import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    _FLASH_ATTN_AVAILABLE = False
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        use_flash_attn: bool | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        if use_flash_attn is None:
            use_flash_attn = os.getenv("NANOVLLM_USE_FLASH_ATTN", "1") != "0"
        self.use_flash_attn = use_flash_attn and _FLASH_ATTN_AVAILABLE

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k.size(1) == self.num_heads:
            return k, v
        assert self.num_heads % k.size(1) == 0
        repeat = self.num_heads // k.size(1)
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        return k, v

    def _gather_kv_from_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = block_table[block_table >= 0]
        if valid.dtype != torch.int64:
            valid = valid.to(torch.int64)
        if valid.numel() == 0:
            empty = k_cache.new_empty((0, k_cache.size(-2), k_cache.size(-1)))
            return empty, empty
        k_blocks = k_cache.index_select(0, valid)
        v_blocks = v_cache.index_select(0, valid)
        k_seq = k_blocks.reshape(-1, k_blocks.size(-2), k_blocks.size(-1))[:seqlen]
        v_seq = v_blocks.reshape(-1, v_blocks.size(-2), v_blocks.size(-1))[:seqlen]
        return k_seq, v_seq

    def _attn_torch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        prefix_len: int,
    ) -> torch.Tensor:
        if q.numel() == 0:
            return q
        k, v = self._expand_kv(k, v)
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal:
            q_len = q.size(1)
            k_len = k.size(1)
            row = torch.arange(q_len, device=q.device)[:, None]
            col = torch.arange(k_len, device=q.device)[None, :]
            mask = col > (prefix_len + row)
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        out = torch.matmul(probs, v)
        return out.transpose(0, 1)

    def _forward_torch_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        cu_q = context.cu_seqlens_q
        cu_k = context.cu_seqlens_k
        num_seqs = cu_q.numel() - 1
        outputs = []
        for i in range(num_seqs):
            q_i = q[cu_q[i]:cu_q[i + 1]]
            seqlen_k = int(cu_k[i + 1] - cu_k[i])
            if context.block_tables is None:
                k_i = k[cu_k[i]:cu_k[i + 1]]
                v_i = v[cu_k[i]:cu_k[i + 1]]
            else:
                k_i, v_i = self._gather_kv_from_cache(
                    k_cache, v_cache, context.block_tables[i], seqlen_k
                )
            prefix_len = k_i.size(0) - q_i.size(0)
            outputs.append(self._attn_torch(q_i, k_i, v_i, causal=True, prefix_len=prefix_len))
        return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.size(1), q.size(2)))

    def _forward_torch_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        outputs = []
        for i in range(q.size(0)):
            seqlen_k = int(context.context_lens[i])
            k_i, v_i = self._gather_kv_from_cache(
                k_cache, v_cache, context.block_tables[i], seqlen_k
            )
            q_i = q[i:i + 1]
            prefix_len = k_i.size(0) - q_i.size(0)
            out_i = self._attn_torch(q_i, k_i, v_i, causal=True, prefix_len=prefix_len)
            outputs.append(out_i.unsqueeze(1))
        return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, 1, q.size(1), q.size(2)))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if self.use_flash_attn:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = self._forward_torch_prefill(q, k, v, k_cache, v_cache, context)
        else:    # decode
            if self.use_flash_attn:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
            else:
                o = self._forward_torch_decode(q, k_cache, v_cache, context)
        return o
