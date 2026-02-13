from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # 块大小
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # 魔术方法: 没有显式调用它，但 Python 会在特定场景下自动调用
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 已缓存的块数量
    # 前缀（Prefix）Cache
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # 属性是变量
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一个块内的 token 数
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 参数 i，即 Sequence 对象用的块的下标（索引）。然后返回这个块里面的所有 token_id
    # NOTE. Sequence 里面的 block，是虚拟的 block，是根据 block 的 size，对于 Sequence 的 token_id 划分的虚拟 block
    # 此时可能还未分配真实的 block 资源。分配的逻辑在 BlockManager 中。
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
