from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


# BlockManger分配的单位
class Block:

    def __init__(self, block_id):
        self.block_id = block_id  # 块的 id，唯一标识
        self.ref_count = 0  # !! 引用计数，这里的计数是当前块被多少个 Sequence（这个 Sequence有相同的前缀）
        self.hash = -1
        self.token_ids = []  # block包含的token_id列表

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

# 显存块的虚拟索引系统。用虚拟的Block维护着显存块对应的数据结构。
## GPU 显存块里面存储着 Block 内的 token_ids 对应的 K 和 V 矩阵（计算出来的），
## 其他 prompt 推理时，通过 BlockManager 发现其部分 Block 已经有缓存，
## 就直接找到对应的 K 和 V，达到减少 prompt 的 prefill 阶段计算的目的，从而提升 LLM 推理系统的关键指标 TTFT
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    ## NOTE. compute_hash 除了传入待计算的 token_ids，还需要传入一个 prefix
    # 先调用 h.update，加上上一次的 hash 值
    # 然后再追加当前 block 的 token_ids 对应的 bytes
    # 最后再算一次 hash  "递推公式"
    # !! 后面 block 的 hash 值是隐式地依赖了前面的 block 的
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 从空闲 block_ids（free_block_ids）移除这个 block_id
        # 加到已使用的 block_ids（used_block_ids）
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # [prefill]
    def can_allocate(self, seq: Sequence) -> bool:
    # 当前空闲的块的数量大于 Sequence 对象（seq）所需的块，即为可分配
        return len(self.free_block_ids) >= seq.num_blocks

    # 分配 block，不过会过滤掉 命中了Cache 的 token_id
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1  # 要计算的 hash 值
        cache_miss = False  # Cache 称之为 Prefix Cache
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # token_ids 的长度和 block 大小相同（说明不是最后一个 block），
            # 则给 token_ids 计算 hash，否则维持 -1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            ## NOTE. dict 类型的属性 hash_to_block_id
            # 通过 hash 取出对应的 block_id，如果不存在设置 block_id 为 -1，
            # 也即 cache_miss
            block_id = self.hash_to_block_id.get(h, -1)

            # 中了 cache
            # double check 一下 BlockManager 中取出来的 token_ids 是不是和 seq 对象中是一样
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
            # 没中 cache，则把当前第一个空闲的 block_id（free_block_ids[0]）
            # 拿出来，调用 _allocate_block 做实际分配
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
            # 中了 cache
            # 更新累计中 cache 的 token 数（num_cached_tokens）的计数
            # block_id 是不是在已使用的 block_id（used_block_ids）中
            # 如果在，则增加这个 block 的引用计数，如果不在，也调用 _allocate_block 去分配
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            # 当前 block 有 hash 值，则把 hash 值同时更新到 block 内
            # 以及当前 BlockManager 的 hash_to_block_id 字典中'
            # 然后更新 seq 中的 block_table 这个 list，追加当前的 block_id。
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # [decode]
    # 判断的就是对于最后一个 token 是否需要新分配 block
    # 只有当这里余 1 的时候（表达式结果为 1 ），说明前面的 block 都用满了
    # 只要空闲的 block 大于 1 个即可分配。
    # 如果余数不是 1，则整个表达式结果为 0，那么变成了恒等式。不需要分配。
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 取出 Sequence 中的 block_table
        # 因为这里是 decode 阶段
        # 所以前面 prefill 的时候必然填充了 block_table，这里不会为空 list
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # 值为 1，表示的当前又到了一个新 block 的第一个 token，此时需要分配 block
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # 如果等于 0，表示之前分配的块，正好填满 token
        elif len(seq) % self.block_size == 0:
            # 当前 block 的 hash 不是 -1，这是必然的
            # 因为只有在整个 block 的 token_ids 都填满的时候才会去计算 block 的 hash
            # 也就是现在!!!
            assert last_block.hash == -1
            # 最后一个 block（也就是当前 block）的 token_ids
            token_ids = seq.block(seq.num_blocks-1)
            # 取上一个 block（如果有）的 hash 值，作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            # 更新 hash
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
