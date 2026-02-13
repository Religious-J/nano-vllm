from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    # 每个 prompt 都有一个对应的 Sequence 对象
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        # 用来分配 KV Cache 块
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 这个方法用来决定下一步该把哪些 Sequence 送去推理。
    def schedule(self) -> tuple[list[Sequence], bool]:
        ## 这边是优先调度 prefill 阶段的 Sequence
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # 从 waiting 队列中提取 Sequence（出队），追加到 running 队列中（入队）
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 对应每个提取的 Sequence，会分配 KV Cache Block
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 并且修改 Sequence 的状态为 RUNNING
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            # 追加到 running 队列中的 Sequence 也会加到 scheduled_seqs 中
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        ## 左侧出队一个Sequence。接下来是判断block_manager是否还能分配显存块
        ## 如果能，则分配显存块，并追加到scheduled_seqs
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # 如果 block_manage 不能分配显存块，需要执行特殊逻辑
            # 从 running 队列右侧出队 Sequence，然后调用 self.preempt()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 加到 waiting 队列左侧
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        # assert 保证，如果能正常走到这里，肯定会有至少一个 Sequence 被调度成功
        assert scheduled_seqs
        # 之前取出来的 Sequence 按原先顺序加回去了（往左侧加一定要逆序加才能保证原顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 后处理
    # 推理生成的新 token_id 追加到对应的 Sequence 中，如果发现 token_id 是序列终止标记（eos）
    # 表示这个 Sequence 已经都生成完毕了，然后修改其状态为 FINISH，然后释放它的显存块，并且从 running 队列中移除这个 Sequence
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
