import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        ## tp
        # model_running有多个
        # 除了LLMEngine成员持有一个外。当tp大于1的时候，还创建了多进程，每个进程都有一个ModelRunner在运行
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        ## 三大成员：model_runner、tokenizer、scheduler
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            # join() 会阻塞当前线程，直到子进程结束
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            # prompt 从字符串转成数字形式的 token_id
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # seqs 是 list[Sequence] 
        # 有一个 prompt 就会有一个 Sequence 对象
        seqs, is_prefill = self.scheduler.schedule()
        # 执行模型的前向传播，具体的推理逻辑
        ## 可以表示的多个 prompt 的结果
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 调用 postprocess，把这次生成的 token_id 追加到 seq 中
        self.scheduler.postprocess(seqs, token_ids)

        # 判断每个 seq 是否达到了完成状态，如果某个 seq 完成了，则构造成 outputs 的输出
        ## attention：
        ## token_id 是包含 prompt 的 token_id，completion_token_ids 用来返回纯生成的 token_id
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 先把采样参数转成数组，和 prompts 的数量相同
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 原始 prompt 和采样参数一起传入 add_request
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 每个 prompt 推理完成的时候，output 才有值
        # 也就是说中间状态都是空列表。此时根据 seq_id,对 outputs 重排序，从 dict 改成 list 结构
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 从 token_id 转成对应的人类可读的 token
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
