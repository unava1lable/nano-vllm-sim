from dataclasses import fields
from time import perf_counter

from nanovllm.config import SimConfig
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import SimModelRunner


class LLMEngine:

    def __init__(self, **kwargs):
        config_fields = {field.name for field in fields(SimConfig)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = SimConfig(**config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []

        self.sim_model_runner = SimModelRunner(config)
        self.scheduler = Scheduler(config)

    def add_request(self, prompt: str | list[int]):
        if isinstance(prompt, str):
            prompt = [ord(c) for c in prompt]
        seq = Sequence(prompt)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids, total_time = self.sim_model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, total_time

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
    ) -> list[str]:
        for prompt in prompts:
            self.add_request(prompt)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, total_time = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (total_time / 1000)
                print(f"prefill throughput: {prefill_throughput:.2f} tokens/s")
            else:
                decode_throughput = -num_tokens / (total_time / 1000)
                print(f"decode throughput: {decode_throughput:.2f} tokens/s")
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": "sim text", "token_ids": token_ids} for token_ids in outputs]
        return outputs
