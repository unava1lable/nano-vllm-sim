from dataclasses import fields
from pprint import pprint

from nanovllm.config import SimConfig, RequestConfig
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import SimModelRunner
from nanovllm.engine.metrics import StepTrace, summarize_traces


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
        self.step_id = 0
        self.traces: list[StepTrace] = []

    def add_request(self, prompt: str | list[int], request_config: RequestConfig):
        if isinstance(prompt, str):
            prompt = [ord(c) for c in prompt]
        seq = Sequence(prompt, request_config)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        phase = "prefill" if is_prefill else "decode"
        scheduled_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        seq_ids = [seq.seq_id for seq in seqs]
        token_ids, latency_ms = self.sim_model_runner.run(seqs, is_prefill)
        trace = StepTrace(
            step_id=self.step_id,
            phase=phase,
            seq_ids=seq_ids,
            scheduled_tokens=scheduled_tokens,
            latency_ms=latency_ms,
            used_blocks=len(self.scheduler.block_manager.used_block_ids),
            free_blocks=len(self.scheduler.block_manager.free_block_ids),
        )
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        self.step_id += 1
        self.traces.append(trace)
        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        request_configs: RequestConfig | list[RequestConfig] | None = None,
    ) -> list[str]:
        if isinstance(request_configs, RequestConfig):
            request_configs = [request_configs] * len(prompts)
        elif request_configs is None:
            request_configs = [RequestConfig()] * len(prompts)
        if len(prompts) != len(request_configs):
            raise ValueError("prompts and request_configs must have the same length")

        for prompt, request_config in zip(prompts, request_configs):
            self.add_request(prompt, request_config)
        outputs = {}
        while not self.is_finished():
            output = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": "sim text", "token_ids": token_ids} for token_ids in outputs]
        return outputs
    

    def print_traces(self):
        pprint(summarize_traces(self.traces))
