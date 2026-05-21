from dataclasses import dataclass


@dataclass
class StepTrace:
    step_id: int
    phase: str
    seq_ids: list[int]
    scheduled_tokens: int
    latency_ms: float
    used_blocks: int
    free_blocks: int


@dataclass
class RunMetrics:
    total_steps: int
    total_latency_ms: float
    prefill_tokens: int
    decode_tokens: int
    prefill_latency_ms: float
    decode_latency_ms: float
    prefill_throughput: float
    decode_throughput: float
    max_used_blocks: int
    avg_used_blocks: float

def throughput(tokens: int, latency_ms: float) -> float:
    return 0.0 if tokens == 0 or latency_ms <= 0 else tokens / (latency_ms / 1000)


def summarize_traces(traces: list[StepTrace]) -> RunMetrics:
    if not traces:
        return RunMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    total_steps = len(traces)
    total_latency_ms = sum(t.latency_ms for t in traces)
    prefill_tokens = sum(t.scheduled_tokens for t in traces if t.phase == "prefill")
    decode_tokens = sum(t.scheduled_tokens for t in traces if t.phase == "decode")
    prefill_latency_ms = sum(t.latency_ms for t in traces if t.phase == "prefill")
    decode_latency_ms = sum(t.latency_ms for t in traces if t.phase == "decode")
    prefill_throughput = throughput(prefill_tokens, prefill_latency_ms)
    decode_throughput = throughput(decode_tokens, decode_latency_ms)
    max_used_blocks = max(t.used_blocks for t in traces)
    avg_used_blocks = sum(t.used_blocks for t in traces) / total_steps

    return RunMetrics(
        total_steps=total_steps,
        total_latency_ms=total_latency_ms,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        prefill_latency_ms=prefill_latency_ms,
        decode_latency_ms=decode_latency_ms,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        max_used_blocks=max_used_blocks,
        avg_used_blocks=avg_used_blocks,
    )