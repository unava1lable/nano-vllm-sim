from dataclasses import dataclass


@dataclass
class SimConfig:
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384
    max_model_len: int = 4096
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 4
    prefill_per_token_ms: float = 5.0
    decode_per_token_ms: float = 0.2
    decode_context_per_token_ms: float = 0.0
    scheduler_policy: str | None = None
    eos: int = 1


@dataclass
class RequestConfig:
    max_tokens: int = 256
    arrival_time_ms: float = 0.0

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.arrival_time_ms < 0.0:
            raise ValueError("arrival_time_ms must be non-negative")