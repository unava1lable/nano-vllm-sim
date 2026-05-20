from nanovllm.config import SimConfig
from nanovllm.engine.sequence import Sequence


class SimModelRunner:
    def __init__(self, simconfig: SimConfig):
        self.config = simconfig
        self.block_size = simconfig.kvcache_block_size


    def run(self, seqs: list[Sequence], is_prefill: bool):
        total_time = 0.0
        for seq in seqs:
            time_ms = 0.0
            if is_prefill:
                prefilled_tokens = seq.num_scheduled_tokens
                time_ms = prefilled_tokens * self.config.prefill_per_token_ms
                print(f"Sequence {seq.seq_id} prefill time: {time_ms} ms")
            else:
                decode_tokens = seq.num_scheduled_tokens
                context_tokens = len(seq)
                time_ms = (
                    decode_tokens * self.config.decode_per_token_ms
                    + context_tokens * self.config.decode_context_per_token_ms
                )
                print(f"Sequence {seq.seq_id} decode time: {time_ms} ms")
            total_time += time_ms    

        token_id = 2 if self.config.eos != 2 else 3
        return [token_id for _ in range(len(seqs))], total_time

