import time
from dataclasses import dataclass, field


@dataclass
class LatencyMetrics:
    ttft: float = 0.0
    tpot: float = 0.0
    e2e_latency: float = 0.0

    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_per_sec: float = 0.0

    _prefill_start: float = field(default=0.0, repr=False)
    _decode_start: float = field(default=0.0, repr=False)
    _e2e_start: float = field(default=0.0, repr=False)

    def start_e2e(self):
        self._e2e_start = time.perf_counter()

    def start_prefill(self):
        self._prefill_start = time.perf_counter()

    def end_prefill(self):
        self.ttft = time.perf_counter() - self._prefill_start

    def start_decode(self):
        self._decode_start = time.perf_counter()

    def end_decode(self, num_output_tokens: int):
        decode_elapsed = time.perf_counter() - self._decode_start
        self.output_tokens = num_output_tokens
        self.tpot = decode_elapsed / max(num_output_tokens, 1)
        self.tokens_per_sec = num_output_tokens / max(decode_elapsed, 1e-9)

    def end_e2e(self):
        self.e2e_latency = time.perf_counter() - self._e2e_start

    def summary(self) -> str:
        return (
            f"TTFT:              {self.ttft * 1000:.1f} ms\n"
            f"TPOT:              {self.tpot * 1000:.2f} ms\n"
            f"E2E latency:       {self.e2e_latency * 1000:.1f} ms\n"
            f"Throughput:        {self.tokens_per_sec:.1f} tokens/sec\n"
            f"Tokens:            {self.prompt_tokens} prompt + {self.output_tokens} output"
        )


@dataclass
class CacheMetrics:
    blocks_used: int = 0
    blocks_total: int = 0
    utilization: float = 0.0

    def record(self, blocks_used: int, blocks_total: int):
        self.blocks_used = blocks_used
        self.blocks_total = blocks_total
        self.utilization = blocks_used / max(blocks_total, 1)

    def summary(self) -> str:
        return f"Cache:             {self.blocks_used}/{self.blocks_total} blocks ({self.utilization:.0%})"


@dataclass
class Metrics:
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    cache: CacheMetrics = field(default_factory=CacheMetrics)

    def summary(self) -> str:
        return f"{self.latency.summary()}\n{self.cache.summary()}"
