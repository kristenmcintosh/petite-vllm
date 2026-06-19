import torch
from dataclasses import dataclass
from transformers import AutoTokenizer

from petvllm.cache.kv_cache_manager import KVCacheConfig, KVCacheManager
from petvllm.config import Qwen3Config
from petvllm.engine.model_runner import ModelRunner
from petvllm.metrics import Metrics


@dataclass
class VllmConfig:
    max_seq_len: int
    kv_config: KVCacheConfig


class LLM:
    def __init__(self, model_name: str, vllm_config: VllmConfig, device: str = "cpu"):
        self.vllm_config = vllm_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_config = Qwen3Config.from_pretrained(model_name)
        self.cache_manager = KVCacheManager(vllm_config.kv_config, model_config)
        self.model_runner = ModelRunner(model_name, "qwen3", model_config, self.cache_manager, device)

    def generate(
        self,
        prompt: str,
        max_output_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        toks = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = toks.shape[1]

        assert self.vllm_config.max_seq_len >= prompt_len + max_output_tokens, (
            f"Expected max_seq_len ({self.vllm_config.max_seq_len}) to be >= {prompt_len} + {max_output_tokens}"
        )

        metrics = Metrics()
        metrics.latency.prompt_tokens = prompt_len
        metrics.latency.start_e2e()

        for i in range(toks.shape[0]):
            self.cache_manager.register_sequence(i)

        metrics.latency.start_prefill()
        first_tok = self.model_runner.prefill(toks, temperature, top_k)
        metrics.latency.end_prefill()
        toks = torch.cat([toks, first_tok.unsqueeze(0)], dim=-1)

        metrics.latency.start_decode()
        toks = self.model_runner.decode(toks, max_output_tokens, temperature, top_k)
        metrics.latency.end_decode(max_output_tokens)

        metrics.latency.end_e2e()
        metrics.cache.record(
            self.cache_manager.block_pool.num_used,
            self.cache_manager.block_pool.num_blocks,
        )

        print(metrics.summary())

        output = self.tokenizer.decode(toks[0, prompt_len:])
        return output
