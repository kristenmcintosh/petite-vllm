import torch

from petvllm.cache.kv_cache_manager import KVCacheManager
from petvllm.config import ModelConfig
from petvllm.layers.sampler import sample
from petvllm.models import MODEL_REGISTRY


class ModelRunner:
    def __init__(
        self,
        model_name: str,
        architecture: str,
        model_config: ModelConfig,
        cache_manager: KVCacheManager,
        device: str = "cpu",
    ):
        model_cls = MODEL_REGISTRY[architecture]
        self.cache_manager = cache_manager
        self.model = model_cls(model_config, cache_manager)
        self.model.load_weights(model_name)
        self.model.to(device)
        self.model.eval()

    def prefill(self, toks: torch.Tensor, temperature: float, top_k: int):
        for seq_id in range(toks.shape[0]):
            self.cache_manager.update_block_tables(seq_id, toks.shape[1])
        positions = torch.arange(toks.shape[1])
        all_logits = self.model.forward(toks, positions)
        last_logits = all_logits[:, -1, :]
        return sample(last_logits, temperature, top_k)

    def decode(self, toks, max_output_tokens: int, temperature, top_k):
        for _ in range(1, max_output_tokens):
            for seq_id in range(toks.shape[0]):
                self.cache_manager.update_block_tables(seq_id, 1)
            positions = torch.tensor([toks.shape[1] - 1])
            all_logits = self.model.forward(toks[:, -1:], positions)
            last_logits = all_logits[:, -1, :]
            new_tkn = sample(last_logits, temperature, top_k)
            toks = torch.cat([toks, new_tkn.unsqueeze(0)], dim=-1)
        return toks
