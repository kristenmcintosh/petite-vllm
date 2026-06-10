import torch
import time
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

from petvllm.cache.kv_cache_manager import KVCacheConfig, KVCacheManager
from petvllm.config import Qwen3Config
from petvllm.models.qwen3 import Qwen3ForCausalLM
from petvllm.layers.sampler import sample


@dataclass
class VllmConfig:
    max_seq_len: int
    kv_config: KVCacheConfig


class LLM:
    def __init__(self, model_name: str, vllm_config: VllmConfig, device: str = "cpu"):
        self.device = device
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vllm_config = vllm_config

        model_config = Qwen3Config.from_pretrained(model_name)
        # Step 1: create our model (random weights)

        self.cache_manager = KVCacheManager(vllm_config.kv_config, model_config)
        self.model = Qwen3ForCausalLM(model_config, self.cache_manager)

        # Step 2: load pretrained weights from HuggingFace
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_state = hf_model.state_dict()

        # remap the hf state dict to match the petite-vllm format
        # 1. strip the hf model prefix
        # 2. skip lm, lm_head
        # 3. concatenate gate_proj and up_proj since those are fused in petvllm
        remapped = {}
        for key, value in hf_state.items():
            # skip lm_head (tied to embed_tokens)
            if key == "lm_head.weight":
                continue

            # strip "model." prefix
            new_key = key.removeprefix("model.")

            # fuse gate_proj + up_proj → gate_up_proj
            if "gate_proj.weight" in new_key:
                up_key = key.replace("gate_proj", "up_proj")
                remapped[new_key.replace("gate_proj", "gate_up_proj")] = torch.cat(
                    [value, hf_state[up_key]], dim=0
                )
                continue
            if "up_proj.weight" in new_key:
                continue  # already handled above

            remapped[new_key] = value

        self.model.load_state_dict(remapped, strict=False)

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

    def generate(
        self,
        prompt: str,
        max_output_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        # Step 1: tokenize the prompt
        # For each new token:
        # input_ids shape: (1, seq_len)

        toks = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = toks.shape[1]

        # validate that the input/output token count does not exceed max_seq_len
        # as we only allocate enough kv cache space for max_seq_len
        assert self.vllm_config.max_seq_len >= prompt_len + max_output_tokens, (
            f"Expected max_seq_len ({self.vllm_config.max_seq_len}) to be >= {prompt_len} + {max_output_tokens}"
        )
        for i in range(toks.shape[0]):
            self.cache_manager.register_sequence(i)

        start_prefill = time.time()
        first_tok = self.prefill(toks, temperature, top_k)
        elapsed_ttft = time.time() - start_prefill
        toks = torch.cat([toks, first_tok.unsqueeze(0)], dim=-1)

        start_decode = time.time()
        toks = self.decode(toks, max_output_tokens, temperature, top_k)
        elapsed = time.time() - start_decode

        # Also print tokens/sec for benchmarking
        print(f"time to first token: {elapsed_ttft}")
        print(f"{max_output_tokens / elapsed:.1f} tokens/sec")

        # Step 3: detokenize and return
        output = self.tokenizer.decode(toks[0, prompt_len:])
        return output
