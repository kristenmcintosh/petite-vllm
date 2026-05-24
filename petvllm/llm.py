import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from petvllm.config import Qwen3Config
from petvllm.models.qwen3 import Qwen3ForCausalLM
from petvllm.layers.sampler import sample


class LLM:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = Qwen3Config.from_pretrained(model_name)

        # Step 1: create our model (random weights)
        self.model = Qwen3ForCausalLM(self.config)

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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        # Step 1: tokenize the prompt
        # input_ids shape: (1, seq_len)
        toks = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = toks.shape[1]
        # Step 2: generation loop
        # For each new token:
        #   - create positions tensor: [0, 1, 2, ..., current_seq_len - 1]
        #   - forward pass through model → logits (1, seq_len, vocab_size)
        #   - take only the LAST token's logits → (1, vocab_size)
        #   - sample → next token id
        #   - append to input_ids
        #   - stop if EOS token
        # Note: without KV cache, we re-run the FULL sequence every step (slow!)
        start = time.time()
        for tok_id in range(max_tokens):
            positions = torch.arange(toks.shape[1]) # get positions based on current_seq_len

            all_logits = self.model.forward(toks, positions)
            last_logits = all_logits[:, -1, :]
            new_tkn = sample(last_logits, temperature, top_k)
            toks = torch.cat([toks, new_tkn.unsqueeze(0)], dim=-1)
        # Step 3: detokenize and return
        elapsed = time.time() - start
        # Also print tokens/sec for benchmarking
        print(f"{max_tokens / elapsed:.1f} tokens/sec")
        output = self.tokenizer.decode(toks[0, prompt_len:])
        return output
