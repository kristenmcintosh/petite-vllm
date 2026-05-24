import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from petvllm.llm import LLM

# Our model
llm = LLM("Qwen/Qwen3-0.6B")
our_output = llm.generate("The capital of France is", max_tokens=20, temperature=0.0)
print(f"Ours: {our_output}")

# HF reference
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")
with torch.no_grad():
    hf_out = hf_model.generate(input_ids, max_new_tokens=20, do_sample=False)
hf_text = tokenizer.decode(hf_out[0, input_ids.shape[1]:])
print(f"HF:   {hf_text}")
