import os

os.environ["HF_HUB_OFFLINE"] = "1"

from petvllm.llm import LLM, VllmConfig

vllm_config = VllmConfig(max_seq_len=1024)
llm = LLM("Qwen/Qwen3-0.6B", vllm_config)

prompt = "The capital of France is"
output = llm.generate(prompt, max_output_tokens=20, temperature=0.0)
print(f"{prompt}: {output}")
