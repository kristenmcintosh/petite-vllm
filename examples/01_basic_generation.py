import os

os.environ["HF_HUB_OFFLINE"] = "1"

from petvllm.llm import LLM

llm = LLM("Qwen/Qwen3-0.6B")

prompt = "The capital of France is"
output = llm.generate(prompt, max_tokens=20, temperature=0.0)
print(f"{prompt}: {output}")
