import os

os.environ["HF_HUB_OFFLINE"] = "1"

from torch.profiler import profile, ProfilerActivity
from petvllm.cache.kv_cache_manager import KVCacheConfig
from petvllm.llm import LLM, VllmConfig

kv_config = KVCacheConfig(num_blocks=32, block_size=32)
vllm_config = VllmConfig(max_seq_len=1024, kv_config=kv_config)
llm = LLM("Qwen/Qwen3-0.6B", vllm_config)

prompt = "The capital of France is"

with profile(
    activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
) as prof:
    output = llm.generate(prompt, max_output_tokens=20, temperature=0.0)

print(f"{prompt}: {output}")
prof.export_chrome_trace("trace.json")
