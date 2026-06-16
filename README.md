# petite-vllm

A from-scratch LLM serving engine, built lesson by lesson. Each module implements a core concept from the vLLM architecture — PagedAttention, continuous batching, Triton kernels, tensor parallelism — with the goal of understanding every layer of the inference stack.

Target model: **Qwen3-0.6B** on Apple Silicon (CPU), with GPU lessons on rented NVIDIA instances.

## What's implemented

| Lesson | What it covers | Key files |
|--------|---------------|-----------|
| 1. Autoregressive Generation | Hand-built transformer forward pass, weight loading from HuggingFace, temperature sampling | `models/qwen3.py`, `layers/*`, `llm.py` |
| 2. KV Cache | Prefill/decode split, pre-allocated cache, O(n^2) → O(n) | `cache/kv_cache.py` |
| 3. Paged KV Cache | Block pool, flat cache tensor, block tables, slot mapping — modeled on vLLM's v1 design | `cache/block_pool.py`, `cache/block_kv_cache.py`, `cache/kv_cache_manager.py` |

## What's next

4. **Continuous Batching** — scheduler, sequence tracking, chunked prefill
5. **API Server** — FastAPI, SSE streaming, async engine
6. **Triton Kernels** — fused RMSNorm, paged attention decode, CUDA graphs
7. **Model Parallelism** — tensor parallel linear layers, pipeline parallelism
8. **Quantization** — INT8 weight-only and dynamic activation quantization

## Project structure

```
petvllm/
├── config.py                  # Qwen3 model config
├── llm.py                     # Top-level generate API (prefill/decode loop)
├── models/
│   └── qwen3.py               # Full Qwen3 model (attention, MLP, decoder layers)
├── layers/
│   ├── activation.py           # SiluAndMul (SwiGLU)
│   ├── embed_head.py           # Embedding + LM head (weight-tied)
│   ├── layernorm.py            # RMSNorm
│   ├── linear.py               # Linear layer
│   ├── rotary_embedding.py     # RoPE (half-split)
│   └── sampler.py              # Temperature + top-k sampling
├── cache/
│   ├── kv_cache.py             # Lesson 2: naive pre-allocated KV cache
│   ├── block_pool.py           # Free block queue (allocate/free)
│   ├── block_kv_cache.py       # Flat paged tensor with index_put_ writes
│   └── kv_cache_manager.py     # Block tables, slot mapping, orchestration
├── engine/                     # (Lesson 4)
├── server/                     # (Lesson 5)
└── quantization/               # (Lesson 8)
```

## Quick start

```bash
pip install -e .
```

```python
from petvllm.cache.kv_cache_manager import KVCacheConfig
from petvllm.llm import LLM, VllmConfig

kv_config = KVCacheConfig(num_blocks=32, block_size=32)
vllm_config = VllmConfig(max_seq_len=1024, kv_config=kv_config)
llm = LLM("Qwen/Qwen3-0.6B", vllm_config)

output = llm.generate("The capital of France is", max_output_tokens=20, temperature=0.0)
print(output)
```

## Dependencies

- Python 3.10+
- PyTorch 2.4+
- `transformers` (tokenizer + weight loading)
- `safetensors`
