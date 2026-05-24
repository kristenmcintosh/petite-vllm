## Lesson 1: Autoregressive Generation

### RMSNorm vs LayerNorm
- RMSNorm is simpler than LayerNorm — no mean subtraction, no bias. Just normalize by root-mean-square and scale by a learnable weight.
- Formula: `x * weight / sqrt(mean(x^2) + eps)`
- The `mean(-1, keepdim=True)` is important — normalize per-token (last dim only), keepdim for broadcasting.

### nn.Parameter vs register_buffer
- `nn.Parameter`: learnable weight, included in `model.parameters()`, gets gradients, updated by optimizer.
- `register_buffer("name", tensor)`: fixed tensor stored on the module. Moves to GPU with `.to(device)`, saved in `state_dict()`, but excluded from `model.parameters()` so optimizers ignore it.
- Both live in internal dicts on `nn.Module` (`_parameters` vs `_buffers`).
- Use `register_buffer` for precomputed values like RoPE frequencies.

### SwiGLU (SiluAndMul)
- Standard gated MLP activation used in most modern LLMs (Llama, Mistral, Qwen, Gemma — not Qwen-specific).
- The first linear layer projects to 2x intermediate size. Split in half: one half is the "gate" (passed through SiLU), the other is the "value". Output = SiLU(gate) * value.
- Fused into one projection (`gate_up_proj`) because one big matmul is faster than two smaller ones.
- Replaced ReLU/GELU around 2022-2023 after a 2020 Google paper showed gated activations consistently outperform ungated.

### RoPE (Rotary Position Embeddings)
- Encodes token position by *rotating* Q and K vectors by an angle proportional to position.
- Attention dot product then depends only on *relative distance* between tokens, not absolute position.
- Unlike additive position embeddings (original transformer), this bakes position into the attention scores naturally.
- Precompute inverse frequency vector: `inv_freq[i] = 1 / (base ^ (2i / head_dim))` — store as buffer, not parameter.
- **Half-split vs even/odd**: Two valid ways to pair dimensions for rotation. HF (and Qwen3's trained weights) uses the half-split approach — pairs dimension `i` with `i + head_dim/2` (0 with 64, 1 with 65, etc.). The alternative even/odd approach pairs 0 with 1, 2 with 3, etc. Must match what the model was trained with or attention scores become meaningless.
- The `rotate_half` function: split into first half and second half, return `cat(-second, first)`. Then: `q_rotated = q * cos + rotate_half(q) * sin`.
- `cos` and `sin` are duplicated to full `head_dim` size: `cat([cos, cos])` so they broadcast against the full Q/K vectors.

### QK Normalization (Qwen3-specific)
- RMSNorm applied to Q and K *per-head* after projection but before RoPE.
- Norm size is `head_dim` (not `hidden_size`) since it operates on individual head vectors.
- Stabilizes attention scores — prevents Q/K magnitudes from growing too large.

### Grouped-Query Attention (GQA)
- Fewer KV heads than Q heads — Qwen3-0.6B has 16 Q heads, 8 KV heads.
- KV heads are repeated via `torch.repeat_interleave` to match Q head count before attention.
- Saves memory and compute on KV projections while maintaining Q expressiveness.

### Weight Tying
- Embedding table and LM head share the exact same weight matrix in memory — not a copy, the same tensor.
- In Qwen3: embedding is (vocab_size, hidden_size). LM head needs the same shape to project back to vocab scores. Sharing avoids storing two 150K x hidden_size matrices.
- Makes conceptual sense: the vector that represents "cat" in embedding space should be the same vector that predicts "cat" at the output.
- Implementation: LM head uses `F.linear(x, self.weight)` with a borrowed weight rather than `nn.Linear` which would create its own parameter.

### Weight Loading & Remapping
- HF and our model use different naming conventions. Three remappings needed:
  1. Strip `model.` prefix (HF wraps layers under `model.`)
  2. Fuse `gate_proj` + `up_proj` into `gate_up_proj` via `torch.cat([gate, up], dim=0)`
  3. Skip `lm_head.weight` (tied to `embed_tokens`) and `inv_freq` buffers (precomputed)
- `load_state_dict(remapped, strict=False)` — `strict=False` allows missing keys (inv_freq buffers, tied lm_head).

### Generation Loop
- Tokenize prompt → loop (forward pass → sample last logit → cat new token → repeat) → detokenize.
- Without KV cache, the full sequence is re-run every step — O(n^2) total work for n tokens. Lesson 2 fixes this.
- `torch.arange(toks.shape[1])` creates positions matching the current sequence length, recomputed each step.
- `torch.cat([toks, new_tkn.unsqueeze(0)], dim=-1)` appends new token. Need `unsqueeze(0)` to go from shape (1,) to (1, 1) to match the 2D toks tensor.

### Sampling
- Temperature = 0: greedy (argmax). Can't divide by 0, so special-cased.
- Temperature > 0: divide logits by temperature, apply top-k mask, softmax, multinomial sample.
- Top-k: keep only the k highest-scoring tokens, set rest to `-inf`. The k-th value is `vals[:, -1:]` (keepdim for broadcasting).
- `squeeze(-1)` removes the trailing dimension from multinomial output: (batch, 1) → (batch,).
- Use dim `-1` to operate on the last dimension — batch dimension should always survive.

### Embedding & LM Head
- Embedding: down-projects from vocab_size to hidden_size (token IDs → vectors).
- LM head: up-projects from hidden_size to vocab_size, outputting logits (scores) for each token.
- Logits are raw scores, not probabilities — softmax converts them to probabilities during sampling.

### torch.nn.functional
- Stateless versions of common operations (no learnable parameters).
- Use `F.silu(x)` when you just need to apply the function vs `nn.SiLU()` when you need a module.
- Key functions: `F.silu()`, `F.scaled_dot_product_attention()`, `F.linear()`, `F.softmax()`, `F.cross_entropy()`.

### Debugging: RoPE Bug
- Wrong `rope_theta` (10000 vs 1000000) causes rotation frequencies to be 100x too fast. Positions that should look similar to the model look wildly different. The model falls into degenerate repetition loops because attention scores become meaningless.
- Even/odd vs half-split interleaving: using the wrong pairing pattern produces similar symptoms — model outputs nonsense or repetition because the learned attention patterns don't match the rotation scheme.
- **Debugging approach**: Compare logits from our model vs HF reference on the same input. If they differ, binary-search inward (embeddings → first layer → attention internals → RoPE) until you find the first point of divergence. Shape mismatches are often the fastest signal.
- Small models (0.6B) are especially brittle — even a typo in the prompt can cause repetition loops that larger models would handle gracefully.
