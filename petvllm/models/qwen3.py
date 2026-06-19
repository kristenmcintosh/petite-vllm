import torch
import torch.nn as nn
import torch.nn.functional as F

from petvllm.cache.kv_cache_manager import KVCacheManager
from petvllm.config import Qwen3Config
from petvllm.layers.activation import SiluAndMul
from petvllm.layers.embed_head import LMHead, VocabEmbedding
from petvllm.layers.layernorm import RMSNorm
from petvllm.layers.linear import Linear
from petvllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kv_cache: KVCacheManager | None = None,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # kv proj output based on num kvheads (less than q heads)
        kv_proj_size = self.num_kv_heads * self.head_dim
        self.k_proj = Linear(config.hidden_size, kv_proj_size)
        self.v_proj = Linear(config.hidden_size, kv_proj_size)

        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size)

        self.rotary_embedding = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

        self.layer_idx = layer_idx
        self.kv_cache = kv_cache

    def forward(self, x, positions):
        batch, seq, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # output of qkv projections will be
        # batch, seq, and compact view of all heads
        # (head_dim * num_heads).
        # Need to reshape to separate heads
        q = q.view(batch, seq, self.num_heads, self.head_dim)
        k = k.view(batch, seq, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rotary_embedding(positions)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.kv_cache is not None:
            full_ks, full_vs = [], []
            for seq_id in range(batch):
                full_k, full_v = self.kv_cache.update(
                    self.layer_idx, seq_id, k[seq_id], v[seq_id]
                )
                full_ks.append(full_k)
                full_vs.append(full_v)

            k = torch.stack(full_ks, dim=0)
            v = torch.stack(full_vs, dim=0)

        k = torch.repeat_interleave(k, self.num_kv_groups, dim=2)
        v = torch.repeat_interleave(v, self.num_kv_groups, dim=2)

        # F.scaled_dot_product_attention expects shapes to be batch, num_heads, seq, head_dim
        # so we need to transpose to swap dims 1 and 2 here
        q = q.transpose(1, 2)  # (batch, num_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # print(q.shape, k.shape, v.shape)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape == k.shape)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # gate_up_proj: hidden_size → 2 * intermediate_size (fused gate + up)
        self.gate_up_proj = Linear(self.hidden_size, 2 * self.intermediate_size)
        # down_proj: intermediate_size → hidden_size
        self.down_proj = Linear(self.intermediate_size, self.hidden_size)
        # activation: SiluAndMul splits gate_up output and applies gating
        self.activation = SiluAndMul()

    def forward(self, x):
        # gate_up_proj → activation → down_proj
        return self.down_proj(self.activation(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int, kv_cache: KVCacheManager):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, layer_idx, kv_cache)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x, positions):
        # pre-norm → attention → residual
        x = x + self.self_attn(self.input_layernorm(x), positions)
        # pre-norm → mlp → residual
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config, kv_cache: KVCacheManager):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx, kv_cache)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = LMHead(
            config.hidden_size,
            config.vocab_size,
            weight=self.embed_tokens.weight if config.tie_word_embeddings else None,
        )

    def load_weights(self, model_name: str):
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_state = hf_model.state_dict()

        remapped = {}
        for key, value in hf_state.items():
            if key == "lm_head.weight":
                continue

            new_key = key.removeprefix("model.")

            if "gate_proj.weight" in new_key:
                up_key = key.replace("gate_proj", "up_proj")
                remapped[new_key.replace("gate_proj", "gate_up_proj")] = torch.cat(
                    [value, hf_state[up_key]], dim=0
                )
                continue
            if "up_proj.weight" in new_key:
                continue

            remapped[new_key] = value

        self.load_state_dict(remapped, strict=False)

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)  # go from vocab_size -> hidden_size
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        return self.lm_head(x)  # go back from hidden_size -> vocab_size
