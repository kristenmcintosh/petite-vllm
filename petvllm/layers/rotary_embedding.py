import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes and caches cos/sin tables for rotary position embeddings."""

    def __init__(self, head_dim, max_position=8192, base=10000.0):
        super().__init__()

        i = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (i / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        # positions: (seq_len,) tensor of integer position ids
        angle = torch.outer(positions.float(), self.inv_freq)
        # duplicate to full head_dim: [cos(θ₀)..cos(θ₆₃), cos(θ₀)..cos(θ₆₃)]
        cos = torch.cat([torch.cos(angle), torch.cos(angle)], dim=-1)
        sin = torch.cat([torch.sin(angle), torch.sin(angle)], dim=-1)
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to q and k tensors.

    Args:
        q: (batch, seq_len, num_heads, head_dim)
        k: (batch, seq_len, num_kv_heads, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Pairs dimension i with i+half (0↔64, 1↔65, etc.)
    """
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated
