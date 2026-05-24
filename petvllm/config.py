from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Qwen3Config:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Qwen3Config":
        hf = AutoConfig.from_pretrained(model_name)
        return cls(
            hidden_size=hf.hidden_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=hf.head_dim,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            vocab_size=hf.vocab_size,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=hf.rope_parameters["rope_theta"],
            max_position_embeddings=hf.max_position_embeddings,
            tie_word_embeddings=hf.tie_word_embeddings,
        )
