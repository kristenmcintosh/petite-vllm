import torch


class KVCache:
    def __init__(
        self,
        num_layers,
        num_kv_heads,
        head_dim,
        max_seq_len,
        batch=1,
        dtype=torch.float32,
    ):
        # TODO: Pre-allocate K and V tensors for all layers
        self.k = torch.zeros(
            num_layers, batch, max_seq_len, num_kv_heads, head_dim, dtype=dtype
        )

        self.v = torch.zeros(
            num_layers, batch, max_seq_len, num_kv_heads, head_dim, dtype=dtype
        )

        self.num_layers = num_layers
        self.pos_id = 0

    @property
    def seq_len(self):
        """How many tokens have been stored so far."""
        return self.pos_id

    def update(self, layer_idx, k, v):
        """Store new K and V entries for a given layer.

        Args:
            layer_idx: which transformer layer
            k: new key tensor - shape (batch, seq_len, num_kv_heads, head_dim)
            v: new value tensor - same shape as k

        Returns:
            full_k, full_v: the complete cached K and V up to current position
        """
        next_pos = self.pos_id + k.shape[1]
        self.k[layer_idx, :, self.pos_id : next_pos, :, :] = k
        self.v[layer_idx, :, self.pos_id : next_pos, :, :] = v

        if self.num_layers == layer_idx + 1:
            self.pos_id = next_pos

        return self.k[layer_idx, :, :next_pos, :, :], self.v[
            layer_idx, :, :next_pos, :, :
        ]
