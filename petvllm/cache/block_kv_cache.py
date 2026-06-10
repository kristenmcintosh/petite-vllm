import torch


class BlockKVCache:
    def __init__(
        self,
        num_layers,
        num_kv_heads,
        head_dim,
        num_blocks,
        block_size,
        dtype=torch.float32,
    ):
        # TODO: Pre-allocate K and V tensors for all layers
        self.k = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype
        )

        self.v = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype
        )

        self.num_layers = num_layers
        self.pos_id = 0

    @property
    def seq_len(self):
        """How many tokens have been stored so far."""
        return self.pos_id

    def update(self, layer_idx, block_mapping, k, v):
        """Store new K and V entries for a given layer.

        Args:
            layer_idx: which transformer layer
            block_mapping: tuple of (block_ids, offsets) tensors for index_put_
            k: new key tensor - shape (N, num_kv_heads, head_dim), where N is number of tokens
            v: new value tensor - same shape as k
        """

        # Note: We assume that the block mapping is valid, which is handled byhte cache manager
        self.k[layer_idx].index_put_((block_mapping[0], block_mapping[1]), k)
        self.v[layer_idx].index_put_((block_mapping[0], block_mapping[1]), v)

    def read(self, layer_idx, block_mapping, seq_len):
        """Read the full K/V cache per sequence

        Args:
            layer_idx; which transformer layer
            block_mapping: tensor of (block_ids, )
            seq_len: current seq_len

        Returns:
            full_k, full_v (seq_len, num_kv_heads, head_dim)

        """
        full_k = self.k[layer_idx, block_mapping].reshape(
            -1, self.k.shape[-2], self.k.shape[-1]
        )[:seq_len]
        full_v = self.v[layer_idx, block_mapping].reshape(
            -1, self.v.shape[-2], self.v.shape[-1]
        )[:seq_len]
        return full_k, full_v
