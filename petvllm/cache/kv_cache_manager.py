import torch
from dataclasses import dataclass

from petvllm.cache.block_pool import Block
from petvllm.config import Qwen3Config
from petvllm.cache.block_kv_cache import BlockKVCache
from petvllm.cache.block_pool import BlockPool

from typing import NamedTuple


class BlockMapping(NamedTuple):
    block_ids: torch.Tensor
    offsets: torch.Tensor


class BlockTableEntry(NamedTuple):
    blocks: list[Block]
    num_tokens: int


@dataclass
class KVCacheConfig:
    block_size: int
    num_blocks: int


class KVCacheManager:
    def __init__(self, kv_config: KVCacheConfig, model_config: Qwen3Config):
        self.block_size = kv_config.block_size
        self.kv_cache = BlockKVCache(
            model_config.num_hidden_layers,
            model_config.num_key_value_heads,
            model_config.head_dim,
            kv_config.num_blocks,
            kv_config.block_size,
        )

        self.block_pool = BlockPool(kv_config.num_blocks, kv_config.block_size)
        self.block_table: dict[int, BlockTableEntry] = {}

    def register_sequence(self, seq_id):
        block = self.block_pool.allocate()
        self.block_table[seq_id] = BlockTableEntry([block], 0)

    def update_block_tables(self, seq_id, num_new_tokens):
        # Get current active block for this sequence id
        blocks, num_tokens = self.block_table[seq_id]
        block_ids = []
        offsets = []

        # for each new token, calculate its offset
        # determine if a new block is needed
        # add the tokens block id and offset to list
        for i in range(num_new_tokens):
            pos = num_tokens + i
            offset = pos % self.block_size

            if offset == 0 and pos > 0:
                blocks.append(self.block_pool.allocate())

            block_ids.append(blocks[pos // self.block_size].id)
            offsets.append(offset)

        # Update kv cache by passing block ids and offsets
        self.mapping = BlockMapping(torch.tensor(block_ids), torch.tensor(offsets))
        self.block_table[seq_id] = BlockTableEntry(blocks, num_tokens + num_new_tokens)

    def update(self, layer_idx: int, seq_id: int, k: torch.Tensor, v: torch.Tensor):
        """Store new K and V entries for a given layer.

        Args:
            layer_idx: which transformer layer
            seq_id: which sequence
            k: new key tensor - shape (seq_len, num_kv_heads, head_dim)
            v: new value tensor - same shape as k

        Returns:
            full_k, full_v: the complete cached K and V up to current position
        """

        self.kv_cache.update(layer_idx, self.mapping, k, v)

        # read back the full kv cache
        full_k, full_v = self.read(layer_idx, seq_id)
        return full_k, full_v

    def read(self, layer_idx, seq_id):
        blocks, num_tokens = self.block_table[seq_id]
        block_ids = torch.tensor([b.id for b in blocks])
        return self.kv_cache.read(layer_idx, block_ids, num_tokens)
