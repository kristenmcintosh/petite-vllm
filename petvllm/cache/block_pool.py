from dataclasses import dataclass


@dataclass
class Block:
    size: int
    id: int


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks_queue = [
            Block(self.block_size, i) for i in range(self.num_blocks)
        ]

    def allocate(self) -> Block:
        """
        Allocate a block, by removing from the free block queue

        """
        block = self.free_blocks_queue.pop()
        return block

    def free(self, block: Block):
        """Return a block to the free block queue"""
        self.free_blocks_queue.append(block)
