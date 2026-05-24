import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def forward(self, x):
        # x shape: (..., 2 * intermediate_size)
        # Split in half along last dim, SiLU the first half, multiply by second half
        gate, value = torch.chunk(x, 2, dim=-1)
        return F.silu(gate) * value
