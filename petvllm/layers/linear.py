import torch.nn as nn


# Simple wrapper around nn.Linear for now,
# We'll build on this to expand to column and row parallel Linear
# when implementing tensor parallelism
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
