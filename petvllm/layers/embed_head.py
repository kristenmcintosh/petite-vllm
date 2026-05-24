import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabEmbedding(nn.Embedding):
    def __init__(self, vocab_size, hidden_size):
        # Simple wrapper for now, will come back to this
        # when implementing parallelism
        super().__init__(vocab_size, hidden_size)


class LMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, weight=None):
        super().__init__()
        if weight is not None:
            # in practice, well always pass in embedding weight,
            # this is a fall back for some (older) models that have
            # a seperate lm head weight
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def forward(self, x):
        # LMHead is just a linear layer, mm with no bias
        return F.linear(x, self.weight)
