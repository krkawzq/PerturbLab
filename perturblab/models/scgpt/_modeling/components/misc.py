"""Miscellaneous utility components for scGPT."""

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Similarity",
    "generate_square_subsequent_mask",
]


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class Similarity(nn.Module):
    """Dot product or cosine similarity."""

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
