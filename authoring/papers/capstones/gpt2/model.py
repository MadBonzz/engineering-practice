import math
import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, width=24, heads=3):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        """Map [B,T,D] to [B,T,D]. Split projected Q/K/V into H heads.
        Scale dot products by sqrt(D/H), block keys later than the query with
        negative infinity, and normalize each query over keys. Concatenate
        the head results in their original order before the output projection.
        """
        raise NotImplementedError("Complete this method using its contract.")


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attn = CausalAttention(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        """Apply pre-normalized causal attention and MLP, each with its own residual.
        Normalize the updated residual stream before the second sublayer.
        """
        raise NotImplementedError("Complete this method using its contract.")


class Model(nn.Module):
    def __init__(self, vocab=13, width=24, context=16):
        super().__init__()
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(context, width)
        self.layers = nn.Sequential(Block(width), Block(width))
        self.norm = nn.LayerNorm(width)

    def forward(self, ids):
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        return self.norm(self.layers(x)) @ self.tokens.weight.T
