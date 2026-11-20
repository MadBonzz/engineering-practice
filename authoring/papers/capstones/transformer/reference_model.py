import math
import torch
from torch import nn


def positions(length, width, like):
    steps = torch.arange(length, device=like.device, dtype=like.dtype)[:, None]
    dims = torch.arange(0, width, 2, device=like.device, dtype=like.dtype)
    phase = steps * torch.exp(-math.log(10000.0) * dims / width)
    result = like.new_zeros(length, width)
    result[:, 0::2], result[:, 1::2] = phase.sin(), phase.cos()
    return result


class DecoderLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])

    def forward(self, target, memory):
        """Decode [B,T,D] against [B,S,D]. Use post-norm after each residual branch.
        In self-attention forbid keys strictly to the right of the query.
        Cross-attention queries come from the updated target. All source keys are visible.
        Finally apply the feed-forward sublayer. Return [B,T,D].
        """

        length = target.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=target.device).triu(1)
        update, _ = self.self_attn(target, target, target, attn_mask=mask, need_weights=False)
        target = self.norms[0](target + update)
        update, _ = self.cross_attn(target, memory, memory, need_weights=False)
        target = self.norms[1](target + update)
        return self.norms[2](target + self.ff(target))



class Model(nn.Module):
    def __init__(self, vocab=12, width=16, heads=2):
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(vocab, width)
        self.encoder = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0, batch_first=True)
        self.decoder = DecoderLayer(width, heads)
        self.readout = nn.Linear(width, vocab)

    def embed(self, ids):
        embedded = self.embedding(ids) * math.sqrt(self.width)
        return embedded + positions(ids.shape[1], self.width, embedded)

    def forward(self, source, target_prefix):
        memory = self.encoder(self.embed(source))
        return self.readout(self.decoder(self.embed(target_prefix), memory))
