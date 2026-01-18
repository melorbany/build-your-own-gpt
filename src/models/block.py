from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.attention import AttentionConfig, CausalSelfAttention
from src.models.mlp import MLPConfig, FeedForward


@dataclass
class BlockConfig:
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0
    bias: bool = True
    activation: str = "gelu"  # passed to MLP
    norm_eps: float = 1e-5


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
      x = x + Attn(LN(x))
      x = x + MLP(LN(x))

    Matches GPT-style blocks (no cross-attn).
    """
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg

        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(
            AttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                bias=cfg.bias,
            )
        )

        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = FeedForward(
            MLPConfig(
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                bias=cfg.bias,
                activation=cfg.activation,
            )
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: (B, T, d_model)
        return_attn: if True, also returns attention matrix from the attention sublayer

        returns:
          x: (B, T, d_model)
          optionally att: (B, nh, T, T)
        """
        if return_attn:
            a_out, att = self.attn(self.ln1(x), return_attn=True)
            x = x + a_out
            x = x + self.mlp(self.ln2(x))
            return x, att

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x