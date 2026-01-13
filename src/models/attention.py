from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    d_model: int
    n_heads: int
    dropout: float = 0.0
    bias: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


def build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # mask shape: (1, 1, T, T) to broadcast over (B, nh, T, T)
    m = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    return m.view(1, 1, T, T)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer("_mask", torch.empty(0), persistent=False)

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self._mask.numel() == 0 or self._mask.shape[-1] < T or self._mask.device != device:
            self._mask = build_causal_mask(T, device)
        return self._mask[:, :, :T, :T]

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: (B, T, d_model)
        returns: y (B, T, d_model)
        optionally returns attention weights (B, nh, T, T)
        """
        B, T, C = x.shape
        nh = self.cfg.n_heads
        hs = self.cfg.head_dim

        qkv = self.qkv(x)                        # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)           # each (B, T, C)

        # (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # attention scores: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (hs ** -0.5)

        mask = self._get_mask(T, x.device)
        att = att.masked_fill(~mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                               # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_drop(self.proj(y))

        if return_attn:
            return y, att
        return y