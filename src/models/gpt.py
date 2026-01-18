from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.block import BlockConfig, TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int

    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024

    dropout: float = 0.0
    bias: bool = True
    activation: str = "gelu"
    norm_eps: float = 1e-5

    tie_weights: bool = True  # tie token embedding and LM head


class GPT(nn.Module):
    """
    Minimal GPT:
      tok_emb + pos_emb -> N * TransformerBlock -> final LN -> lm_head
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        block_cfg = BlockConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            bias=cfg.bias,
            activation=cfg.activation,
            norm_eps=cfg.norm_eps,
        )
        self.blocks = nn.ModuleList([TransformerBlock(block_cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Weight tying (common in GPT)
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # Simple, stable defaults. You can later switch to GPT-2 style init if desired.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """
        input_ids: (B, T) token ids
        targets:   (B, T) next-token targets (optional)
        return_attn: if True, return list of attention matrices per layer

        returns:
          logits: (B, T, vocab_size)
          optionally loss: scalar
          optionally attn_list: list length n_layers of (B, nh, T, T)
        """
        B, T = input_ids.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}")

        pos = torch.arange(0, T, device=input_ids.device, dtype=torch.long)  # (T,)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]          # (B,T,C)
        x = self.drop(x)

        attn_list = [] if return_attn else None
        for block in self.blocks:
            if return_attn:
                x, att = block(x, return_attn=True)
                attn_list.append(att)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        if targets is None:
            if return_attn:
                return logits, attn_list
            return logits

        # Standard LM loss: flatten B*T
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        if return_attn:
            return logits, loss, attn_list
        return logits, loss