from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLPConfig:
    d_model: int
    d_ff: int
    dropout: float = 0.0
    bias: bool = True
    activation: str = "gelu"  # "gelu" or "relu"


class FeedForward(nn.Module):
    """
    Standard Transformer MLP:
      x -> Linear(d_model,d_ff) -> activation -> Linear(d_ff,d_model) -> dropout
    """
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation == "gelu":
            x = F.gelu(self.fc1(x))
        elif self.cfg.activation == "relu":
            x = F.relu(self.fc1(x))
        else:
            raise ValueError(f"Unsupported activation: {self.cfg.activation}")
        x = self.fc2(x)
        x = self.drop(x)
        return x