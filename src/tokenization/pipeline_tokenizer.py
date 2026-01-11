from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.tokenization.train_tokenizer import train_bpe_tokenizer


def run_tokenizer_training(cfg: Dict[str, Any]) -> Path:
    return train_bpe_tokenizer(cfg)