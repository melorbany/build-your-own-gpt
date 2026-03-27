from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BinDatasetConfig:
    seq_len: int
    dtype: str = "uint16"


_DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


class PackedBinDataset(Dataset):
    """
    Reads:
      - ids.bin: contiguous token ids
      - ids.idx: uint64 offsets (token positions) for each sample; length = n_samples+1

    Returns (x, y) where:
      x: tokens[0:seq_len]
      y: tokens[1:seq_len+1]
    Each stored sample is seq_len+1 tokens; x and y are derived by slicing without wrapping.
    """
    def __init__(self, bin_file: Path, idx_file: Path, cfg: BinDatasetConfig):
        self.bin_file = Path(bin_file)
        self.idx_file = Path(idx_file)
        self.cfg = cfg

        if cfg.dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {cfg.dtype}")

        self.ids = np.memmap(self.bin_file, mode="r", dtype=_DTYPE_MAP[cfg.dtype])
        self.idx = np.memmap(self.idx_file, mode="r", dtype=np.uint64)

        if len(self.idx) < 2:
            raise ValueError("Index file is too small")

        self.n_samples = len(self.idx) - 1
        self.seq_len = int(cfg.seq_len)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.idx[i])
        end = int(self.idx[i + 1])
        seq = self.ids[start:end]

        if len(seq) != self.seq_len + 1:
            # In a correct pack, all sequences are seq_len+1 tokens.
            # If not, defensively crop (but this should rarely happen).
            seq = seq[: self.seq_len + 1]

        x = torch.from_numpy(seq[:-1].astype(np.int64))
        y = torch.from_numpy(seq[1:].astype(np.int64))
        return x, y