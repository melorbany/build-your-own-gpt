from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset

from src.utils.logging import get_logger

log = get_logger(__name__)


def write_jsonl(records: Iterable[dict], out_path: Path) -> Path:
    """
    Write iterable of dict records to JSONL (UTF-8), one JSON object per line.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    log.info("Wrote %d records -> %s", n, out_path)
    return out_path


def _limit_iter(ds, limit: Optional[int]):
    if limit is None:
        for x in ds:
            yield x
    else:
        for i, x in enumerate(ds):
            if i >= limit:
                break
            yield x


def download_tinystories_jsonl(
    out_path: Path,
    *,
    split: str = "train",
    limit: Optional[int] = None,
    dataset_name: str = "roneneldan/TinyStories",
    text_field: str = "text",
) -> Path:
    """
    Download TinyStories using HuggingFace datasets and write JSONL with schema:
      {"source":"tinystories","split": "...", "text": "..."}
    """
    log.info("Loading dataset=%s split=%s", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    def records():
        for ex in _limit_iter(ds, limit):
            text = ex.get(text_field)
            if not text:
                continue
            yield {"source": "tinystories", "split": split, "text": text}

    return write_jsonl(records(), out_path)