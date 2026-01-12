from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer

from src.data.jsonl import read_jsonl
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PackConfig:
    seq_len: int = 512
    concat_docs: bool = True
    add_eos_between_docs: bool = True
    limit_rows: Optional[int] = None


def _dtype_for_vocab(vocab_size: int) -> np.dtype:
    if vocab_size <= 2**16:
        return np.uint16
    if vocab_size <= 2**32:
        return np.uint32
    return np.uint64


def pack_jsonl_to_bin(
    *,
    input_file: Path,
    tokenizer_file: Path,
    text_field: str,
    out_dir: Path,
    name: str,
    cfg: PackConfig,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(tokenizer_file))
    vocab_size = tok.get_vocab_size()
    dtype = _dtype_for_vocab(vocab_size)

    eos_id = tok.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("Tokenizer is missing [EOS] token id")

    bin_path = out_dir / f"{name}_ids.bin"
    idx_path = out_dir / f"{name}_ids.idx"
    stats_path = out_dir / f"{name}_stats.json"

    seq_len = int(cfg.seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    # We will write:
    # - bin: raw token ids, contiguous (dtype)
    # - idx: uint64 start offsets in number of tokens for each sequence (length = n_sequences+1)
    offsets: List[int] = [0]
    buffer: List[int] = []
    n_rows = 0
    n_docs = 0
    n_tokens_total = 0
    n_sequences = 0

    def flush_sequence(seq: List[int], f_bin) -> None:
        nonlocal n_tokens_total, n_sequences
        arr = np.asarray(seq, dtype=dtype)
        arr.tofile(f_bin)
        n_tokens_total += len(seq)
        n_sequences += 1
        offsets.append(n_tokens_total)

    with bin_path.open("wb") as f_bin:
        for rec in read_jsonl(input_file):
            n_rows += 1
            if cfg.limit_rows is not None and n_rows > int(cfg.limit_rows):
                break

            text = rec.get(text_field, "")
            if not isinstance(text, str) or not text:
                continue

            # Important: our tokenizer already adds BOS/EOS via TemplateProcessing.
            # For packing, we typically want raw text -> ids WITHOUT padding/truncation.
            enc = tok.encode(text)
            ids = enc.ids

            if cfg.concat_docs:
                buffer.extend(ids)
                n_docs += 1
                if cfg.add_eos_between_docs:
                    buffer.append(eos_id)

                while len(buffer) >= seq_len:
                    seq = buffer[:seq_len]
                    flush_sequence(seq, f_bin)
                    buffer = buffer[seq_len:]
            else:
                # One doc => chunk into sequences; drop remainder < seq_len
                n_docs += 1
                i = 0
                while i + seq_len <= len(ids):
                    flush_sequence(ids[i : i + seq_len], f_bin)
                    i += seq_len

            if n_rows % 50000 == 0:
                log.info(
                    "rows=%d docs=%d sequences=%d tokens=%d",
                    n_rows, n_docs, n_sequences, n_tokens_total
                )

        # Optionally drop remainder buffer (< seq_len). (Common choice)
        log.info("Finished tokenization. Dropping remainder buffer of %d tokens.", len(buffer))

    # Write idx offsets
    np.asarray(offsets, dtype=np.uint64).tofile(str(idx_path))

    stats = {
        "input_file": str(input_file),
        "tokenizer_file": str(tokenizer_file),
        "text_field": text_field,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "seq_len": seq_len,
        "concat_docs": cfg.concat_docs,
        "add_eos_between_docs": cfg.add_eos_between_docs,
        "rows_seen": n_rows,
        "docs_used": n_docs,
        "tokens_written": n_tokens_total,
        "sequences_written": n_sequences,
        "idx_len": len(offsets),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log.info("Wrote: %s", bin_path)
    log.info("Wrote: %s", idx_path)
    log.info("Wrote: %s", stats_path)

    return bin_path, idx_path, stats_path