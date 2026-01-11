from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from src.data.jsonl import read_jsonl
from src.utils.logging import get_logger

log = get_logger(__name__)


def iter_texts(path: Path, *, text_field: str, limit: Optional[int] = None) -> Iterator[str]:
    n = 0
    for rec in read_jsonl(path):
        t = rec.get(text_field, "")
        if not isinstance(t, str) or not t:
            continue
        yield t
        n += 1
        if limit is not None and n >= int(limit):
            break


def train_bpe_tokenizer(cfg: Dict[str, Any]) -> Path:
    root = Path(cfg.get("project", {}).get("root", ".")).resolve()

    input_file = root / cfg["data"]["input_file"]
    text_field = cfg["data"].get("text_field", "text_clean")

    tcfg = cfg["tokenizer"]
    out_dir = root / tcfg.get("out_dir", "artifacts/tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)

    name = tcfg.get("name", "bpe_tokenizer")
    vocab_size = int(tcfg.get("vocab_size", 16000))
    min_frequency = int(tcfg.get("min_frequency", 2))
    special_tokens = list(tcfg.get("special_tokens", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]))
    lowercase = bool(tcfg.get("lowercase", False))
    limit = tcfg.get("limit", None)

    unk_token = "[UNK]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"

    log.info("Training BPE tokenizer on: %s (field=%s)", input_file, text_field)
    log.info("vocab_size=%d min_frequency=%d limit=%s", vocab_size, min_frequency, str(limit))

    tokenizer = Tokenizer(BPE(unk_token=unk_token))

    # Normalization: NFKC is a good default; optional Lowercase.
    normalizers = [NFKC()]
    if lowercase:
        normalizers.append(Lowercase())
    tokenizer.normalizer = Sequence(normalizers)

    # ByteLevel pre-tokenizer gives robust handling of punctuation/whitespace.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        iter_texts(input_file, text_field=text_field, limit=limit),
        trainer=trainer,
    )

    # Add post-processing to automatically wrap with BOS/EOS
    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[
            (bos_token, tokenizer.token_to_id(bos_token)),
            (eos_token, tokenizer.token_to_id(eos_token)),
        ],
    )

    tokenizer.enable_truncation(max_length=2048)  # safe default; you can change later
    tokenizer.enable_padding(
        direction="right",
        pad_id=tokenizer.token_to_id("[PAD]"),
        pad_token="[PAD]",
    )

    tokenizer_path = out_dir / f"{name}.json"
    tokenizer.save(str(tokenizer_path))

    meta_path = out_dir / f"{name}.meta.txt"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(f"input_file={input_file}\n")
        f.write(f"text_field={text_field}\n")
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"min_frequency={min_frequency}\n")
        f.write(f"lowercase={lowercase}\n")
        f.write(f"special_tokens={special_tokens}\n")

    log.info("Saved tokenizer: %s", tokenizer_path)
    log.info("Saved metadata:  %s", meta_path)

    return tokenizer_path