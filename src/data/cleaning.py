from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CleanConfig:
    strip: bool = True
    normalize_newlines: bool = True
    rstrip_lines: bool = True
    max_consecutive_newlines: int = 2


_newline_re = re.compile(r"\n{3,}")


def clean_text(text: str, cfg: CleanConfig) -> str:
    if text is None:
        return ""

    t = text

    if cfg.normalize_newlines:
        t = t.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.rstrip_lines:
        # Remove trailing spaces on each line but keep line breaks
        t = "\n".join(line.rstrip() for line in t.split("\n"))

    if cfg.max_consecutive_newlines is not None and cfg.max_consecutive_newlines >= 1:
        # Collapse 3+ newlines down to (max_consecutive_newlines)
        repl = "\n" * cfg.max_consecutive_newlines
        t = _newline_re.sub(repl, t)

    if cfg.strip:
        t = t.strip()

    return t


def is_acceptable(text: str, *, min_chars: int = 1) -> bool:
    if text is None:
        return False
    return len(text) >= min_chars