from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.data.cleaning import CleanConfig, clean_text, is_acceptable
from src.data.jsonl import read_jsonl, write_jsonl
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

log = get_logger(__name__)


def run_clean(cfg: Dict[str, Any]) -> Path:
    root = Path(cfg.get("project", {}).get("root", ".")).resolve()
    ProjectPaths(root=root).ensure_dirs()

    raw_file = root / cfg["data"]["raw_file"]
    interim_dir = root / cfg["data"]["interim_dir"]
    out_path = interim_dir / cfg["data"]["out_file"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    clean_cfg = cfg.get("clean", {})
    input_field = clean_cfg.get("input_text_field", "text")
    output_field = clean_cfg.get("output_text_field", "text_clean")

    cc = CleanConfig(
        strip=clean_cfg.get("strip", True),
        normalize_newlines=clean_cfg.get("normalize_newlines", True),
        rstrip_lines=clean_cfg.get("rstrip_lines", True),
        max_consecutive_newlines=clean_cfg.get("max_consecutive_newlines", 2),
    )

    drop_empty = clean_cfg.get("drop_empty", True)
    min_chars = int(clean_cfg.get("min_chars", 1))
    limit = clean_cfg.get("limit", None)

    log.info("Cleaning input: %s", raw_file)
    log.info("Cleaning output: %s", out_path)

    def records():
        n_in = 0
        n_out = 0

        for rec in read_jsonl(raw_file):
            n_in += 1
            if limit is not None and n_in > int(limit):
                break

            text = rec.get(input_field, "")
            cleaned = clean_text(text, cc)

            if drop_empty and not is_acceptable(cleaned, min_chars=min_chars):
                continue

            out = dict(rec)
            out["id"] = rec.get("id", n_in)  # stable enough for now
            out[output_field] = cleaned
            out["n_chars"] = len(cleaned)
            out["n_lines"] = cleaned.count("\n") + 1 if cleaned else 0

            n_out += 1
            if n_out % 50000 == 0:
                log.info("Processed %d -> kept %d", n_in, n_out)

            yield out

        log.info("Done. Processed %d -> kept %d", n_in, n_out)

    write_jsonl(records(), out_path)
    return out_path