from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.data.pack_tokens import PackConfig, pack_jsonl_to_bin


def run_pack(cfg: Dict[str, Any]) -> Path:
    root = Path(cfg.get("project", {}).get("root", ".")).resolve()

    input_file = root / cfg["data"]["input_file"]
    text_field = cfg["data"].get("text_field", "text_clean")

    tokenizer_file = root / cfg["tokenizer"]["file"]

    pcfg = cfg["packing"]
    out_dir = root / pcfg.get("out_dir", "data/processed")
    name = pcfg.get("name", "train")

    pack_cfg = PackConfig(
        seq_len=int(pcfg.get("seq_len", 512)),
        concat_docs=bool(pcfg.get("concat_docs", True)),
        add_eos_between_docs=bool(pcfg.get("add_eos_between_docs", True)),
        limit_rows=pcfg.get("limit_rows", None),
    )

    pack_jsonl_to_bin(
        input_file=input_file,
        tokenizer_file=tokenizer_file,
        text_field=text_field,
        out_dir=out_dir,
        name=name,
        cfg=pack_cfg,
    )

    return out_dir