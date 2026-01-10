from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.data.download import download_tinystories_jsonl
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

log = get_logger(__name__)


def run_download(cfg: Dict[str, Any]) -> Path:
    """
    Download step runner: reads cfg and downloads TinyStories into data/raw/*.jsonl.
    Returns the output JSONL path.
    """
    root = Path(cfg.get("project", {}).get("root", ".")).resolve()
    paths = ProjectPaths(root=root)
    paths.ensure_dirs()

    raw_dir = root / cfg.get("data", {}).get("raw_dir", "data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    ts_cfg = cfg.get("download", {}).get("tinystories", {})
    enabled = ts_cfg.get("enabled", True)
    if not enabled:
        raise ValueError("Config has download.tinystories.enabled=false; nothing to do.")

    out_file = ts_cfg.get("out_file", f"tinystories_{ts_cfg.get('split', 'train')}.jsonl")
    out_path = raw_dir / out_file

    log.info("Download output path: %s", out_path)

    return download_tinystories_jsonl(
        out_path=out_path,
        split=ts_cfg.get("split", "train"),
        limit=ts_cfg.get("limit", None),
        dataset_name=ts_cfg.get("dataset_name", "roneneldan/TinyStories"),
        text_field=ts_cfg.get("text_field", "text"),
    )