from __future__ import annotations

import argparse
from pathlib import Path

from src.data.pipeline_download import run_download
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TinyStories -> data/raw JSONL")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    out_path = run_download(cfg)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()