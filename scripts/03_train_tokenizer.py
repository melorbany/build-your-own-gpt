from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenization.pipeline_tokenizer import run_tokenizer_training  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer from cleaned JSONL")
    parser.add_argument("--config", type=str, default="configs/tokenizer.yaml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    out = run_tokenizer_training(cfg)
    print(f"Done. Saved tokenizer: {out}")


if __name__ == "__main__":
    main()