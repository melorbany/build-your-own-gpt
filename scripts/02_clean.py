from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline_clean import run_clean  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean TinyStories JSONL -> data/interim")
    parser.add_argument("--config", type=str, default="configs/clean.yaml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    out_path = run_clean(cfg)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()