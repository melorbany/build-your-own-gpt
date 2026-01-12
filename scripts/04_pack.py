from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline_pack import run_pack  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack cleaned JSONL into fixed-length token sequences")
    parser.add_argument("--config", type=str, default="configs/pack.yaml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    out_dir = run_pack(cfg)
    print(f"Done. Wrote packed dataset to: {out_dir}")


if __name__ == "__main__":
    main()