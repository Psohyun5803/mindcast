#!/usr/bin/env python3
"""오프라인/온라인 공통: 원본 데이터 → 정규화 parquet"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
REPO_SRC = ROOT / "src"
EC_SRC   = ROOT / "emotion_classifier" / "src"
for _p in (REPO_SRC, EC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sarcasm_emotion_adapter.dataio import load_dataset_frame, write_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default=None)
    parser.add_argument("--hf-source",    default=None)
    parser.add_argument("--hf-revision",  default="main")
    parser.add_argument("--hf-token",     default=None)
    parser.add_argument("--hf-max-files", type=int, default=None)
    parser.add_argument("--max-rows",     type=int, default=None)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--output",       required=True)
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        raise ValueError("Either --input or --hf-source must be provided")

    df = load_dataset_frame(input_path=args.input, hf_source=args.hf_source,
                            hf_revision=args.hf_revision, hf_token=args.hf_token,
                            hf_max_files=args.hf_max_files, max_rows=args.max_rows, seed=args.seed)
    output_path = write_dataframe(df, args.output)
    print(json.dumps({"rows": int(len(df)), "columns": list(df.columns),
                      "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
