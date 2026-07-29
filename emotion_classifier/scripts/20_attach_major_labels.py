#!/usr/bin/env python3
"""후처리: 소분류 레이블 컬럼 → 대분류 컬럼 추가"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emotion_classifier_utils import map_cell
from sarcasm_emotion_adapter.labels import get_default_major_mapping_path, load_small_to_major


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--mapping",  default=str(get_default_major_mapping_path()))
    parser.add_argument("--columns",  nargs="*",
                        default=["base_pred_label", "corrected_pred_label",
                                 "target_emotion_label", "final_emotion_target"])
    parser.add_argument("--list-sep", default=" | ")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    small_to_major = load_small_to_major(args.mapping)
    added = []
    for col in args.columns:
        if col not in df.columns:
            continue
        new_col = f"{col}_major"
        df[new_col] = df[col].apply(lambda x: map_cell(x, small_to_major, args.list_sep))
        added.append(new_col)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(json.dumps({"rows": int(len(df)), "added_columns": added, "output": str(output_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
