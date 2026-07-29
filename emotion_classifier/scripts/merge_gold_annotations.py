#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sarcasm_emotion_adapter.modeling import load_label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stageb-input", required=True)
    parser.add_argument("--annotation-input", required=True, help="xlsx or csv with candidate_id and gold_emotion_label")
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--missing-output", required=True)
    args = parser.parse_args()

    stageb_df = pd.read_csv(args.stageb_input)
    ann_path = Path(args.annotation_input)
    if ann_path.suffix.lower() in {".xlsx", ".xls"}:
        ann_df = pd.read_excel(ann_path)
    else:
        ann_df = pd.read_csv(ann_path)

    label_map = load_label_map(args.label_map)
    label2id = {label: idx for idx, label in label_map.items()}

    ann_df = ann_df[["candidate_id", "gold_emotion_label", "gold_emotion_confidence", "annotator_note"]].copy()
    ann_df["gold_emotion_label"] = ann_df["gold_emotion_label"].fillna("").astype(str).str.strip()
    ann_df["gold_emotion_confidence"] = ann_df["gold_emotion_confidence"].fillna("").astype(str).str.strip()
    ann_df["annotator_note"] = ann_df["annotator_note"].fillna("").astype(str).str.strip()

    merged = stageb_df.merge(ann_df, on="candidate_id", how="left")
    merged["gold_emotion_label"] = merged["gold_emotion_label"].fillna("").astype(str).str.strip()
    merged["gold_emotion_available"] = merged["gold_emotion_label"].ne("").astype(int)

    unknown = sorted(set(merged.loc[merged["gold_emotion_available"] == 1, "gold_emotion_label"]) - set(label2id))
    if unknown:
        raise ValueError(f"unknown gold labels: {unknown}")

    gold_mask = merged["gold_emotion_available"] == 1
    merged["final_emotion_target_before_gold"] = merged["final_emotion_target"]
    merged.loc[gold_mask, "final_emotion_target"] = merged.loc[gold_mask, "gold_emotion_label"]
    merged.loc[gold_mask, "final_emotion_target_id"] = merged.loc[gold_mask, "gold_emotion_label"].map(label2id).astype(int)
    merged.loc[gold_mask, "final_emotion_target_source"] = "human_gold_emotion"
    merged["gold_overrode_target"] = (
        gold_mask & merged["final_emotion_target_before_gold"].astype(str).ne(merged["final_emotion_target"].astype(str))
    ).astype(int)

    missing_gold_df = merged[(merged["sarcasm_label"] == 1) & (merged["gold_emotion_available"] == 0)].copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    missing_path = Path(args.missing_output)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_gold_df.to_csv(missing_path, index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "rows": int(len(merged)),
                "gold_rows": int(merged["gold_emotion_available"].sum()),
                "missing_gold_rows": int(len(missing_gold_df)),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
