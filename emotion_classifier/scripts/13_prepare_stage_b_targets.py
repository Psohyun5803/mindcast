#!/usr/bin/env python3
"""Stage B 준비: 데이터 정규화 + 풍자/감정 타겟 생성"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
REPO_SRC = ROOT.parent / "src"
EC_SRC = ROOT / "src"
for _p in (REPO_SRC, EC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sarcasm_emotion_adapter.dataio import load_dataset_frame, write_dataframe
from sarcasm_emotion_adapter.modeling import load_label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",                        default=None)
    parser.add_argument("--hf-source",                    default=None)
    parser.add_argument("--hf-revision",                  default="main")
    parser.add_argument("--hf-token",                     default=None)
    parser.add_argument("--hf-max-files",                 type=int, default=None)
    parser.add_argument("--label-map",                    required=True)
    parser.add_argument("--output",                       required=True)
    parser.add_argument("--drop-positive-without-target", action="store_true")
    parser.add_argument("--max-rows",                     type=int, default=None)
    parser.add_argument("--seed",                         type=int, default=42)
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        raise ValueError("Either --input or --hf-source must be provided")

    df = load_dataset_frame(input_path=args.input, hf_source=args.hf_source,
                            hf_revision=args.hf_revision, hf_token=args.hf_token,
                            hf_max_files=args.hf_max_files, max_rows=args.max_rows, seed=args.seed)
    label_map = load_label_map(args.label_map)
    label2id  = {label: idx for idx, label in label_map.items()}

    if "sarcasm_label" not in df.columns and "sarcasm_annotation" not in df.columns:
        raise ValueError("Expected sarcasm_label or sarcasm_annotation in the input dataset")
    if "base_emotion_label" not in df.columns:
        raise ValueError("Expected base_emotion_label in the input dataset")

    work = df.copy()
    if "sarcasm_label" not in work.columns:
        work["sarcasm_label"] = 0
    work["sarcasm_label"]       = work["sarcasm_label"].fillna(0).astype(int)
    work["base_emotion_label"]  = work["base_emotion_label"].fillna("").astype(str).str.strip()
    work["actual_emotion_label"]= work.get("actual_emotion_target", "").fillna("").astype(str).str.strip()
    work["use_actual_emotion_target"] = ((work["sarcasm_label"] == 1) & work["actual_emotion_label"].ne("")).astype(int)

    if args.drop_positive_without_target:
        work = work[~((work["sarcasm_label"] == 1) & (work["use_actual_emotion_target"] == 0))].reset_index(drop=True)

    actual_mask = work["use_actual_emotion_target"] == 1
    work["final_emotion_target"] = work["base_emotion_label"]
    work.loc[actual_mask, "final_emotion_target"] = work.loc[actual_mask, "actual_emotion_label"]
    work["final_emotion_target_source"] = "comment_only_pred_proxy"
    work.loc[actual_mask, "final_emotion_target_source"] = "positive_actual_emotion_target"

    unknown = sorted(set(work["final_emotion_target"]) - set(label2id))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")

    work["base_emotion_id"]         = work["base_emotion_label"].map(label2id).astype(int)
    work["final_emotion_target_id"] = work["final_emotion_target"].map(label2id).astype(int)

    output_path = write_dataframe(work, args.output)
    print(json.dumps({"rows": int(len(work)), "sarcasm_positive": int(work["sarcasm_label"].sum()),
                      "used_actual_emotion_target": int(work["use_actual_emotion_target"].sum()),
                      "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
