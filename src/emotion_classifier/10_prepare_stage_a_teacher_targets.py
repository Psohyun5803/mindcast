#!/usr/bin/env python3
"""Stage A 준비: HF 댓글 → KOTE 교사 확률 생성"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT     = Path(__file__).resolve().parents[2]
REPO_SRC = ROOT / "src"
EC_SRC = Path(__file__).resolve().parent
for _p in (REPO_SRC, EC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from emotion_classifier_utils import list_target_files, flatten_file, batched_sigmoid_probs
import random


def normalize_id2label(raw: dict) -> dict[int, str]:
    return {int(k): v for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model",         default="searle-j/kote_for_easygoing_people")
    parser.add_argument("--output",                required=True)
    parser.add_argument("--years",                 nargs="*", default=["2020", "2022", "2023", "2025", "2026"])
    parser.add_argument("--include-2025-variant",  choices=["ver1", "ver2_with_region", "all"], default="ver1")
    parser.add_argument("--include-2026-variant",  choices=["ver1", "ver2_with_region", "all"], default="ver1")
    parser.add_argument("--revision",              default="main")
    parser.add_argument("--token",                 default=None)
    parser.add_argument("--max-files",             type=int, default=None)
    parser.add_argument("--max-comments-per-file", type=int, default=300)
    parser.add_argument("--max-rows",              type=int, default=None)
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--batch-size",            type=int, default=32)
    parser.add_argument("--max-length",            type=int, default=192)
    parser.add_argument("--local-files-only",      action="store_true")
    args = parser.parse_args()

    rng          = random.Random(args.seed)
    target_files = list_target_files(args.years, args.include_2025_variant,
                                     args.include_2026_variant, args.revision, args.token)
    if args.max_files:
        target_files = target_files[:args.max_files]
    if not target_files:
        raise SystemExit("No target files matched the requested years")

    rows, file_stats = [], []
    for idx, repo_path in enumerate(target_files, start=1):
        file_rows = flatten_file(repo_path, args.revision, args.token)
        raw_count = len(file_rows)
        if args.max_comments_per_file and raw_count > args.max_comments_per_file:
            file_rows = rng.sample(file_rows, args.max_comments_per_file)
        rows.extend(file_rows)
        file_stats.append({"source_file": repo_path, "raw_rows": raw_count, "kept_rows": len(file_rows)})
        print(f"[{idx}/{len(target_files)}] {repo_path} raw={raw_count:,} kept={len(file_rows):,}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["source_file", "post_index_in_day", "comment_index"]).reset_index(drop=True)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=args.local_files_only)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model, local_files_only=args.local_files_only).to(device)
    id2label  = normalize_id2label(model.config.id2label)

    probs    = batched_sigmoid_probs(model, tokenizer, df["comment"].astype(str).tolist(),
                                     device, args.batch_size, args.max_length)
    pred_idx = probs.argmax(axis=1)
    df["teacher_pred_idx"]   = pred_idx
    df["teacher_pred_label"] = [id2label[int(i)] for i in pred_idx]
    df["teacher_top_prob"]   = probs.max(axis=1)
    for idx in sorted(id2label.keys()):
        safe = id2label[idx].replace("/", "_").replace(" ", "_")
        df[f"teacher_prob_{idx:02d}_{safe}"] = probs[:, idx]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    pd.DataFrame(file_stats).to_csv(output_path.with_name(output_path.stem + "_file_stats.csv"),
                                    index=False, encoding="utf-8-sig")
    with open(output_path.with_name(output_path.stem + "_id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
