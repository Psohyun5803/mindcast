#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ID = "MindCastSogang/Youtube_news_preprocessed_data"
REPO_TYPE = "dataset"


def iter_posts(payload: dict):
    for day_item in payload.get("data", []):
        if isinstance(day_item.get("posts"), list):
            dataset_date = day_item.get("date")
            for post_idx, post in enumerate(day_item.get("posts", [])):
                yield {
                    "dataset_date": dataset_date,
                    "post_index_in_day": post_idx,
                    "title": post.get("title"),
                    "raw_title": post.get("raw_title"),
                    "news_date": post.get("news_date"),
                    "comments": post.get("comments", []),
                }
            continue

        for nested_day in day_item.get("dates", []) or []:
            dataset_date = nested_day.get("date")
            for post_idx, post in enumerate(nested_day.get("posts", [])):
                yield {
                    "dataset_date": dataset_date,
                    "post_index_in_day": post_idx,
                    "title": post.get("title"),
                    "raw_title": post.get("raw_title"),
                    "news_date": post.get("news_date"),
                    "comments": post.get("comments", []),
                }


def normalize_id2label(raw_id2label: dict) -> dict[int, str]:
    return {int(k): v for k, v in raw_id2label.items()}


def list_target_files(years: list[str], include_2025_variant: str, include_2026_variant: str, revision: str, token: str | None) -> list[str]:
    repo_files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE, revision=revision, token=token)
    matched = []
    for path in repo_files:
        if not path.endswith("news_comments.json"):
            continue
        if years and not any(f"/{year}/" in path for year in years):
            continue
        if "/2025/" in path:
            if include_2025_variant == "all" or f"/2025/{include_2025_variant}/" in path:
                matched.append(path)
            continue
        if "/2026/" in path:
            if include_2026_variant == "all" or f"/2026/{include_2026_variant}/" in path:
                matched.append(path)
            continue
        matched.append(path)
    return sorted(matched)


def flatten_file(repo_path: str, revision: str, token: str | None) -> list[dict]:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_path,
        revision=revision,
        token=token,
    )
    with open(local_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    for post in iter_posts(payload):
        title = (post.get("title") or "").strip()
        for comment_idx, comment in enumerate(post["comments"]):
            if not isinstance(comment, str):
                continue
            comment = comment.strip()
            if not comment:
                continue
            rows.append(
                {
                    "source_file": repo_path,
                    "dataset_date": post["dataset_date"],
                    "news_date": post["news_date"],
                    "news_title": title,
                    "post_index_in_day": post["post_index_in_day"],
                    "comment_index": comment_idx,
                    "comment": comment,
                }
            )
    return rows


def batched_sigmoid_probs(model, tokenizer, texts: list[str], device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", default="searle-j/kote_for_easygoing_people")
    parser.add_argument("--output", required=True)
    parser.add_argument("--years", nargs="*", default=["2020", "2022", "2023", "2025", "2026"])
    parser.add_argument("--include-2025-variant", choices=["ver1", "ver2_with_region", "all"], default="ver1")
    parser.add_argument("--include-2026-variant", choices=["ver1", "ver2_with_region", "all"], default="ver1")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-comments-per-file", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_files = list_target_files(args.years, args.include_2025_variant, args.include_2026_variant, args.revision, args.token)
    if args.max_files:
        target_files = target_files[: args.max_files]
    if not target_files:
        raise SystemExit("No target files matched the requested years")

    rows: list[dict] = []
    file_stats: list[dict] = []
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, local_files_only=args.local_files_only).to(device)
    id2label = normalize_id2label(model.config.id2label)

    probs = batched_sigmoid_probs(model, tokenizer, df["comment"].astype(str).tolist(), device, args.batch_size, args.max_length)
    pred_idx = probs.argmax(axis=1)
    df["teacher_pred_idx"] = pred_idx
    df["teacher_pred_label"] = [id2label[int(i)] for i in pred_idx]
    df["teacher_top_prob"] = probs.max(axis=1)
    for idx in sorted(id2label.keys()):
        safe_label = id2label[idx].replace("/", "_").replace(" ", "_")
        df[f"teacher_prob_{idx:02d}_{safe_label}"] = probs[:, idx]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    pd.DataFrame(file_stats).to_csv(output_path.with_name(output_path.stem + "_file_stats.csv"), index=False, encoding="utf-8-sig")
    with open(output_path.with_name(output_path.stem + "_id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
