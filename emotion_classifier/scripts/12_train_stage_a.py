#!/usr/bin/env python3
"""Stage A 학습: 교사 확률 → 학생 모델 지식 증류"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT     = Path(__file__).resolve().parents[1]
REPO_SRC = ROOT.parent / "src"
EC_SRC = ROOT / "src"
for _p in (REPO_SRC, EC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from emotion_classifier_utils import (
    DistillDataset, evaluate_stage_a, run_epoch_stage_a, seed_everything, teacher_prob_columns,
)
from sarcasm_emotion_adapter.dataio import load_dataset_frame
from sarcasm_emotion_adapter.modeling import StageAStudent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          default=None)
    parser.add_argument("--hf-source",      default=None)
    parser.add_argument("--hf-revision",    default="main")
    parser.add_argument("--hf-token",       default=None)
    parser.add_argument("--hf-max-files",   type=int, default=None)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--student-model",  default="beomi/KcELECTRA-base")
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--epochs",         type=int, default=8)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--max-length",     type=int, default=192)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--val-size",       type=float, default=0.1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-rows",       type=int, default=None)
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        raise ValueError("Either --input or --hf-source must be provided")

    seed_everything(args.seed)
    df = load_dataset_frame(input_path=args.input, hf_source=args.hf_source,
                            hf_revision=args.hf_revision, hf_token=args.hf_token,
                            hf_max_files=args.hf_max_files, max_rows=args.max_rows, seed=args.seed)

    pcols = teacher_prob_columns(df)
    if not pcols:
        raise ValueError("No teacher_prob_* columns found")

    df      = df.dropna(subset=["comment"] + pcols).reset_index(drop=True)
    texts   = df["comment"].astype(str).tolist()
    targets = df[pcols].to_numpy(dtype=np.float32)
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=args.val_size,
                                          random_state=args.seed, shuffle=True)

    tokenizer  = AutoTokenizer.from_pretrained(args.student_model, local_files_only=args.local_files_only)
    train_ds   = DistillDataset([texts[i] for i in train_idx], targets[train_idx], tokenizer, args.max_length)
    val_ds     = DistillDataset([texts[i] for i in val_idx],   targets[val_idx],   tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = StageAStudent(args.student_model, len(pcols), args.local_files_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history, best_score, best_state = [], None, None
    print(f"[DEVICE] {device}")
    print(f"[DATA] rows={len(df)} train={len(train_idx)} val={len(val_idx)} labels={len(pcols)}")

    for epoch in range(1, args.epochs + 1):
        train_loss  = run_epoch_stage_a(model, train_loader, optimizer, device)
        val_metrics = evaluate_stage_a(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"top1_match={val_metrics['top1_match']:.4f} prob_mae={val_metrics['prob_mae']:.4f}")
        score = (-val_metrics["prob_mae"], val_metrics["top1_match"])
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    created_at_utc   = datetime.now(timezone.utc).isoformat()
    checkpoint_path  = output_dir / "student_comment_distill.pt"
    torch.save(best_state, checkpoint_path)
    with open(output_dir / "student_comment_distill.meta.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(checkpoint_path), "created_at_utc": created_at_utc,
                   "student_model": args.student_model, "rows": int(len(df)),
                   "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)),
                   "prob_columns": pcols, "seed": args.seed}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
