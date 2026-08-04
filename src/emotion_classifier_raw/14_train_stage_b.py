#!/usr/bin/env python3
"""Stage B 학습: 풍자 감정 어댑터 학습"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT     = Path(__file__).resolve().parents[2]
REPO_SRC = ROOT / "src"
EC_SRC = Path(__file__).resolve().parent
for _p in (REPO_SRC, EC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from emotion_classifier_utils import (
    StageBCollator, StageBDataset, apply_title_optional, compute_losses,
    evaluate_stage_b, prepare_multilabel_targets, seed_everything,
)
from sarcasm_emotion_adapter.dataio import load_dataset_frame
from sarcasm_emotion_adapter.labels import get_default_label_map_path
from sarcasm_emotion_adapter.modeling import StageBSarcasmAdapter, load_label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",               default=None)
    parser.add_argument("--hf-source",           default=None)
    parser.add_argument("--hf-revision",         default="main")
    parser.add_argument("--hf-token",            default=None)
    parser.add_argument("--hf-max-files",        type=int, default=None)
    parser.add_argument("--stagea-checkpoint",   required=True)
    parser.add_argument("--label-map",           default=str(get_default_label_map_path()))
    parser.add_argument("--student-model",       default="beomi/KcELECTRA-base")
    parser.add_argument("--output-dir",          required=True)
    parser.add_argument("--batch-size",          type=int, default=16)
    parser.add_argument("--epochs",              type=int, default=5)
    parser.add_argument("--lr",                  type=float, default=2e-5)
    parser.add_argument("--max-length",          type=int, default=192)
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--val-size",            type=float, default=0.2)
    parser.add_argument("--gate-loss-weight",    type=float, default=1.0)
    parser.add_argument("--emotion-loss-weight", type=float, default=1.0)
    parser.add_argument("--identity-loss-weight",type=float, default=0.5)
    parser.add_argument("--max-rows",            type=int, default=None)
    parser.add_argument("--local-files-only",    action="store_true")
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        raise ValueError("Either --input or --hf-source must be provided")

    seed_everything(args.seed)
    df = load_dataset_frame(input_path=args.input, hf_source=args.hf_source,
                            hf_revision=args.hf_revision, hf_token=args.hf_token,
                            hf_max_files=args.hf_max_files, max_rows=args.max_rows, seed=args.seed)

    id2label = load_label_map(args.label_map)
    label2id = {label: idx for idx, label in id2label.items()}

    if "sarcasm_label" not in df.columns:
        raise ValueError("Expected sarcasm_label in the input dataset")

    df = prepare_multilabel_targets(df, label2id, id2label)
    df = df.dropna(subset=["comment", "sarcasm_label"]).reset_index(drop=True)
    df["sarcasm_label"] = df["sarcasm_label"].astype(int)
    df["news_title"]    = df["news_title"].fillna("").astype(str)
    df["comment"]       = df["comment"].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, local_files_only=args.local_files_only)
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=args.val_size,
                                          random_state=args.seed,
                                          stratify=df["sarcasm_label"].to_numpy())
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    collator = StageBCollator(tokenizer, args.max_length)
    train_loader = DataLoader(StageBDataset(train_df), batch_size=args.batch_size, shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(StageBDataset(val_df),   batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = StageBSarcasmAdapter(args.student_model, args.stagea_checkpoint,
                                     num_labels=len(id2label), local_files_only=args.local_files_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pos_count = max(1, int(train_df["sarcasm_label"].sum()))
    neg_count = max(1, int((train_df["sarcasm_label"] == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_metric, best_state, best_val_preds = [], -math.inf, None, None

    print(f"[DEVICE] {device}")
    print(f"[DATA] rows={len(df)} train={len(train_df)} val={len(val_df)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = gate_sum = emo_sum = id_sum = gate_tgt_sum = steps = 0
        for batch in train_loader:
            tp = batch["title_present"].to(device)
            out = model(comment_input_ids=batch["comment_input_ids"].to(device),
                        comment_attention_mask=batch["comment_attention_mask"].to(device),
                        context_input_ids=batch["context_input_ids"].to(device),
                        context_attention_mask=batch["context_attention_mask"].to(device))
            out = apply_title_optional(out, tp)
            loss, li = compute_losses(out, batch["sarcasm_labels"].to(device),
                                      batch["emotion_targets"].to(device), pos_weight,
                                      args.gate_loss_weight, args.emotion_loss_weight,
                                      args.identity_loss_weight)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()); gate_sum += li["gate_loss"]
            emo_sum  += li["emotion_loss"]; id_sum   += li["identity_loss"]
            gate_tgt_sum += li["gate_target_mean"]; steps += 1

        val_metrics, val_preds = evaluate_stage_b(model, val_loader, device, id2label)
        row = {"epoch": epoch, "train_loss": loss_sum / max(steps, 1),
               "train_gate_loss": gate_sum / max(steps, 1),
               "train_emotion_loss": emo_sum / max(steps, 1),
               "train_identity_loss": id_sum / max(steps, 1),
               "train_gate_target_mean": gate_tgt_sum / max(steps, 1), **val_metrics}
        history.append(row)
        print(f"[epoch {epoch}] loss={row['train_loss']:.4f} sarc_f1={row['sarcasm_f1']:.4f} "
              f"pos_top1_gain={row['positive_emotion_top1_hit_gain']:.4f} "
              f"neg_identity={row['negative_identity_rate']:.4f}")

        sel = row["positive_emotion_top1_hit_corrected"] + 0.5 * row["sarcasm_f1"] + 0.25 * row["negative_identity_rate"]
        if sel > best_metric:
            best_metric = sel
            best_state  = {"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                           "best_epoch": epoch, "best_metrics": row,
                           "label_map": id2label, "args": vars(args)}
            best_val_preds = val_preds.copy()

    created_at_utc  = datetime.now(timezone.utc).isoformat()
    best_state["created_at_utc"] = created_at_utc
    ckpt_path = output_dir / "stageB_adapter_checkpoint.pt"
    torch.save(best_state, ckpt_path)
    with open(output_dir / "stageB_adapter_checkpoint.meta.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(ckpt_path), "created_at_utc": created_at_utc,
                   "input_rows": int(len(df)), "train_rows": int(len(train_df)),
                   "val_rows": int(len(val_df)), "best_epoch": int(best_state["best_epoch"]),
                   "student_model": args.student_model}, f, ensure_ascii=False, indent=2)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv",         index=False, encoding="utf-8-sig")
    best_val_preds.to_csv(output_dir / "best_val_predictions.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
