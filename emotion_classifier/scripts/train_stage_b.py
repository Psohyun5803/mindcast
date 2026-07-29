#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sarcasm_emotion_adapter.dataio import load_dataset_frame
from sarcasm_emotion_adapter.labels import get_default_label_map_path
from sarcasm_emotion_adapter.modeling import StageBSarcasmAdapter, build_context_text, load_label_map

META_FIELDS = ("candidate_id", "date", "news_date", "dataset_date", "source_file")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item).strip() for item in value.tolist() if str(item).strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, (list, tuple)):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
        for sep in ("|", ";", ","):
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
        return [text]
    if pd.isna(value):
        return []
    return [str(value).strip()]


def prepare_multilabel_targets(df: pd.DataFrame, label2id: dict[str, int], id2label: dict[int, str]) -> pd.DataFrame:
    work = df.copy()
    if "emotion_labels" in work.columns:
        label_lists = work["emotion_labels"].apply(parse_label_list)
    elif "emotion_label" in work.columns:
        label_lists = work["emotion_label"].apply(parse_label_list)
    elif "final_emotion_target" in work.columns:
        label_lists = work["final_emotion_target"].apply(parse_label_list)
    elif "actual_emotion_target" in work.columns:
        label_lists = work["actual_emotion_target"].apply(parse_label_list)
    else:
        raise ValueError("Expected one of emotion_labels, emotion_label, final_emotion_target, or actual_emotion_target")

    normalized_lists = []
    for labels in label_lists.tolist():
        deduped = []
        seen = set()
        for label in labels:
            if label not in seen:
                deduped.append(label)
                seen.add(label)
        normalized_lists.append(deduped)

    unknown = sorted({label for labels in normalized_lists for label in labels if label not in label2id})
    if unknown:
        raise ValueError(f"unknown emotion labels: {unknown}")

    vectors = []
    primary_labels = []
    for labels in normalized_lists:
        vec = np.zeros(len(label2id), dtype=np.float32)
        for label in labels:
            vec[label2id[label]] = 1.0
        vectors.append(vec)
        if labels:
            primary_labels.append(labels[0])
        else:
            primary_labels.append("")

    work["emotion_labels_list"] = normalized_lists
    work["emotion_label_count"] = [int(vec.sum()) for vec in vectors]
    work["emotion_target_vector"] = vectors
    work["primary_emotion_label"] = primary_labels
    work["primary_emotion_id"] = [label2id[label] if label else -1 for label in primary_labels]
    work["emotion_labels_json"] = [json.dumps(labels, ensure_ascii=False) for labels in normalized_lists]

    if "final_emotion_target_source" not in work.columns:
        work["final_emotion_target_source"] = "gold_emotion_labels"
    if "news_title" not in work.columns:
        work["news_title"] = work.get("title", "")
    if "date" not in work.columns:
        work["date"] = work.get("news_date", "")
    if "news_date" not in work.columns:
        work["news_date"] = work.get("date", "")
    if "dataset_date" not in work.columns:
        work["dataset_date"] = ""

    # Optional legacy fields for analysis convenience only.
    if "final_emotion_target" not in work.columns:
        work["final_emotion_target"] = work["primary_emotion_label"]
    if "final_emotion_target_id" not in work.columns:
        work["final_emotion_target_id"] = work["primary_emotion_id"]
    if "base_emotion_label" not in work.columns:
        work["base_emotion_label"] = ""
    if "base_emotion_id" not in work.columns:
        work["base_emotion_id"] = -1

    work = work[work["emotion_label_count"] > 0].reset_index(drop=True)
    return work


class StageBDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        return {
            "comment_text": str(row["comment"]),
            "context_text": build_context_text(row.get("news_title", ""), str(row["comment"])),
            "title_present": int(str(row.get("news_title", "")).strip() != ""),
            "sarcasm_label": int(row["sarcasm_label"]),
            "emotion_targets": torch.tensor(row["emotion_target_vector"], dtype=torch.float32),
            "candidate_id": str(row.get("candidate_id", "")),
            "date": str(row.get("date", "")),
            "news_date": str(row.get("news_date", "")),
            "dataset_date": str(row.get("dataset_date", "")),
            "news_title": str(row.get("news_title", "")),
            "comment": str(row["comment"]),
            "primary_emotion_label": str(row.get("primary_emotion_label", "")),
            "emotion_labels_json": str(row.get("emotion_labels_json", "[]")),
            "target_source": str(row["final_emotion_target_source"]),
        }


class StageBCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        comment_texts = [item["comment_text"] for item in batch]
        context_texts = [item["context_text"] for item in batch]
        comment_enc = self.tokenizer(
            comment_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        context_enc = self.tokenizer(
            context_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "comment_input_ids": comment_enc["input_ids"],
            "comment_attention_mask": comment_enc["attention_mask"],
            "context_input_ids": context_enc["input_ids"],
            "context_attention_mask": context_enc["attention_mask"],
            "title_present": torch.tensor([item["title_present"] for item in batch], dtype=torch.float32),
            "sarcasm_labels": torch.tensor([item["sarcasm_label"] for item in batch], dtype=torch.float32),
            "emotion_targets": torch.stack([item["emotion_targets"] for item in batch], dim=0),
            "candidate_ids": [item["candidate_id"] for item in batch],
            "dates": [item["date"] for item in batch],
            "news_dates": [item["news_date"] for item in batch],
            "dataset_dates": [item["dataset_date"] for item in batch],
            "titles": [item["news_title"] for item in batch],
            "comments": [item["comment"] for item in batch],
            "primary_emotion_labels": [item["primary_emotion_label"] for item in batch],
            "emotion_labels_json": [item["emotion_labels_json"] for item in batch],
            "target_sources": [item["target_source"] for item in batch],
        }


def apply_title_optional(outputs: dict[str, torch.Tensor], title_present: torch.Tensor) -> dict[str, torch.Tensor]:
    present_mask = title_present.unsqueeze(-1)
    outputs["gate_prob"] = outputs["gate_prob"] * title_present
    outputs["gate_logits"] = outputs["gate_logits"] * title_present
    outputs["corrected_logits"] = outputs["base_logits"] + outputs["gate_prob"].unsqueeze(-1) * outputs["delta_logits"] * present_mask
    return outputs


def compute_losses(
    outputs: dict[str, torch.Tensor],
    sarcasm_labels: torch.Tensor,
    emotion_targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gate_loss_weight: float,
    emotion_loss_weight: float,
    identity_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        base_probs = torch.sigmoid(outputs["base_logits"])
        target_count = emotion_targets.sum(dim=-1).clamp_min(1.0)
        p_base_gold = (base_probs * emotion_targets).sum(dim=-1) / target_count
        gate_targets = sarcasm_labels * (1.0 - p_base_gold)

    gate_weights = torch.where(
        sarcasm_labels > 0.5,
        torch.full_like(sarcasm_labels, float(pos_weight.item())),
        torch.ones_like(sarcasm_labels),
    )
    gate_loss = torch.mean(gate_weights * torch.square(outputs["gate_prob"] - gate_targets))
    emotion_loss = F.binary_cross_entropy_with_logits(outputs["corrected_logits"], emotion_targets)

    negative_mask = sarcasm_labels < 0.5
    if negative_mask.any():
        identity_loss = F.mse_loss(outputs["corrected_logits"][negative_mask], outputs["base_logits"][negative_mask])
    else:
        identity_loss = outputs["corrected_logits"].new_tensor(0.0)

    total_loss = gate_loss_weight * gate_loss + emotion_loss_weight * emotion_loss + identity_loss_weight * identity_loss
    metrics = {
        "gate_loss": float(gate_loss.detach().cpu().item()),
        "emotion_loss": float(emotion_loss.detach().cpu().item()),
        "identity_loss": float(identity_loss.detach().cpu().item()),
        "gate_target_mean": float(gate_targets.mean().detach().cpu().item()),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_model(model, loader, device, id2label: dict[int, str]) -> tuple[dict, pd.DataFrame]:
    model.eval()
    gate_labels_all = []
    gate_probs_all = []
    base_top1_hit_all = []
    corrected_top1_hit_all = []
    base_top5_hit_all = []
    corrected_top5_hit_all = []
    negative_identity_flags = []
    rows = []

    for batch in loader:
        candidate_ids = batch.pop("candidate_ids")
        dates = batch.pop("dates")
        news_dates = batch.pop("news_dates")
        dataset_dates = batch.pop("dataset_dates")
        titles = batch.pop("titles")
        comments = batch.pop("comments")
        primary_emotion_labels = batch.pop("primary_emotion_labels")
        emotion_labels_json = batch.pop("emotion_labels_json")
        target_sources = batch.pop("target_sources")

        title_present = batch["title_present"].to(device)
        emotion_targets = batch["emotion_targets"].to(device)
        outputs = model(
            comment_input_ids=batch["comment_input_ids"].to(device),
            comment_attention_mask=batch["comment_attention_mask"].to(device),
            context_input_ids=batch["context_input_ids"].to(device),
            context_attention_mask=batch["context_attention_mask"].to(device),
        )
        outputs = apply_title_optional(outputs, title_present)

        sarcasm_labels = batch["sarcasm_labels"].cpu()
        gate_probs = outputs["gate_prob"].detach().cpu()
        base_logits = outputs["base_logits"].detach().cpu()
        corrected_logits = outputs["corrected_logits"].detach().cpu()
        target_cpu = emotion_targets.detach().cpu()

        base_prob = torch.sigmoid(base_logits)
        corrected_prob = torch.sigmoid(corrected_logits)
        base_top1 = base_prob.argmax(dim=-1)
        corrected_top1 = corrected_prob.argmax(dim=-1)
        top_k = min(5, base_prob.shape[-1])
        base_top5_idx = base_prob.topk(top_k, dim=-1).indices
        corrected_top5_idx = corrected_prob.topk(top_k, dim=-1).indices

        gate_labels_all.extend(sarcasm_labels.tolist())
        gate_probs_all.extend(gate_probs.tolist())

        for idx in range(len(titles)):
            target_vec = target_cpu[idx]
            base_top1_hit = float(target_vec[int(base_top1[idx].item())].item() > 0.5)
            corrected_top1_hit = float(target_vec[int(corrected_top1[idx].item())].item() > 0.5)
            base_top5_hit = float(target_vec[base_top5_idx[idx]].max().item() > 0.5)
            corrected_top5_hit = float(target_vec[corrected_top5_idx[idx]].max().item() > 0.5)
            same_primary = float(int(base_top1[idx].item()) == int(corrected_top1[idx].item()))

            base_top1_hit_all.append(base_top1_hit)
            corrected_top1_hit_all.append(corrected_top1_hit)
            base_top5_hit_all.append(base_top5_hit)
            corrected_top5_hit_all.append(corrected_top5_hit)
            if int(sarcasm_labels[idx].item()) == 0:
                negative_identity_flags.append(same_primary)

            rows.append(
                {
                    "candidate_id": candidate_ids[idx],
                    "date": dates[idx],
                    "news_date": news_dates[idx],
                    "dataset_date": dataset_dates[idx],
                    "news_title": titles[idx],
                    "comment": comments[idx],
                    "sarcasm_label": int(sarcasm_labels[idx].item()),
                    "gate_prob": float(gate_probs[idx].item()),
                    "target_primary_label": primary_emotion_labels[idx],
                    "target_emotion_labels": emotion_labels_json[idx],
                    "target_source": target_sources[idx],
                    "base_primary_pred_label": id2label[int(base_top1[idx].item())],
                    "corrected_primary_pred_label": id2label[int(corrected_top1[idx].item())],
                    "base_top1_hit": base_top1_hit,
                    "corrected_top1_hit": corrected_top1_hit,
                    "base_top5_hit": base_top5_hit,
                    "corrected_top5_hit": corrected_top5_hit,
                }
            )

    gate_binary = [1 if prob >= 0.5 else 0 for prob in gate_probs_all]
    sarcasm_acc = accuracy_score(gate_labels_all, gate_binary)
    sarcasm_precision = precision_score(gate_labels_all, gate_binary, zero_division=0)
    sarcasm_recall = recall_score(gate_labels_all, gate_binary, zero_division=0)
    sarcasm_f1 = f1_score(gate_labels_all, gate_binary, zero_division=0)

    positive_idx = [i for i, label in enumerate(gate_labels_all) if int(label) == 1]
    negative_idx = [i for i, label in enumerate(gate_labels_all) if int(label) == 0]

    def subset_mean(indices: list[int], values: list[float]) -> float:
        if not indices:
            return float("nan")
        return float(np.mean([values[i] for i in indices]))

    metrics = {
        "sarcasm_acc": float(sarcasm_acc),
        "sarcasm_precision": float(sarcasm_precision),
        "sarcasm_recall": float(sarcasm_recall),
        "sarcasm_f1": float(sarcasm_f1),
        "emotion_top1_hit_base": float(np.mean(base_top1_hit_all)) if base_top1_hit_all else float("nan"),
        "emotion_top1_hit_corrected": float(np.mean(corrected_top1_hit_all)) if corrected_top1_hit_all else float("nan"),
        "emotion_top1_hit_gain": float(np.mean(corrected_top1_hit_all) - np.mean(base_top1_hit_all)) if base_top1_hit_all else float("nan"),
        "emotion_top5_hit_base": float(np.mean(base_top5_hit_all)) if base_top5_hit_all else float("nan"),
        "emotion_top5_hit_corrected": float(np.mean(corrected_top5_hit_all)) if corrected_top5_hit_all else float("nan"),
        "emotion_top5_hit_gain": float(np.mean(corrected_top5_hit_all) - np.mean(base_top5_hit_all)) if base_top5_hit_all else float("nan"),
        "positive_emotion_top1_hit_base": subset_mean(positive_idx, base_top1_hit_all),
        "positive_emotion_top1_hit_corrected": subset_mean(positive_idx, corrected_top1_hit_all),
        "positive_emotion_top1_hit_gain": subset_mean(positive_idx, corrected_top1_hit_all) - subset_mean(positive_idx, base_top1_hit_all) if positive_idx else float("nan"),
        "positive_emotion_top5_hit_base": subset_mean(positive_idx, base_top5_hit_all),
        "positive_emotion_top5_hit_corrected": subset_mean(positive_idx, corrected_top5_hit_all),
        "positive_emotion_top5_hit_gain": subset_mean(positive_idx, corrected_top5_hit_all) - subset_mean(positive_idx, base_top5_hit_all) if positive_idx else float("nan"),
        "negative_identity_rate": float(np.mean(negative_identity_flags)) if negative_identity_flags else float("nan"),
        "positive_gate_mean": float(np.mean([gate_probs_all[i] for i in positive_idx])) if positive_idx else float("nan"),
        "negative_gate_mean": float(np.mean([gate_probs_all[i] for i in negative_idx])) if negative_idx else float("nan"),
    }
    return metrics, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="local CSV/JSON/Parquet/XLSX")
    parser.add_argument("--hf-source", default=None, help="e.g. MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-max-files", type=int, default=None)
    parser.add_argument("--stagea-checkpoint", required=True)
    parser.add_argument("--label-map", default=str(get_default_label_map_path()))
    parser.add_argument("--student-model", default="beomi/KcELECTRA-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--emotion-loss-weight", type=float, default=1.0)
    parser.add_argument("--identity-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        raise ValueError("Either --input or --hf-source must be provided")

    seed_everything(args.seed)
    df = load_dataset_frame(
        input_path=args.input,
        hf_source=args.hf_source,
        hf_revision=args.hf_revision,
        hf_token=args.hf_token,
        hf_max_files=args.hf_max_files,
        max_rows=args.max_rows,
        seed=args.seed,
    )

    id2label = load_label_map(args.label_map)
    label2id = {label: idx for idx, label in id2label.items()}

    if "sarcasm_label" not in df.columns:
        raise ValueError("Expected sarcasm_label in the input dataset")

    df = prepare_multilabel_targets(df, label2id, id2label)
    df = df.dropna(subset=["comment", "sarcasm_label"]).reset_index(drop=True)
    df["sarcasm_label"] = df["sarcasm_label"].astype(int)
    df["news_title"] = df["news_title"].fillna("").astype(str)
    df["comment"] = df["comment"].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, local_files_only=args.local_files_only)
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["sarcasm_label"].to_numpy(),
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(StageBDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=StageBCollator(tokenizer, args.max_length))
    val_loader = DataLoader(StageBDataset(val_df), batch_size=args.batch_size, shuffle=False, collate_fn=StageBCollator(tokenizer, args.max_length))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StageBSarcasmAdapter(args.student_model, args.stagea_checkpoint, num_labels=len(id2label), local_files_only=args.local_files_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pos_count = max(1, int(train_df["sarcasm_label"].sum()))
    neg_count = max(1, int((train_df["sarcasm_label"] == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_metric = -math.inf
    best_state = None
    best_val_predictions = None

    print(f"[DEVICE] {device}")
    print(f"[DATA] rows={len(df)} train={len(train_df)} val={len(val_df)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = gate_loss_sum = emotion_loss_sum = identity_loss_sum = gate_target_sum = 0.0
        steps = 0
        for batch in train_loader:
            title_present = batch["title_present"].to(device)
            outputs = model(
                comment_input_ids=batch["comment_input_ids"].to(device),
                comment_attention_mask=batch["comment_attention_mask"].to(device),
                context_input_ids=batch["context_input_ids"].to(device),
                context_attention_mask=batch["context_attention_mask"].to(device),
            )
            outputs = apply_title_optional(outputs, title_present)
            loss, loss_items = compute_losses(
                outputs=outputs,
                sarcasm_labels=batch["sarcasm_labels"].to(device),
                emotion_targets=batch["emotion_targets"].to(device),
                pos_weight=pos_weight,
                gate_loss_weight=args.gate_loss_weight,
                emotion_loss_weight=args.emotion_loss_weight,
                identity_loss_weight=args.identity_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item())
            gate_loss_sum += loss_items["gate_loss"]
            emotion_loss_sum += loss_items["emotion_loss"]
            identity_loss_sum += loss_items["identity_loss"]
            gate_target_sum += loss_items["gate_target_mean"]
            steps += 1

        val_metrics, val_predictions = evaluate_model(model, val_loader, device, id2label)
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(steps, 1),
            "train_gate_loss": gate_loss_sum / max(steps, 1),
            "train_emotion_loss": emotion_loss_sum / max(steps, 1),
            "train_identity_loss": identity_loss_sum / max(steps, 1),
            "train_gate_target_mean": gate_target_sum / max(steps, 1),
            **val_metrics,
        }
        history.append(epoch_row)
        print(
            f"[epoch {epoch}] loss={epoch_row['train_loss']:.4f} "
            f"sarc_f1={epoch_row['sarcasm_f1']:.4f} "
            f"pos_top1_gain={epoch_row['positive_emotion_top1_hit_gain']:.4f} "
            f"overall_top1_gain={epoch_row['emotion_top1_hit_gain']:.4f} "
            f"neg_identity={epoch_row['negative_identity_rate']:.4f}"
        )

        selection_metric = epoch_row["positive_emotion_top1_hit_corrected"] + 0.5 * epoch_row["sarcasm_f1"] + 0.25 * epoch_row["negative_identity_rate"]
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "best_epoch": epoch,
                "best_metrics": epoch_row,
                "label_map": id2label,
                "args": vars(args),
            }
            best_val_predictions = val_predictions.copy()

    assert best_state is not None and best_val_predictions is not None
    created_at_utc = datetime.now(timezone.utc).isoformat()
    best_state["created_at_utc"] = created_at_utc
    checkpoint_path = output_dir / "stageB_adapter_checkpoint.pt"
    torch.save(best_state, checkpoint_path)
    with open(output_dir / "stageB_adapter_checkpoint.meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "created_at_utc": created_at_utc,
                "input": args.input,
                "hf_source": args.hf_source,
                "input_rows": int(len(df)),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "best_epoch": int(best_state["best_epoch"]),
                "student_model": args.student_model,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    best_val_predictions.to_csv(output_dir / "best_val_predictions.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_rows": int(len(df)),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "best_epoch": int(best_state["best_epoch"]),
                "best_metrics": best_state["best_metrics"],
                "output_dir": str(output_dir),
                "created_at_utc": created_at_utc,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
