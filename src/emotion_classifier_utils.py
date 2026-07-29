#!/usr/bin/env python3
"""emotion_classifier_utils.py
통합 유틸리티

섹션
------
1. 공통 I/O         오프라인 추론·예측 공유 로직
2. 교사 타겟 준비   Stage A 교사 확률 준비
3. Stage A 학습     지식 증류 학습 루프
4. Stage B 학습     풍자 감정 어댑터 학습 루프
5. 레이블 후처리    소분류→대분류 매핑
"""
from __future__ import annotations

import ast
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_EC_SRC = Path(__file__).resolve().parents[1] / "emotion_classifier" / "src"
if str(_EC_SRC) not in sys.path:
    sys.path.insert(0, str(_EC_SRC))

from sarcasm_emotion_adapter.modeling import StageBSarcasmAdapter, StageAStudent, build_context_text, load_label_map

REPO_ID   = "MindCastSogang/Youtube_news_preprocessed_data"
REPO_TYPE = "dataset"
META_FIELDS = ("candidate_id", "date", "news_date", "dataset_date", "source_file")


# ════════════════════════════════════════════════════════════════════════
# 1. 공통 I/O
# ════════════════════════════════════════════════════════════════════════

def normalize_json_rows(raw) -> tuple[list[dict[str, str]], bool]:
    if isinstance(raw, dict):
        if "items" in raw and isinstance(raw["items"], list):
            raw_rows, single_input = raw["items"], False
        else:
            raw_rows, single_input = [raw], True
    elif isinstance(raw, list):
        raw_rows, single_input = raw, False
    else:
        raise ValueError("JSON input must be an object, an array, or an object with an items array")

    rows = []
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("Each JSON item must be an object")
        comment = str(row.get("comment", "") or "").strip()
        if not comment:
            raise ValueError("Each JSON item must contain a non-empty comment field")
        normalized = {"title": str(row.get("title", row.get("news_title", "")) or ""), "comment": comment}
        for key in META_FIELDS:
            if key in row and row[key] is not None and str(row[key]).strip():
                normalized[key] = str(row[key]).strip()
        rows.append(normalized)
    return rows, single_input


def read_rows(args) -> tuple[list[dict[str, str]], bool]:
    if args.input:
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                return normalize_json_rows(json.load(f))
        if input_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)
        if "comment" not in df.columns:
            raise ValueError("input file must contain a comment column")
        if "title" not in df.columns:
            df["title"] = df["news_title"] if "news_title" in df.columns else ""
        keep_cols = [col for col in [*META_FIELDS, "title", "comment"] if col in df.columns]
        return df[keep_cols].to_dict("records"), False

    if hasattr(args, "input_json") and args.input_json:
        return normalize_json_rows(json.loads(args.input_json))

    if not getattr(args, "comment", None):
        raise ValueError("Either --input, --input-json, or --comment must be provided")
    row = {"title": getattr(args, "title", "") or "", "comment": args.comment}
    if getattr(args, "date", None):
        row["date"] = args.date
    return [row], True


def _json_rows_from_frame(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        payload = {}
        for key in META_FIELDS:
            if key in df.columns and pd.notna(row.get(key)) and str(row.get(key)).strip():
                payload[key] = str(row.get(key)).strip()
        payload.update({
            "title": str(row.get("title", "") or ""),
            "comment": str(row.get("comment", "") or ""),
            "title_provided": bool(row.get("title_provided", False)),
            "gate_score": float(row.get("gate_score", 0.0)),
            "base_emotion": {
                "label": str(row.get("base_emotion_label_fine", "") or ""),
                "major": str(row.get("base_emotion_label_major", "") or ""),
                "score": float(row.get("base_emotion_score", 0.0)),
            },
            "primary_emotion": {
                "label": str(row.get("primary_emotion_label", "") or ""),
                "major": str(row.get("primary_emotion_major", "") or ""),
                "score": float(row.get("primary_emotion_score", 0.0)),
            },
            "top_5": json.loads(str(row.get("top_5_json", "[]") or "[]")),
        })
        rows.append(payload)
    return rows


def write_output(df: pd.DataFrame, output_path: Path, single_input: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        records = _json_rows_from_frame(df)
        payload = records[0] if single_input and len(records) == 1 else records
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    elif output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")


# ════════════════════════════════════════════════════════════════════════
# 2. 교사 타겟 준비 (Stage A)
# ════════════════════════════════════════════════════════════════════════

def iter_posts(payload: dict):
    for day_item in payload.get("data", []):
        if isinstance(day_item.get("posts"), list):
            dataset_date = day_item.get("date")
            for post_idx, post in enumerate(day_item.get("posts", [])):
                yield {"dataset_date": dataset_date, "post_index_in_day": post_idx,
                       "title": post.get("title"), "raw_title": post.get("raw_title"),
                       "news_date": post.get("news_date"), "comments": post.get("comments", [])}
            continue
        for nested_day in day_item.get("dates", []) or []:
            dataset_date = nested_day.get("date")
            for post_idx, post in enumerate(nested_day.get("posts", [])):
                yield {"dataset_date": dataset_date, "post_index_in_day": post_idx,
                       "title": post.get("title"), "raw_title": post.get("raw_title"),
                       "news_date": post.get("news_date"), "comments": post.get("comments", [])}


def list_target_files(years: list[str], include_2025_variant: str, include_2026_variant: str,
                      revision: str, token: str | None) -> list[str]:
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
    local_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                  filename=repo_path, revision=revision, token=token)
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
            rows.append({"source_file": repo_path, "dataset_date": post["dataset_date"],
                         "news_date": post["news_date"], "news_title": title,
                         "post_index_in_day": post["post_index_in_day"],
                         "comment_index": comment_idx, "comment": comment})
    return rows


def batched_sigmoid_probs(model, tokenizer, texts: list[str], device: torch.device,
                           batch_size: int, max_length: int) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start:start + batch_size], truncation=True, padding=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            chunks.append(torch.sigmoid(model(**enc).logits).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ════════════════════════════════════════════════════════════════════════
# 3. Stage A 학습
# ════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DistillDataset(Dataset):
    def __init__(self, texts: list[str], targets: np.ndarray, tokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=max_length, return_tensors="pt")
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "targets": self.targets[idx]}


def teacher_prob_columns(df: pd.DataFrame) -> list[str]:
    return sorted([col for col in df.columns if col.startswith("teacher_prob_")])


def compute_metrics_stage_a(student_prob: torch.Tensor, teacher_prob: torch.Tensor) -> dict[str, float]:
    return {
        "prob_mae": torch.abs(student_prob - teacher_prob).mean().item(),
        "prob_mse": torch.square(student_prob - teacher_prob).mean().item(),
        "top1_match": (student_prob.argmax(dim=-1) == teacher_prob.argmax(dim=-1)).float().mean().item(),
    }


def run_epoch_stage_a(model, loader, optimizer, device) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = criterion(model(batch["input_ids"], batch["attention_mask"]), batch["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_stage_a(model, loader, device) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses, all_student, all_teacher = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        losses.append(criterion(logits, batch["targets"]).item())
        all_student.append(torch.sigmoid(logits).cpu())
        all_teacher.append(batch["targets"].cpu())
    metrics = compute_metrics_stage_a(torch.cat(all_student), torch.cat(all_teacher))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


# ════════════════════════════════════════════════════════════════════════
# 4. Stage B 학습
# ════════════════════════════════════════════════════════════════════════

def parse_label_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, np.ndarray):
        return [str(item).strip() for item in value.tolist() if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(i).strip() for i in parsed if str(i).strip()]
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(i).strip() for i in parsed if str(i).strip()]
            except Exception:
                pass
        for sep in ("|", ";", ","):
            if sep in text:
                return [p.strip() for p in text.split(sep) if p.strip()]
        return [text]
    return [str(value).strip()]


def prepare_multilabel_targets(df: pd.DataFrame, label2id: dict[str, int],
                                id2label: dict[int, str]) -> pd.DataFrame:
    work = df.copy()
    for col in ("emotion_labels", "emotion_label", "final_emotion_target", "actual_emotion_target"):
        if col in work.columns:
            label_lists = work[col].apply(parse_label_list)
            break
    else:
        raise ValueError("Expected one of emotion_labels, emotion_label, final_emotion_target, or actual_emotion_target")

    normalized_lists = []
    for labels in label_lists.tolist():
        seen, deduped = set(), []
        for label in labels:
            if label not in seen:
                deduped.append(label)
                seen.add(label)
        normalized_lists.append(deduped)

    unknown = sorted({label for labels in normalized_lists for label in labels if label not in label2id})
    if unknown:
        raise ValueError(f"unknown emotion labels: {unknown}")

    vectors, primary_labels = [], []
    for labels in normalized_lists:
        vec = np.zeros(len(label2id), dtype=np.float32)
        for label in labels:
            vec[label2id[label]] = 1.0
        vectors.append(vec)
        primary_labels.append(labels[0] if labels else "")

    work["emotion_labels_list"]   = normalized_lists
    work["emotion_label_count"]   = [int(vec.sum()) for vec in vectors]
    work["emotion_target_vector"] = vectors
    work["primary_emotion_label"] = primary_labels
    work["primary_emotion_id"]    = [label2id[label] if label else -1 for label in primary_labels]
    work["emotion_labels_json"]   = [json.dumps(labels, ensure_ascii=False) for labels in normalized_lists]

    if "final_emotion_target_source" not in work.columns:
        work["final_emotion_target_source"] = "gold_emotion_labels"
    for src, dst in [("title", "news_title"), ("news_date", "date"), ("date", "news_date")]:
        if dst not in work.columns:
            work[dst] = work.get(src, "")
    if "dataset_date" not in work.columns:
        work["dataset_date"] = ""
    if "final_emotion_target" not in work.columns:
        work["final_emotion_target"] = work["primary_emotion_label"]
    if "final_emotion_target_id" not in work.columns:
        work["final_emotion_target_id"] = work["primary_emotion_id"]
    if "base_emotion_label" not in work.columns:
        work["base_emotion_label"] = ""
    if "base_emotion_id" not in work.columns:
        work["base_emotion_id"] = -1

    return work[work["emotion_label_count"] > 0].reset_index(drop=True)


class StageBDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        return {
            "comment_text":         str(row["comment"]),
            "context_text":         build_context_text(row.get("news_title", ""), str(row["comment"])),
            "title_present":        int(str(row.get("news_title", "")).strip() != ""),
            "sarcasm_label":        int(row["sarcasm_label"]),
            "emotion_targets":      torch.tensor(row["emotion_target_vector"], dtype=torch.float32),
            "candidate_id":         str(row.get("candidate_id", "")),
            "date":                 str(row.get("date", "")),
            "news_date":            str(row.get("news_date", "")),
            "dataset_date":         str(row.get("dataset_date", "")),
            "news_title":           str(row.get("news_title", "")),
            "comment":              str(row["comment"]),
            "primary_emotion_label":str(row.get("primary_emotion_label", "")),
            "emotion_labels_json":  str(row.get("emotion_labels_json", "[]")),
            "target_source":        str(row["final_emotion_target_source"]),
        }


class StageBCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        def enc(texts):
            return self.tokenizer(texts, truncation=True, padding=True,
                                  max_length=self.max_length, return_tensors="pt")
        ce = enc([b["comment_text"] for b in batch])
        xe = enc([b["context_text"] for b in batch])
        return {
            "comment_input_ids":      ce["input_ids"],
            "comment_attention_mask": ce["attention_mask"],
            "context_input_ids":      xe["input_ids"],
            "context_attention_mask": xe["attention_mask"],
            "title_present":          torch.tensor([b["title_present"] for b in batch], dtype=torch.float32),
            "sarcasm_labels":         torch.tensor([b["sarcasm_label"] for b in batch], dtype=torch.float32),
            "emotion_targets":        torch.stack([b["emotion_targets"] for b in batch]),
            "candidate_ids":          [b["candidate_id"] for b in batch],
            "dates":                  [b["date"] for b in batch],
            "news_dates":             [b["news_date"] for b in batch],
            "dataset_dates":          [b["dataset_date"] for b in batch],
            "titles":                 [b["news_title"] for b in batch],
            "comments":               [b["comment"] for b in batch],
            "primary_emotion_labels": [b["primary_emotion_label"] for b in batch],
            "emotion_labels_json":    [b["emotion_labels_json"] for b in batch],
            "target_sources":         [b["target_source"] for b in batch],
        }


def apply_title_optional(outputs: dict, title_present: torch.Tensor) -> dict:
    mask = title_present.unsqueeze(-1)
    outputs["gate_prob"]        = outputs["gate_prob"] * title_present
    outputs["gate_logits"]      = outputs["gate_logits"] * title_present
    outputs["corrected_logits"] = (outputs["base_logits"]
                                   + outputs["gate_prob"].unsqueeze(-1)
                                   * outputs["delta_logits"] * mask)
    return outputs


def compute_losses(outputs, sarcasm_labels, emotion_targets, pos_weight,
                   gate_loss_weight, emotion_loss_weight, identity_loss_weight):
    with torch.no_grad():
        base_probs    = torch.sigmoid(outputs["base_logits"])
        target_count  = emotion_targets.sum(dim=-1).clamp_min(1.0)
        p_base_gold   = (base_probs * emotion_targets).sum(dim=-1) / target_count
        gate_targets  = sarcasm_labels * (1.0 - p_base_gold)

    gate_weights  = torch.where(sarcasm_labels > 0.5,
                                torch.full_like(sarcasm_labels, float(pos_weight.item())),
                                torch.ones_like(sarcasm_labels))
    gate_loss     = torch.mean(gate_weights * torch.square(outputs["gate_prob"] - gate_targets))
    emotion_loss  = F.binary_cross_entropy_with_logits(outputs["corrected_logits"], emotion_targets)
    neg_mask      = sarcasm_labels < 0.5
    identity_loss = (F.mse_loss(outputs["corrected_logits"][neg_mask], outputs["base_logits"][neg_mask])
                     if neg_mask.any() else outputs["corrected_logits"].new_tensor(0.0))

    total_loss = (gate_loss_weight * gate_loss
                  + emotion_loss_weight * emotion_loss
                  + identity_loss_weight * identity_loss)
    return total_loss, {
        "gate_loss":       float(gate_loss.detach().cpu()),
        "emotion_loss":    float(emotion_loss.detach().cpu()),
        "identity_loss":   float(identity_loss.detach().cpu()),
        "gate_target_mean": float(gate_targets.mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate_stage_b(model, loader, device, id2label: dict[int, str]) -> tuple[dict, pd.DataFrame]:
    model.eval()
    gate_labels_all, gate_probs_all = [], []
    base_top1_hit, corr_top1_hit, base_top5_hit, corr_top5_hit, neg_identity = [], [], [], [], []
    rows = []

    for batch in loader:
        candidate_ids = batch.pop("candidate_ids");  dates         = batch.pop("dates")
        news_dates    = batch.pop("news_dates");     dataset_dates = batch.pop("dataset_dates")
        titles        = batch.pop("titles");         comments      = batch.pop("comments")
        prim_labels   = batch.pop("primary_emotion_labels")
        emo_json      = batch.pop("emotion_labels_json")
        target_srcs   = batch.pop("target_sources")

        title_present    = batch["title_present"].to(device)
        emotion_targets  = batch["emotion_targets"].to(device)
        outputs = model(comment_input_ids=batch["comment_input_ids"].to(device),
                        comment_attention_mask=batch["comment_attention_mask"].to(device),
                        context_input_ids=batch["context_input_ids"].to(device),
                        context_attention_mask=batch["context_attention_mask"].to(device))
        outputs = apply_title_optional(outputs, title_present)

        sl         = batch["sarcasm_labels"].cpu()
        gate_probs = outputs["gate_prob"].detach().cpu()
        base_prob  = torch.sigmoid(outputs["base_logits"].detach().cpu())
        corr_prob  = torch.sigmoid(outputs["corrected_logits"].detach().cpu())
        target_cpu = emotion_targets.detach().cpu()
        top_k      = min(5, base_prob.shape[-1])

        gate_labels_all.extend(sl.tolist())
        gate_probs_all.extend(gate_probs.tolist())

        b1 = base_prob.argmax(dim=-1);  c1 = corr_prob.argmax(dim=-1)
        b5 = base_prob.topk(top_k, dim=-1).indices
        c5 = corr_prob.topk(top_k, dim=-1).indices

        for i in range(len(titles)):
            tv = target_cpu[i]
            bh1 = float(tv[int(b1[i])].item() > 0.5)
            ch1 = float(tv[int(c1[i])].item() > 0.5)
            bh5 = float(tv[b5[i]].max().item() > 0.5)
            ch5 = float(tv[c5[i]].max().item() > 0.5)
            base_top1_hit.append(bh1); corr_top1_hit.append(ch1)
            base_top5_hit.append(bh5); corr_top5_hit.append(ch5)
            if int(sl[i].item()) == 0:
                neg_identity.append(float(int(b1[i].item()) == int(c1[i].item())))
            rows.append({"candidate_id": candidate_ids[i], "date": dates[i],
                         "news_date": news_dates[i], "dataset_date": dataset_dates[i],
                         "news_title": titles[i], "comment": comments[i],
                         "sarcasm_label": int(sl[i].item()),
                         "gate_prob": float(gate_probs[i].item()),
                         "target_primary_label": prim_labels[i],
                         "target_emotion_labels": emo_json[i],
                         "target_source": target_srcs[i],
                         "base_primary_pred_label": id2label[int(b1[i].item())],
                         "corrected_primary_pred_label": id2label[int(c1[i].item())],
                         "base_top1_hit": bh1, "corrected_top1_hit": ch1,
                         "base_top5_hit": bh5, "corrected_top5_hit": ch5})

    gate_binary = [1 if p >= 0.5 else 0 for p in gate_probs_all]
    pos_idx = [i for i, l in enumerate(gate_labels_all) if int(l) == 1]
    neg_idx = [i for i, l in enumerate(gate_labels_all) if int(l) == 0]

    def smean(idx, vals):
        return float(np.mean([vals[i] for i in idx])) if idx else float("nan")

    metrics = {
        "sarcasm_acc":                    float(accuracy_score(gate_labels_all, gate_binary)),
        "sarcasm_precision":              float(precision_score(gate_labels_all, gate_binary, zero_division=0)),
        "sarcasm_recall":                 float(recall_score(gate_labels_all, gate_binary, zero_division=0)),
        "sarcasm_f1":                     float(f1_score(gate_labels_all, gate_binary, zero_division=0)),
        "emotion_top1_hit_base":          float(np.mean(base_top1_hit)) if base_top1_hit else float("nan"),
        "emotion_top1_hit_corrected":     float(np.mean(corr_top1_hit)) if corr_top1_hit else float("nan"),
        "emotion_top1_hit_gain":          float(np.mean(corr_top1_hit) - np.mean(base_top1_hit)) if base_top1_hit else float("nan"),
        "emotion_top5_hit_base":          float(np.mean(base_top5_hit)) if base_top5_hit else float("nan"),
        "emotion_top5_hit_corrected":     float(np.mean(corr_top5_hit)) if corr_top5_hit else float("nan"),
        "emotion_top5_hit_gain":          float(np.mean(corr_top5_hit) - np.mean(base_top5_hit)) if base_top5_hit else float("nan"),
        "positive_emotion_top1_hit_base": smean(pos_idx, base_top1_hit),
        "positive_emotion_top1_hit_corrected": smean(pos_idx, corr_top1_hit),
        "positive_emotion_top1_hit_gain": smean(pos_idx, corr_top1_hit) - smean(pos_idx, base_top1_hit) if pos_idx else float("nan"),
        "positive_emotion_top5_hit_base": smean(pos_idx, base_top5_hit),
        "positive_emotion_top5_hit_corrected": smean(pos_idx, corr_top5_hit),
        "positive_emotion_top5_hit_gain": smean(pos_idx, corr_top5_hit) - smean(pos_idx, base_top5_hit) if pos_idx else float("nan"),
        "negative_identity_rate":         float(np.mean(neg_identity)) if neg_identity else float("nan"),
        "positive_gate_mean":             smean(pos_idx, gate_probs_all),
        "negative_gate_mean":             smean(neg_idx, gate_probs_all),
    }
    return metrics, pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# 5. 레이블 후처리
# ════════════════════════════════════════════════════════════════════════

def map_cell(value: object, small_to_major: dict[str, str], sep: str) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if sep in text:
        parts = [p.strip() for p in text.split(sep)]
        return sep.join([small_to_major.get(p, "") for p in parts])
    return small_to_major.get(text, "")
