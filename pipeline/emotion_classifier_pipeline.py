"""Emotion classifier training & inference pipelines.

Usage
-----
# Stage A: KOTE 교사 확률 생성
python emotion_classifier_pipeline.py --run teacher --output data/teacher_targets.parquet

# Stage A: 학생 모델 지식 증류 학습
python emotion_classifier_pipeline.py --run train-a --input data/teacher_targets.parquet --output-dir models/stage_a

# Stage B: 풍자/감정 타겟 준비
python emotion_classifier_pipeline.py --run prep-b --input data/teacher_targets.parquet --output data/stageb_targets.parquet

# Stage B: 풍자 감정 어댑터 학습
python emotion_classifier_pipeline.py --run train-b --input data/stageb_targets.parquet --stagea-checkpoint models/stage_a/student_comment_distill.pt --output-dir models/stage_b

# 번들 내보내기 (Stage A+B → offline_bundle.pt)
python emotion_classifier_pipeline.py --run export --base-checkpoint models/stage_a/student_comment_distill.pt --stageb-checkpoint models/stage_b/stageB_adapter_checkpoint.pt --output models/offline_bundle.pt

# 오프라인 추론
python emotion_classifier_pipeline.py --run predict --bundle models/offline_bundle.pt --input data/comments.json --output data/predictions.csv

# 후처리: 소분류 → 대분류 컬럼 추가
python emotion_classifier_pipeline.py --run attach-major --input data/predictions.csv --output data/predictions_major.csv
"""
# ── 경로 설정 ────────────────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_BASE    = _Path(__file__).resolve().parents[1]
_EC_SRC  = _BASE / "emotion_classifier" / "src"
for _p in (_BASE / "src", _EC_SRC):
    _sys.path.insert(0, str(_p))
# ────────────────────────────────────────────────────────────────────────────
import argparse, json, math, random, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotion_classifier_utils import (
    DistillDataset, StageBCollator, StageBDataset,
    apply_title_optional, batched_sigmoid_probs, compute_losses,
    evaluate_stage_a, evaluate_stage_b, flatten_file, list_target_files,
    map_cell, prepare_multilabel_targets, read_rows, run_epoch_stage_a,
    seed_everything, teacher_prob_columns, write_output,
)
from sarcasm_emotion_adapter.dataio import load_dataset_frame, write_dataframe
from sarcasm_emotion_adapter.labels import (
    get_default_label_map_path, get_default_major_mapping_path, load_small_to_major,
)
from sarcasm_emotion_adapter.modeling import (
    StageAStudent, StageBSarcasmAdapter, load_label_map,
)
from sarcasm_emotion_adapter.offline import (
    OfflineSarcasmEmotionPredictor, export_offline_bundle,
    read_bundle_metadata, resolve_bundle_checkpoint,
)

# ── 로그 유틸 ────────────────────────────────────────────────────────────────
def _ts(): return datetime.now().strftime("%H:%M:%S")
def log(s): print(f"[{_ts()}] {s}")
def ok(s):  print(f"[{_ts()}] OK  {s}")
def die(s): print(f"[ERROR]  {s}", file=sys.stderr); sys.exit(1)


# ════════════════════════════════════════════════════════════════════════
# Pipeline 0 — 교사 확률 생성 (10_prepare_stage_a_teacher_targets)
# ════════════════════════════════════════════════════════════════════════

def run_teacher(args):
    def normalize_id2label(raw):
        return {int(k): v for k, v in raw.items()}

    log("=== 교사 확률 생성 시작 ===")
    rng = random.Random(args.seed)
    target_files = list_target_files(
        args.years, args.include_2025_variant, args.include_2026_variant,
        args.revision, args.hf_token,
    )
    if args.max_files:
        target_files = target_files[:args.max_files]
    if not target_files:
        die("No target files matched the requested years")

    rows, file_stats = [], []
    for idx, repo_path in enumerate(target_files, start=1):
        file_rows = flatten_file(repo_path, args.revision, args.hf_token)
        raw_count = len(file_rows)
        if args.max_comments_per_file and raw_count > args.max_comments_per_file:
            file_rows = rng.sample(file_rows, args.max_comments_per_file)
        rows.extend(file_rows)
        file_stats.append({"source_file": repo_path, "raw_rows": raw_count, "kept_rows": len(file_rows)})
        log(f"[{idx}/{len(target_files)}] {repo_path} raw={raw_count:,} kept={len(file_rows):,}")

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
    for i in sorted(id2label.keys()):
        safe = id2label[i].replace("/", "_").replace(" ", "_")
        df[f"teacher_prob_{i:02d}_{safe}"] = probs[:, i]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    pd.DataFrame(file_stats).to_csv(output_path.with_name(output_path.stem + "_file_stats.csv"),
                                    index=False, encoding="utf-8-sig")
    with open(output_path.with_name(output_path.stem + "_id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)
    ok(f"교사 확률 저장: {output_path}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 1 — Stage A 학습 (12_train_stage_a)
# ════════════════════════════════════════════════════════════════════════

def run_train_a(args):
    log("=== Stage A 학습 시작 ===")
    seed_everything(args.seed)
    df = load_dataset_frame(input_path=args.input, hf_source=args.hf_source,
                            hf_revision=args.hf_revision, hf_token=args.hf_token,
                            hf_max_files=args.hf_max_files, max_rows=args.max_rows, seed=args.seed)

    pcols = teacher_prob_columns(df)
    if not pcols:
        die("No teacher_prob_* columns found in input")

    df      = df.dropna(subset=["comment"] + pcols).reset_index(drop=True)
    texts   = df["comment"].astype(str).tolist()
    targets = df[pcols].to_numpy(dtype=np.float32)
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=args.val_size,
                                          random_state=args.seed, shuffle=True)

    tokenizer    = AutoTokenizer.from_pretrained(args.student_model, local_files_only=args.local_files_only)
    train_ds     = DistillDataset([texts[i] for i in train_idx], targets[train_idx], tokenizer, args.max_length)
    val_ds       = DistillDataset([texts[i] for i in val_idx],   targets[val_idx],   tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = StageAStudent(args.student_model, len(pcols), args.local_files_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[DEVICE] {device}  rows={len(df)} train={len(train_idx)} val={len(val_idx)} labels={len(pcols)}")
    history, best_score, best_state = [], None, None

    for epoch in range(1, args.epochs + 1):
        train_loss  = run_epoch_stage_a(model, train_loader, optimizer, device)
        val_metrics = evaluate_stage_a(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        log(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"top1_match={val_metrics['top1_match']:.4f} prob_mae={val_metrics['prob_mae']:.4f}")
        score = (-val_metrics["prob_mae"], val_metrics["top1_match"])
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    created_at_utc  = datetime.now(timezone.utc).isoformat()
    ckpt_path       = output_dir / "student_comment_distill.pt"
    torch.save(best_state, ckpt_path)
    with open(output_dir / "student_comment_distill.meta.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(ckpt_path), "created_at_utc": created_at_utc,
                   "student_model": args.student_model, "rows": int(len(df)),
                   "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)),
                   "prob_columns": pcols, "seed": args.seed}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    ok(f"Stage A 체크포인트 저장: {ckpt_path}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 2 — Stage B 타겟 준비 (13_prepare_stage_b_targets)
# ════════════════════════════════════════════════════════════════════════

def run_prep_b(args):
    log("=== Stage B 타겟 준비 시작 ===")
    df = load_dataset_frame(input_path=args.input, hf_source=args.hf_source,
                            hf_revision=args.hf_revision, hf_token=args.hf_token,
                            hf_max_files=args.hf_max_files, max_rows=args.max_rows, seed=args.seed)
    label_map = load_label_map(args.label_map)
    label2id  = {label: idx for idx, label in label_map.items()}

    if "sarcasm_label" not in df.columns and "sarcasm_annotation" not in df.columns:
        die("Expected sarcasm_label or sarcasm_annotation in the input dataset")
    if "base_emotion_label" not in df.columns:
        die("Expected base_emotion_label in the input dataset")

    work = df.copy()
    if "sarcasm_label" not in work.columns:
        work["sarcasm_label"] = 0
    work["sarcasm_label"]       = work["sarcasm_label"].fillna(0).astype(int)
    work["base_emotion_label"]  = work["base_emotion_label"].fillna("").astype(str).str.strip()
    work["actual_emotion_label"]= work.get("actual_emotion_target", "").fillna("").astype(str).str.strip()
    work["use_actual_emotion_target"] = (
        (work["sarcasm_label"] == 1) & work["actual_emotion_label"].ne("")
    ).astype(int)

    if args.drop_positive_without_target:
        work = work[~((work["sarcasm_label"] == 1) & (work["use_actual_emotion_target"] == 0))].reset_index(drop=True)

    actual_mask = work["use_actual_emotion_target"] == 1
    work["final_emotion_target"] = work["base_emotion_label"]
    work.loc[actual_mask, "final_emotion_target"] = work.loc[actual_mask, "actual_emotion_label"]
    work["final_emotion_target_source"] = "comment_only_pred_proxy"
    work.loc[actual_mask, "final_emotion_target_source"] = "positive_actual_emotion_target"

    unknown = sorted(set(work["final_emotion_target"]) - set(label2id))
    if unknown:
        die(f"unknown labels: {unknown}")

    work["base_emotion_id"]         = work["base_emotion_label"].map(label2id).astype(int)
    work["final_emotion_target_id"] = work["final_emotion_target"].map(label2id).astype(int)

    output_path = write_dataframe(work, args.output)
    ok(f"Stage B 타겟 저장: {output_path}  rows={len(work)} sarcasm_positive={int(work['sarcasm_label'].sum())}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 3 — Stage B 학습 (14_train_stage_b)
# ════════════════════════════════════════════════════════════════════════

def run_train_b(args):
    log("=== Stage B 학습 시작 ===")
    seed_everything(args.seed)
    df = load_dataset_frame(input_path=args.input, hf_source=args.hf_source,
                            hf_revision=args.hf_revision, hf_token=args.hf_token,
                            hf_max_files=args.hf_max_files, max_rows=args.max_rows, seed=args.seed)

    id2label = load_label_map(args.label_map)
    label2id = {label: idx for idx, label in id2label.items()}

    if "sarcasm_label" not in df.columns:
        die("Expected sarcasm_label in the input dataset")

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
    pos_count  = max(1, int(train_df["sarcasm_label"].sum()))
    neg_count  = max(1, int((train_df["sarcasm_label"] == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_metric, best_state, best_val_preds = [], -math.inf, None, None

    log(f"[DEVICE] {device}  rows={len(df)} train={len(train_df)} val={len(val_df)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = gate_sum = emo_sum = id_sum = gate_tgt_sum = steps = 0
        for batch in train_loader:
            tp  = batch["title_present"].to(device)
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
        log(f"[epoch {epoch}] loss={row['train_loss']:.4f} sarc_f1={row['sarcasm_f1']:.4f} "
            f"pos_top1_gain={row['positive_emotion_top1_hit_gain']:.4f} "
            f"neg_identity={row['negative_identity_rate']:.4f}")

        sel = (row["positive_emotion_top1_hit_corrected"] + 0.5 * row["sarcasm_f1"]
               + 0.25 * row["negative_identity_rate"])
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
    ok(f"Stage B 체크포인트 저장: {ckpt_path}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 4 — 번들 내보내기 (15_export_offline_bundle)
# ════════════════════════════════════════════════════════════════════════

def run_export(args):
    log("=== 오프라인 번들 내보내기 ===")
    bundle_path = export_offline_bundle(
        bundle_path=args.output,
        student_model=args.student_model,
        base_checkpoint=args.base_checkpoint,
        label_map_path=args.label_map,
        adapter_checkpoint=args.stageb_checkpoint,
        major_mapping_path=args.major_mapping,
        max_length=args.max_length,
    )
    variant = "stagea_stageb" if args.stageb_checkpoint else "stagea_only"
    ok(f"번들 저장: {bundle_path}  variant={variant}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 5 — 오프라인 추론 (01_offline_infer)
# ════════════════════════════════════════════════════════════════════════

def run_predict(args):
    log("=== 오프라인 추론 시작 ===")
    rows, single_input = read_rows(args)
    meta = read_bundle_metadata(args.bundle)
    ckpt = resolve_bundle_checkpoint(args.bundle, meta)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(meta["student_model"])
    predictor = OfflineSarcasmEmotionPredictor(
        ckpt, tokenizer, device=device,
        major_mapping_path=args.major_mapping or str(get_default_major_mapping_path()),
    )
    df = predictor.predict_dataframe(rows)
    write_output(df, Path(args.output), single_input)
    ok(f"추론 결과 저장: {args.output}  rows={len(df)}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 6 — 후처리: 대분류 컬럼 추가 (20_attach_major_labels)
# ════════════════════════════════════════════════════════════════════════

def run_attach_major(args):
    log("=== 대분류 컬럼 추가 ===")
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
    ok(f"대분류 컬럼 추가 완료: {output_path}  added={added}")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Emotion classifier pipelines")
    sub = ap.add_subparsers(dest="run", required=True)

    # ── teacher ──────────────────────────────────────────────────────────
    p = sub.add_parser("teacher", help="교사 확률 생성 (Stage A 전처리)")
    p.add_argument("--teacher-model",         default="searle-j/kote_for_easygoing_people")
    p.add_argument("--output",                required=True)
    p.add_argument("--years",                 nargs="*", default=["2020", "2022", "2023", "2025", "2026"])
    p.add_argument("--include-2025-variant",  choices=["ver1", "ver2_with_region", "all"], default="ver1")
    p.add_argument("--include-2026-variant",  choices=["ver1", "ver2_with_region", "all"], default="ver1")
    p.add_argument("--revision",              default="main")
    p.add_argument("--hf-token",              default=None)
    p.add_argument("--max-files",             type=int, default=None)
    p.add_argument("--max-comments-per-file", type=int, default=300)
    p.add_argument("--max-rows",              type=int, default=None)
    p.add_argument("--seed",                  type=int, default=42)
    p.add_argument("--batch-size",            type=int, default=32)
    p.add_argument("--max-length",            type=int, default=192)
    p.add_argument("--local-files-only",      action="store_true")

    # ── train-a ──────────────────────────────────────────────────────────
    p = sub.add_parser("train-a", help="Stage A 학습: 지식 증류")
    p.add_argument("--input",             default=None)
    p.add_argument("--hf-source",         default=None)
    p.add_argument("--hf-revision",       default="main")
    p.add_argument("--hf-token",          default=None)
    p.add_argument("--hf-max-files",      type=int, default=None)
    p.add_argument("--output-dir",        required=True)
    p.add_argument("--student-model",     default="beomi/KcELECTRA-base")
    p.add_argument("--batch-size",        type=int, default=16)
    p.add_argument("--epochs",            type=int, default=8)
    p.add_argument("--lr",                type=float, default=2e-5)
    p.add_argument("--max-length",        type=int, default=192)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--val-size",          type=float, default=0.1)
    p.add_argument("--local-files-only",  action="store_true")
    p.add_argument("--max-rows",          type=int, default=None)

    # ── prep-b ───────────────────────────────────────────────────────────
    p = sub.add_parser("prep-b", help="Stage B 타겟 준비")
    p.add_argument("--input",                        default=None)
    p.add_argument("--hf-source",                    default=None)
    p.add_argument("--hf-revision",                  default="main")
    p.add_argument("--hf-token",                     default=None)
    p.add_argument("--hf-max-files",                 type=int, default=None)
    p.add_argument("--label-map",                    default=str(get_default_label_map_path()))
    p.add_argument("--output",                       required=True)
    p.add_argument("--drop-positive-without-target", action="store_true")
    p.add_argument("--max-rows",                     type=int, default=None)
    p.add_argument("--seed",                         type=int, default=42)

    # ── train-b ──────────────────────────────────────────────────────────
    p = sub.add_parser("train-b", help="Stage B 학습: 풍자 감정 어댑터")
    p.add_argument("--input",                default=None)
    p.add_argument("--hf-source",            default=None)
    p.add_argument("--hf-revision",         default="main")
    p.add_argument("--hf-token",            default=None)
    p.add_argument("--hf-max-files",        type=int, default=None)
    p.add_argument("--stagea-checkpoint",   required=True)
    p.add_argument("--label-map",           default=str(get_default_label_map_path()))
    p.add_argument("--student-model",       default="beomi/KcELECTRA-base")
    p.add_argument("--output-dir",          required=True)
    p.add_argument("--batch-size",          type=int, default=16)
    p.add_argument("--epochs",              type=int, default=5)
    p.add_argument("--lr",                  type=float, default=2e-5)
    p.add_argument("--max-length",          type=int, default=192)
    p.add_argument("--seed",                type=int, default=42)
    p.add_argument("--val-size",            type=float, default=0.2)
    p.add_argument("--gate-loss-weight",    type=float, default=1.0)
    p.add_argument("--emotion-loss-weight", type=float, default=1.0)
    p.add_argument("--identity-loss-weight",type=float, default=0.5)
    p.add_argument("--max-rows",            type=int, default=None)
    p.add_argument("--local-files-only",    action="store_true")

    # ── export ───────────────────────────────────────────────────────────
    p = sub.add_parser("export", help="오프라인 번들 내보내기")
    p.add_argument("--base-checkpoint",   required=True)
    p.add_argument("--stageb-checkpoint", default=None)
    p.add_argument("--student-model",     default="beomi/KcELECTRA-base")
    p.add_argument("--label-map",         default=str(get_default_label_map_path()))
    p.add_argument("--major-mapping",     default=str(get_default_major_mapping_path()))
    p.add_argument("--max-length",        type=int, default=192)
    p.add_argument("--output",            required=True)

    # ── predict ──────────────────────────────────────────────────────────
    p = sub.add_parser("predict", help="오프라인 추론")
    p.add_argument("--bundle",        required=True)
    p.add_argument("--input",         default=None)
    p.add_argument("--json-input",    default=None)
    p.add_argument("--output",        required=True)
    p.add_argument("--major-mapping", default=None)

    # ── attach-major ─────────────────────────────────────────────────────
    p = sub.add_parser("attach-major", help="소분류 → 대분류 컬럼 추가")
    p.add_argument("--input",    required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--mapping",  default=str(get_default_major_mapping_path()))
    p.add_argument("--columns",  nargs="*",
                   default=["base_pred_label", "corrected_pred_label",
                            "target_emotion_label", "final_emotion_target"])
    p.add_argument("--list-sep", default=" | ")

    args = ap.parse_args()

    dispatch = {
        "teacher":      run_teacher,
        "train-a":      run_train_a,
        "prep-b":       run_prep_b,
        "train-b":      run_train_b,
        "export":       run_export,
        "predict":      run_predict,
        "attach-major": run_attach_major,
    }
    dispatch[args.run](args)


if __name__ == "__main__":
    main()
