#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT     = Path(__file__).resolve().parents[2]
REPO_SRC = ROOT / "src"
EC_SRC   = ROOT / "emotion_classifier" / "src"
for _p in (REPO_SRC, EC_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sarcasm_emotion_adapter.labels import get_default_major_mapping_path, load_small_to_major
from sarcasm_emotion_adapter.modeling import StageAStudent, StageBSarcasmAdapter, build_context_text, load_label_map

META_FIELDS = ("candidate_id", "date", "news_date", "dataset_date", "source_file")


def normalize_json_rows(raw) -> tuple[pd.DataFrame, bool]:
    if isinstance(raw, dict):
        if "items" in raw and isinstance(raw["items"], list):
            raw_rows = raw["items"]
            single_input = False
        else:
            raw_rows = [raw]
            single_input = True
    elif isinstance(raw, list):
        raw_rows = raw
        single_input = False
    else:
        raise ValueError("JSON input must be an object, an array, or an object with an items array")

    rows = []
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("Each JSON item must be an object")
        comment = str(row.get("comment", "") or "").strip()
        if not comment:
            raise ValueError("Each JSON item must contain a non-empty comment field")
        normalized = {
            "title": str(row.get("title", row.get("news_title", "")) or ""),
            "comment": comment,
        }
        for key in META_FIELDS:
            if key in row and row[key] is not None and str(row[key]).strip():
                normalized[key] = str(row[key]).strip()
        rows.append(normalized)
    return pd.DataFrame(rows), single_input


def read_rows(args) -> tuple[pd.DataFrame, bool]:
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
            if "news_title" in df.columns:
                df["title"] = df["news_title"]
            else:
                df["title"] = ""
        keep_cols = [col for col in [*META_FIELDS, "title", "comment"] if col in df.columns]
        return df[keep_cols].copy(), False
    if args.input_json:
        return normalize_json_rows(json.loads(args.input_json))
    row = {"title": args.title or "", "comment": args.comment}
    if args.date:
        row["date"] = args.date
    return pd.DataFrame([row]), True


def _json_rows_from_frame(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        payload = {}
        for key in META_FIELDS:
            if key in df.columns and pd.notna(row.get(key)) and str(row.get(key)).strip():
                payload[key] = str(row.get(key)).strip()
        payload.update(
            {
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
            }
        )
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


def load_base_model(model_name: str, checkpoint_path: str, num_labels: int, local_files_only: bool):
    model = StageAStudent(model_name=model_name, num_labels=num_labels, local_files_only=local_files_only)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--student-model", default="beomi/KcELECTRA-base")
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--stageb-checkpoint", default=None)
    parser.add_argument("--input", default=None, help="JSON/CSV/Excel/Parquet with title(optional), comment")
    parser.add_argument("--input-json", default=None, help="Inline JSON object or array with title/comment fields")
    parser.add_argument("--comment", default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--major-mapping", default=str(get_default_major_mapping_path()))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.input and not args.input_json and not args.comment:
        raise ValueError("Either --input, --input-json, or --comment must be provided")

    rows, single_input = read_rows(args)
    id2label = load_label_map(args.label_map)
    small_to_major = load_small_to_major(args.major_mapping)
    num_labels = len(id2label)
    top_k = min(max(1, int(args.top_k)), num_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, local_files_only=args.local_files_only)
    base_model = load_base_model(args.student_model, args.base_checkpoint, num_labels, args.local_files_only)
    adapter_model = None
    if args.stageb_checkpoint:
        adapter_meta = torch.load(args.stageb_checkpoint, map_location="cpu")
        adapter_model = StageBSarcasmAdapter(
            model_name=args.student_model,
            stagea_checkpoint=args.base_checkpoint,
            num_labels=num_labels,
            local_files_only=args.local_files_only,
        )
        adapter_model.load_state_dict(adapter_meta["model_state_dict"], strict=True)
        adapter_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)
    if adapter_model is not None:
        adapter_model = adapter_model.to(device)

    results = []
    for start in range(0, len(rows), args.batch_size):
        batch_df = rows.iloc[start : start + args.batch_size].copy()
        comments = batch_df["comment"].astype(str).tolist()
        titles = batch_df["title"].fillna("").astype(str).tolist()
        title_present = torch.tensor([1.0 if title.strip() else 0.0 for title in titles], dtype=torch.float32, device=device)

        comment_enc = tokenizer(comments, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        context_texts = [build_context_text(title, comment) for title, comment in zip(titles, comments)]
        context_enc = tokenizer(context_texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        comment_enc = {k: v.to(device) for k, v in comment_enc.items()}
        context_enc = {k: v.to(device) for k, v in context_enc.items()}

        base_logits = base_model(comment_enc["input_ids"], comment_enc["attention_mask"])
        gate_prob = torch.zeros(len(batch_df), dtype=torch.float32, device=device)
        final_logits = base_logits.clone()
        if adapter_model is not None:
            outputs = adapter_model(
                comment_input_ids=comment_enc["input_ids"],
                comment_attention_mask=comment_enc["attention_mask"],
                context_input_ids=context_enc["input_ids"],
                context_attention_mask=context_enc["attention_mask"],
            )
            gate_prob = outputs["gate_prob"] * title_present
            final_logits = outputs["base_logits"] + gate_prob.unsqueeze(-1) * outputs["delta_logits"]

        base_prob = torch.softmax(base_logits, dim=-1)
        final_prob = torch.softmax(final_logits, dim=-1)
        base_top_prob, base_top_idx = base_prob.max(dim=-1)
        final_top_prob, final_top_idx = final_prob.max(dim=-1)
        topk_prob, topk_idx = final_prob.topk(top_k, dim=-1)

        for row_idx in range(len(batch_df)):
            final_fine = id2label[int(final_top_idx[row_idx].item())]
            base_fine = id2label[int(base_top_idx[row_idx].item())]
            top5 = []
            for label_idx, score in zip(topk_idx[row_idx].tolist(), topk_prob[row_idx].tolist()):
                label = id2label[int(label_idx)]
                top5.append(
                    {
                        "label": label,
                        "major": small_to_major.get(label, ""),
                        "score": float(score),
                    }
                )
            result = {}
            for key in META_FIELDS:
                if key in batch_df.columns:
                    value = batch_df.iloc[row_idx][key]
                    if pd.notna(value) and str(value).strip():
                        result[key] = str(value).strip()
            result.update(
                {
                    "title": titles[row_idx],
                    "comment": comments[row_idx],
                    "title_provided": bool(titles[row_idx].strip()),
                    "gate_score": float(gate_prob[row_idx].item()),
                    "base_emotion_label_fine": base_fine,
                    "base_emotion_label_major": small_to_major.get(base_fine, ""),
                    "base_emotion_score": float(base_top_prob[row_idx].item()),
                    "final_emotion_label_fine": final_fine,
                    "final_emotion_label_major": small_to_major.get(final_fine, ""),
                    "final_emotion_score": float(final_top_prob[row_idx].item()),
                    "primary_emotion_label": final_fine,
                    "primary_emotion_major": small_to_major.get(final_fine, ""),
                    "primary_emotion_score": float(final_top_prob[row_idx].item()),
                    "top_5_json": json.dumps(top5, ensure_ascii=False),
                }
            )
            results.append(result)

    output_df = pd.DataFrame(results)
    output_path = Path(args.output)
    write_output(output_df, output_path, single_input)

    print(json.dumps({"rows": int(len(output_df)), "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
