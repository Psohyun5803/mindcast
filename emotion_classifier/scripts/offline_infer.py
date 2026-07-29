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

from sarcasm_emotion_adapter.labels import get_default_major_mapping_path
from sarcasm_emotion_adapter.offline import (
    OfflineSarcasmEmotionPredictor,
    read_bundle_metadata,
    resolve_bundle_checkpoint,
)

META_FIELDS = ("candidate_id", "date", "news_date", "dataset_date", "source_file")


def normalize_json_rows(raw) -> tuple[list[dict[str, str]], bool]:
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
            if "news_title" in df.columns:
                df["title"] = df["news_title"]
            else:
                df["title"] = ""
        keep_cols = [col for col in [*META_FIELDS, "title", "comment"] if col in df.columns]
        return df[keep_cols].to_dict("records"), False

    if args.input_json:
        return normalize_json_rows(json.loads(args.input_json))

    if not args.comment:
        raise ValueError("Either --input, --input-json, or --comment must be provided")
    row = {"title": args.title or "", "comment": args.comment}
    if args.date:
        row["date"] = args.date
    return [row], True


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-checkpoint", default=None)
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--bundle-pattern", default="offline_bundle*.pt")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--stageb-checkpoint", default=None)
    parser.add_argument("--student-model", default="beomi/KcELECTRA-base")
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--comment", default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--major-mapping", default=str(get_default_major_mapping_path()))
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    resolved_bundle = None
    bundle_meta = {}
    using_bundle_mode = args.bundle_checkpoint is not None or not args.base_checkpoint

    if using_bundle_mode:
        resolved_bundle = resolve_bundle_checkpoint(
            bundle_checkpoint=args.bundle_checkpoint,
            bundle_dir=args.bundle_dir,
            pattern=args.bundle_pattern,
        )
        bundle_meta = read_bundle_metadata(resolved_bundle)
        predictor = OfflineSarcasmEmotionPredictor.from_bundle(
            bundle_checkpoint=resolved_bundle,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
    else:
        if not args.base_checkpoint or not args.label_map:
            raise ValueError("Without bundle mode, both --base-checkpoint and --label-map are required")
        predictor = OfflineSarcasmEmotionPredictor(
            student_model=args.student_model,
            base_checkpoint=args.base_checkpoint,
            adapter_checkpoint=args.stageb_checkpoint,
            label_map_path=args.label_map,
            major_mapping_path=args.major_mapping,
            max_length=args.max_length,
            batch_size=args.batch_size,
            top_k=args.top_k,
            local_files_only=args.local_files_only,
        )

    rows, single_input = read_rows(args)
    result_df = predictor.predict(rows)

    output_path = Path(args.output)
    write_output(result_df, output_path, single_input)

    summary = {"rows": int(len(result_df)), "output": str(output_path)}
    if resolved_bundle is not None:
        summary["bundle_checkpoint"] = str(resolved_bundle)
        if bundle_meta.get("created_at_utc"):
            summary["bundle_created_at_utc"] = bundle_meta["created_at_utc"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
