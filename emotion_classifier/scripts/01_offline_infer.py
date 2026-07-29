#!/usr/bin/env python3
"""오프라인 추론: 번들 체크포인트로 감정 예측"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emotion_classifier_utils import read_rows, write_output
from sarcasm_emotion_adapter.labels import get_default_major_mapping_path
from sarcasm_emotion_adapter.offline import (
    OfflineSarcasmEmotionPredictor, read_bundle_metadata, resolve_bundle_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-checkpoint", default=None)
    parser.add_argument("--bundle-dir",        default=None)
    parser.add_argument("--bundle-pattern",    default="offline_bundle*.pt")
    parser.add_argument("--base-checkpoint",   default=None)
    parser.add_argument("--stageb-checkpoint", default=None)
    parser.add_argument("--student-model",     default="beomi/KcELECTRA-base")
    parser.add_argument("--label-map",         default=None)
    parser.add_argument("--input",             default=None)
    parser.add_argument("--input-json",        default=None)
    parser.add_argument("--comment",           default=None)
    parser.add_argument("--title",             default="")
    parser.add_argument("--date",              default="")
    parser.add_argument("--output",            required=True)
    parser.add_argument("--major-mapping",     default=str(get_default_major_mapping_path()))
    parser.add_argument("--max-length",        type=int, default=192)
    parser.add_argument("--batch-size",        type=int, default=32)
    parser.add_argument("--top-k",             type=int, default=5)
    parser.add_argument("--local-files-only",  action="store_true")
    args = parser.parse_args()

    using_bundle = args.bundle_checkpoint is not None or not args.base_checkpoint
    resolved_bundle = bundle_meta = None

    if using_bundle:
        resolved_bundle = resolve_bundle_checkpoint(bundle_checkpoint=args.bundle_checkpoint,
                                                    bundle_dir=args.bundle_dir, pattern=args.bundle_pattern)
        bundle_meta = read_bundle_metadata(resolved_bundle)
        predictor = OfflineSarcasmEmotionPredictor.from_bundle(
            bundle_checkpoint=resolved_bundle, batch_size=args.batch_size,
            local_files_only=args.local_files_only)
    else:
        if not args.base_checkpoint or not args.label_map:
            raise ValueError("Without bundle mode, --base-checkpoint and --label-map are required")
        predictor = OfflineSarcasmEmotionPredictor(
            student_model=args.student_model, base_checkpoint=args.base_checkpoint,
            adapter_checkpoint=args.stageb_checkpoint, label_map_path=args.label_map,
            major_mapping_path=args.major_mapping, max_length=args.max_length,
            batch_size=args.batch_size, top_k=args.top_k, local_files_only=args.local_files_only)

    rows, single_input = read_rows(args)
    result_df = predictor.predict(rows)
    write_output(result_df, Path(args.output), single_input)

    summary = {"rows": int(len(result_df)), "output": args.output}
    if resolved_bundle:
        summary["bundle_checkpoint"] = str(resolved_bundle)
        if bundle_meta and bundle_meta.get("created_at_utc"):
            summary["bundle_created_at_utc"] = bundle_meta["created_at_utc"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
