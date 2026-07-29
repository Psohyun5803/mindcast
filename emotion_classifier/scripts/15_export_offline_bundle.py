#!/usr/bin/env python3
"""번들 내보내기: Stage A/B 체크포인트 → offline_bundle.pt"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sarcasm_emotion_adapter.labels import get_default_label_map_path, get_default_major_mapping_path
from sarcasm_emotion_adapter.offline import export_offline_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint",  required=True)
    parser.add_argument("--stageb-checkpoint",default=None)
    parser.add_argument("--student-model",    default="beomi/KcELECTRA-base")
    parser.add_argument("--label-map",        default=str(get_default_label_map_path()))
    parser.add_argument("--major-mapping",    default=str(get_default_major_mapping_path()))
    parser.add_argument("--max-length",       type=int, default=192)
    parser.add_argument("--output",           required=True)
    args = parser.parse_args()

    bundle_path = export_offline_bundle(
        bundle_path=args.output, student_model=args.student_model,
        base_checkpoint=args.base_checkpoint, label_map_path=args.label_map,
        adapter_checkpoint=args.stageb_checkpoint, major_mapping_path=args.major_mapping,
        max_length=args.max_length)

    variant = "stagea_stageb" if args.stageb_checkpoint else "stagea_only"
    print(json.dumps({"bundle": str(bundle_path), "bundle_variant": variant},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
