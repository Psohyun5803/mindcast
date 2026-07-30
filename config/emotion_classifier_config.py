"""emotion_classifier 기본 경로 설정.

환경변수로 덮어쓸 수 있으며, 지정하지 않으면 여기 정의된 경로를 사용한다.
"""
from __future__ import annotations

import os
from pathlib import Path

# personalrepo/ 루트
ROOT = Path(__file__).resolve().parents[1]

# ── 학습 경로 ───────────────────────────────────────────────── (미지정시 기본경로)
DATA_DIR  = Path(os.environ.get("EC_DATA_DIR",  str(ROOT / "data"   / "emotion_classifier")))
MODEL_DIR = Path(os.environ.get("EC_MODEL_DIR", str(ROOT / "models" / "emotion_classifier")))

DEFAULT_TEACHER_OUT       = Path(os.environ.get("EC_TEACHER_OUT",       str(DATA_DIR  / "teacher_targets.parquet")))
DEFAULT_STAGEA_NORMALIZED = Path(os.environ.get("EC_STAGEA_NORMALIZED", str(DATA_DIR  / "stagea_normalized.parquet")))
DEFAULT_STAGEA_DIR        = Path(os.environ.get("EC_STAGEA_DIR",        str(MODEL_DIR / "stage_a")))
DEFAULT_STAGEB_NORMALIZED = Path(os.environ.get("EC_STAGEB_NORMALIZED", str(DATA_DIR  / "stageb_normalized.parquet")))
DEFAULT_STAGEB_DIR        = Path(os.environ.get("EC_STAGEB_DIR",        str(MODEL_DIR / "stage_b")))
DEFAULT_BUNDLE_OUT        = Path(os.environ.get("EC_BUNDLE_OUT",        str(MODEL_DIR / "offline_bundle.pt")))

# ── 추론 경로 ─────────────────────────────────────────────────
OUTPUTS_DIR = DATA_DIR / "outputs"

# 번들 파일을 지정하지 않았을 때 자동 탐색하는 디렉토리 (우선순위 순)
DEFAULT_BUNDLE_SEARCH_DIRS = [MODEL_DIR, OUTPUTS_DIR]

# 추론 결과 기본 저장 디렉토리
DEFAULT_PREDICTION_DIR = OUTPUTS_DIR

# ── HuggingFace 기본 소스 ────────────────────────────────────
DEFAULT_HF_SOURCE = os.environ.get(
    "EC_HF_SOURCE", "MindCastSogang/Youtube_news_preprocessed_data"
)
