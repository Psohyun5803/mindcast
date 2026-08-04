from __future__ import annotations

import json
from pathlib import Path

KOTE_LABELS = [
    "불평/불만",
    "환영/호의",
    "감동/감탄",
    "지긋지긋",
    "고마움",
    "슬픔",
    "화남/분노",
    "존경",
    "기대감",
    "우쭐댐/무시함",
    "안타까움/실망",
    "비장함",
    "의심/불신",
    "뿌듯함",
    "편안/쾌적",
    "신기함/관심",
    "아껴주는",
    "부끄러움",
    "공포/무서움",
    "절망",
    "한심함",
    "역겨움/징그러움",
    "짜증",
    "어이없음",
    "없음",
    "패배/자기혐오",
    "귀찮음",
    "힘듦/지침",
    "즐거움/신남",
    "깨달음",
    "죄책감",
    "증오/혐오",
    "흐뭇함(귀여움/예쁨)",
    "당황/난처",
    "경악",
    "부담/안_내킴",
    "서러움",
    "재미없음",
    "불쌍함/연민",
    "놀람",
    "행복",
    "불안/걱정",
    "기쁨",
    "안심/신뢰",
]


def get_default_label_map_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "emotion_classifier" / "assets" / "kote_id2label.json"


def get_default_major_mapping_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "emotion_classifier" / "assets" / "mapping_ver1.json"


def load_major_mapping(path: str | Path | None = None) -> dict[str, list[str]]:
    mapping_path = Path(path) if path is not None else get_default_major_mapping_path()
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_small_to_major(path: str | Path | None = None) -> dict[str, str]:
    major_to_small = load_major_mapping(path)
    small_to_major: dict[str, str] = {}
    for major, labels in major_to_small.items():
        for label in labels:
            if label in small_to_major and small_to_major[label] != major:
                raise ValueError(f"duplicate mapping for label: {label}")
            small_to_major[label] = major
    return small_to_major
