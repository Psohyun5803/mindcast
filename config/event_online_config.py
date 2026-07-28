"""All constants, environment variables, and parameters for mindcast-online."""
import os
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1] / "data" / "event_online"
DATA  = ROOT / "data"
OUT   = ROOT / "outputs"
FIGS  = ROOT / "figures"
STAGE = ROOT / "hf_staging"

# ── Environment variables ───────────────────────────────────────────────────
HF_REPO = os.environ.get("MINDCAST_HF_REPO", "MindCastSogang/mindcast-news-events")
DUMP    = os.environ.get("MINDCAST_DUMP", "")        # local sql.gz path (source=local)
DEVICE  = os.environ.get("MINDCAST_DEVICE", "")      # forced embedding device
FONT    = os.environ.get("MINDCAST_FONT", "")        # Korean font override for viz

# ── Datasource: table schemas (mysqldump column order) ─────────────────────
SCHEMA = {
    "video_video":   ["id","video_id","video_url","title","description","duration",
                      "thumbnail_url","tags","like_count","comment_count","view_count",
                      "created_at","updated_at","collected_at","channel_id"],
    "video_channel": ["id","channel_id","title","description","subscriber_count",
                      "video_count","view_count","thumbnail_url","created_at","is_main","region"],
    "video_comment": ["id","comment_id","author","text","like_count","created_at","video_id"],
}
NUMERIC = {
    "video_video":   ["like_count","comment_count","view_count","channel_id"],
    "video_channel": ["subscriber_count","video_count","view_count","is_main"],
    "video_comment": ["like_count","video_id"],
}
DATETIME = {
    "video_video":   ["created_at","collected_at","updated_at"],
    "video_channel": ["created_at"],
    "video_comment": ["created_at"],
}
# mysqldump escape sequences
UNESCAPE = {"\\n":"\n","\\r":"\r","\\t":"\t","\\0":"\0","\\Z":"\x1a",
            "\\'":"'",'\\"':'"',"\\\\":"\\"}

# ── Prep ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
EMBED_BATCH = 128

# boilerplate hashtags to filter during prep (editorial/channel tags)
HASHTAG_BOILER = {
    "MBC","뉴스","뉴스데스크","뉴스투데이","MBC뉴스","뉴스ZIP","뉴스꾹","오늘이뉴스",
    "자막뉴스","iMBC","뉴스25","SBS","KBS","뉴스A","속보","단독","현장",
}

# ── Common: channel/format stop-words for entity filtering ──────────────────
STOP = {
    "MBC","뉴스","뉴스데스크","뉴스투데이","MBC뉴스","뉴스ZIP","뉴스꾹","오늘이뉴스",
    "자막뉴스","iMBC","뉴스25","SBS","KBS","뉴스A","속보","단독","현장","직캠",
    "TBC","TBC뉴스","TBC8뉴스","부산MBC","대구MBC","광주MBC","대전MBC","제주MBC",
    "부산문화방송","부산엠비씨","뉴스영상","풀영상","다시보기","LIVE","날씨","기자",
    "정치in직캠","이슈","오늘","현장영상","영상","포토","화제",
}

# ── Tracker: offline (provenance only) and online params ────────────────────
P_OFFLINE = dict(
    min_cluster_size=3, min_samples=1, selection="leaf", window=1,
    tau_link=0.38, w_cos=0.45, w_ent=0.40, w_key=0.15,
    gap_days=6, dead_days=7, ema=0.6, merge_gap=3, merge_thr=0.6,
)

P_ONLINE = dict(
    tau=0.34, K=2, w_cos=0.6, w_ent=0.4, topN=12, ema=0.6,
    min_cluster_size=4, min_samples=1, residual_window=3,
    active_gap=3, dormant_gap=7, archive_gap=30,
    merge_cos=0.5, merge_thr=0.55,
    cons_cos=0.72, cons_ov=0.60, cons_gap=3,
    confirm_days=2, confirm_comments=3000,
)

# ── Viz: Korean font search candidates ─────────────────────────────────────
FONT_CANDIDATES = [
    FONT,
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
]
FONT_NAMES = ["Noto Sans CJK KR","NanumGothic","Malgun Gothic","AppleGothic"]

# ── Export HTML ─────────────────────────────────────────────────────────────
MONTHS      = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-03"]
GAP_ACTIVE  = 3
GAP_DORMANT = 7
CAP         = 20      # max news titles embedded per event per day
COH_THR     = 0.55    # median NN cosine below this → 기타(저응집)
FRAC        = 0.30    # AND-label: entity must appear in >= this fraction of members

# ── HF upload ───────────────────────────────────────────────────────────────
UPLOAD_TABLES  = ["video_video", "video_channel", "video_comment"]
FORBIDDEN_COLS = {
    "password","pw_display","userid","email","session_data",
    "token","jti","first_name","last_name",
}

HF_CARD = """---
license: cc-by-nc-4.0
language: [ko]
tags: [news, event-tracking, korean, comments]
pretty_name: Mindcast Korean News + Comments (sanitized)
---

# Mindcast Korean News Events (sanitized, private)

한국 뉴스 영상 메타데이터와 댓글. **개인정보 정제본**:
- 크리덴셜/어드민 테이블(계정·비밀번호·세션·토큰) **전부 제외**.
- 댓글 작성자(`author`)는 **SHA-256(16 hex)로 가명화** — 원본 @핸들 없음.

## Files
- `video_video.parquet` — 뉴스 영상 (title/description/tags/counts/dates/channel).
- `video_channel.parquet` — 채널 메타(언론사).
- `video_comment.parquet` — 댓글 (`author`=가명 해시, `text`, `like_count`, `created_at`, `video_id`).

## Usage
```python
from event_online_utils import load_table
vv = load_table("video_video", source="hf")   # source만 교체, 이후 처리 동일
```
"""
