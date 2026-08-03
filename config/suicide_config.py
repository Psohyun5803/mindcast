"""All experiment constants, paths, and configurations."""
from pathlib import Path
import numpy as np

ROOT        = Path(__file__).resolve().parents[1] / "data" / "suicide_predict"
EMO         = ROOT / "cache" / "emotion"
TOP         = ROOT / "cache" / "topic"
RAW         = ROOT / "cache" / "raw"
DATA        = ROOT / "data"
OUT         = ROOT / "outputs"
DELIVERABLE = ROOT / "deliverable"
CK          = OUT / "checkpoints"
PRED_MAIN   = ROOT / "prediction" / "main"
PRED_MONTH  = ROOT / "prediction" / "monthly"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# ── Dataset ────────────────────────────────────────────────────────────
N_EMO = 44

SELECTED_EMO = [
    "불안/걱정", "절망", "슬픔", "서러움", "불쌍함/연민",
    "짜증", "화남/분노", "증오/혐오", "안타까움/실망", "의심/불신",
]

GROUP_ORDER = ["기쁨", "슬픔", "분노", "중립"]

EMO_GROUPS = {
    "기쁨": ["환영/호의","감동/감탄","고마움","존경","기대감","뿌듯함","편안/쾌적",
             "신기함/관심","아껴주는","즐거움/신남","흐뭇함(귀여움/예쁨)","행복","기쁨","안심/신뢰"],
    "슬픔": ["슬픔","안타까움/실망","부끄러움","절망","패배/자기혐오","귀찮음","힘듦/지침",
             "죄책감","당황/난처","부담/안_내킴","서러움","재미없음","불쌍함/연민","불안/걱정"],
    "분노": ["불평/불만","지긋지긋","화남/분노","의심/불신","한심함","역겨움/징그러움",
             "짜증","어이없음","증오/혐오","경악"],
    "중립": ["우쭐댐/무시함","공포/무서움","비장함","없음","깨달음","놀람"],
} #mapping version 1 


# ── Training ────────────────────────────────────────────────────────────
SEEDS  = [42, 1, 7, 2025]
LR     = 5e-4
WD     = 5e-4
COMMON = dict(hidden=64, dropout=0.5, emotion_dropout=0.3, emotion_mode="group")

# ── Ablation settings ──────────────────────────────────────────────────
ABL = {
    "A": dict(use_calendar=False, use_volume=False, use_emotion=False, use_topic=False, use_interaction=False),
    "B": dict(use_calendar=True,  use_volume=False, use_emotion=False, use_topic=False, use_interaction=False),
    "C": dict(use_calendar=True,  use_volume=True,  use_emotion=False, use_topic=False, use_interaction=False),
    "D": dict(use_calendar=True,  use_volume=True,  use_emotion=True,  use_topic=False, use_interaction=False),
    "E": dict(use_calendar=True,  use_volume=True,  use_emotion=True,  use_topic=True,  use_interaction=False),
    "F": dict(use_calendar=True,  use_volume=True,  use_emotion=True,  use_topic=True,  use_interaction=True),
}
ABL_DESC = {
    "A": "target", "B": "+calendar", "C": "+volume",
    "D": "+emotion(4grp)", "E": "+topic", "F": "+emo×topic",
}
METRICS = ["MAE", "RMSE", "MASE", "WAPE", "sMAPE", "NLL", "CRPS",
           "Cov80", "Cov90", "Width80", "Width90"]

# ── Main model (일별 예측기) ────────────────────────────────────────────
MAIN_CFG = dict(
    model="exodlinear", lookback=56, head="point", emotion_mode="group",
    hidden=64, dropout=0.5, emotion_dropout=0.0,
    use_calendar=False, use_volume=False, use_emotion=False,
    use_topic=False, use_interaction=False,
)

# ── 전처리: 상담전화 건수 (target) ──────────────────────────────────────
CALL_REPO         = "MindCastSogang/suicide_prevent_call"

# ── 전처리: 댓글 데이터 ────────────────────────────────────────────────
COMMENT_REPO      = "MindCastSogang/Youtube_news_preprocessed_data"
COMMENT_BASE_DIR  = "preprocessed/v1"

# ── 전처리: KOTE 감정 추론 ───────────────────────────────────────────────
KOTE_MODEL  = "searle-j/kote_for_easygoing_people"
KOTE_BATCH  = 512
KOTE_MAXLEN = 128

# ── 전처리: 토픽 클러스터링 ──────────────────────────────────────────────

# ── Monthly suicide (월별 예측기) ────────────────────────────────────────
REPO     = "MindCastSogang/SuicideDataset"
BASE_CSV = "suicide_base_data_2020_2024_20260222.csv"
K_TOPIC  = 10
ALPHAS   = np.logspace(-2, 4, 25)
