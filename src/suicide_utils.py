"""Consolidated utilities: data, models, metrics, training loop, suite runner,
and monthly suicide helpers.

Sections
--------
1. Data / Dataset      (was dataset.py)
2. Models              (was models.py)
3. Metrics             (was metrics.py)
4. Training            (was train.py)
5. Suite runner        (was suite_lib.py)
6. Monthly suicide     (utility functions from 13_monthly_suicide.py)
"""
# ── 경로 설정 (config/ 폴더를 sys.path에 추가) ─────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
# ────────────────────────────────────────────────────────────────────────────
import copy, glob, hashlib, json, os
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import properscoring as ps
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from torch.distributions import NegativeBinomial
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from suicide_config import (
    EMO, TOP, RAW, DATA, OUT, CK,
    N_EMO, SELECTED_EMO, GROUP_ORDER, EMO_GROUPS,
    SEEDS, LR, WD, COMMON, ABL, ABL_DESC, METRICS,
    CALL_REPO,
    COMMENT_REPO, COMMENT_BASE_DIR,
    KOTE_MODEL, KOTE_BATCH, KOTE_MAXLEN,
    REPO, BASE_CSV, K_TOPIC, ALPHAS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DATA_CACHE: dict = {}

# ensure output sub-dirs exist at import time
for _d in ["metrics", "predictions", "checkpoints", "parts"]:
    (OUT / _d).mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# 0.  PREPROCESSING  (was 01~04번 스크립트)
# ════════════════════════════════════════════════════════════════════════

def _compute_splits(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """교집합 날짜 범위에서 train/valid/test split 자동 계산.
    마지막 1년 = test, 그 전 1년 = valid, 나머지 = train.
    """
    test_end   = end
    test_start = (test_end - pd.DateOffset(years=1) + pd.Timedelta(days=1))
    valid_end  = test_start - pd.Timedelta(days=1)
    valid_start = (valid_end - pd.DateOffset(years=1) + pd.Timedelta(days=1))
    train_start = start
    train_end   = valid_start - pd.Timedelta(days=1)
    return {
        "train": (str(train_start.date()), str(train_end.date())),
        "valid": (str(valid_start.date()), str(valid_end.date())),
        "test":  (str(test_start.date()),  str(test_end.date())),
    }


def load_splits() -> dict:
    """preprocess 완료 후 저장된 splits.json을 읽어 반환."""
    path = RAW / "splits.json"
    if not path.exists():
        raise FileNotFoundError(
            "splits.json이 없습니다. 먼저 preprocess를 실행하세요:\n"
            "  ./suicide_run.sh preprocess"
        )
    return json.load(open(path))


def build_target():
    """01: HuggingFace → cache/raw/target_call_counts.parquet + target_meta.json"""
    files = [f for f in list_repo_files(repo_id=CALL_REPO, repo_type="dataset")
             if f.endswith(".csv")]
    print(f"[target] {len(files)} csv files from {CALL_REPO}")
    rows = []
    for f in sorted(files):
        lp = hf_hub_download(repo_id=CALL_REPO, repo_type="dataset", filename=f)
        df = pd.read_csv(lp, encoding="cp949")
        df.columns = ["year", "month", "day", "y"]
        rows.append(df)
    t = pd.concat(rows, ignore_index=True)
    t["date"] = pd.to_datetime(dict(year=t.year, month=t.month, day=t.day))
    t = t[["date", "y"]].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "target_call_counts.parquet"
    t.to_parquet(out)
    meta = {"start": str(t.date.min().date()), "end": str(t.date.max().date())}
    json.dump(meta, open(RAW / "target_meta.json", "w"))
    print(f"[target] {len(t)} days {meta['start']}..{meta['end']} "
          f"y range {t.y.min()}-{t.y.max()} mean {t.y.mean():.1f}")
    return out


def build_comments():
    """02: HuggingFace 댓글 JSON → cache/raw/youtube_news_comments.parquet
    target_meta.json의 날짜 범위를 읽어 HF에서 가용한 연도와 교집합만 다운로드.
    전처리 후 splits.json 자동 계산·저장.
    """
    # target 범위 읽기
    meta_path = RAW / "target_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "target_meta.json이 없습니다. build_target()을 먼저 실행하세요."
        )
    meta = json.load(open(meta_path))
    target_start = pd.Timestamp(meta["start"])
    target_end   = pd.Timestamp(meta["end"])

    # HF에서 가용 연도 자동 탐지
    all_files = list(list_repo_files(repo_id=COMMENT_REPO, repo_type="dataset"))
    available_years = sorted({
        int(p[2]) for f in all_files
        if COMMENT_BASE_DIR in f and f.endswith("news_comments.json")
        for p in [f.split("/")]
        if len(p) > 2 and p[2].isdigit()
    })
    years = [y for y in available_years
             if target_start.year <= y <= target_end.year]
    print(f"[comments] HF 가용 연도: {available_years}  →  교집합: {years}")

    files = sorted(
        f for f in all_files
        if COMMENT_BASE_DIR in f and f.endswith("news_comments.json")
        and any(f"/{y}/" in f for y in years)
    )
    print(f"[comments] {len(files)} json files to process")
    recs = []
    for f in tqdm(files, desc="parse"):
        lp = hf_hub_download(repo_id=COMMENT_REPO, repo_type="dataset", filename=f)
        data = json.load(open(lp))["data"]
        for day in data:
            date = day["date"]
            for pi, post in enumerate(day.get("posts", [])):
                title = post.get("title") or post.get("raw_title") or ""
                for c in post.get("comments", []):
                    if not c or not str(c).strip():
                        continue
                    recs.append((date, str(c), title, pi))
    df = pd.DataFrame(recs, columns=["date", "text", "title", "post_idx"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["comment_id"] = [hashlib.md5((r.date.isoformat() + str(i)).encode()).hexdigest()[:16]
                        for i, r in enumerate(df.itertuples())]
    df = df[["comment_id", "date", "text", "title", "post_idx"]]
    out = RAW / "youtube_news_comments.parquet"
    df.to_parquet(out)

    # 교집합 범위로 splits 자동 계산
    comment_start = df.date.min()
    comment_end   = df.date.max()
    intersect_start = max(target_start, comment_start)
    intersect_end   = min(target_end,   comment_end)
    splits = _compute_splits(intersect_start, intersect_end)
    json.dump(splits, open(RAW / "splits.json", "w"), indent=2)

    print(f"[comments] {len(df):,} comments {comment_start.date()}..{comment_end.date()}")
    print(f"[splits] 교집합 {intersect_start.date()}..{intersect_end.date()}")
    print(f"  train: {splits['train'][0]} ~ {splits['train'][1]}")
    print(f"  valid: {splits['valid'][0]} ~ {splits['valid'][1]}")
    print(f"  test:  {splits['test'][0]}  ~ {splits['test'][1]}")
    return out


def kote_inference(device=None):
    """02: 댓글별 KOTE 44감정 추론 → cache/emotion/comment_kote_probs.parquet"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = device or DEVICE
    EMO.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(RAW / "youtube_news_comments.parquet")
    texts = df["text"].astype(str).tolist()
    n = len(texts)
    print(f"[kote] {n:,} comments | device {dev}")

    tok   = AutoTokenizer.from_pretrained(KOTE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(KOTE_MODEL).to(dev).half().eval()
    id2label = model.config.id2label
    labels   = [id2label[i] for i in range(len(id2label))]
    json.dump(labels, open(EMO / "emotion_labels.json", "w"), ensure_ascii=False, indent=2)
    n_emo = len(labels)

    probs = np.zeros((n, n_emo), dtype=np.float16)
    with torch.no_grad():
        for s in tqdm(range(0, n, KOTE_BATCH), desc="kote"):
            batch = texts[s:s + KOTE_BATCH]
            enc   = tok(batch, padding=True, truncation=True, max_length=KOTE_MAXLEN,
                        return_tensors="pt").to(dev)
            p = torch.sigmoid(model(**enc).logits).float().cpu().numpy()
            probs[s:s + len(batch)] = p.astype(np.float16)

    pf      = probs.astype(np.float32)
    top_id  = pf.argmax(1).astype(np.int16)
    top_conf = pf.max(1).astype(np.float16)
    pn      = pf / (pf.sum(1, keepdims=True) + 1e-8)
    ent     = (-(pn * np.log(pn + 1e-8)).sum(1)).astype(np.float16)

    out = pd.DataFrame(probs, columns=[f"emotion_{i}" for i in range(n_emo)])
    out.insert(0, "comment_id", df["comment_id"].values)
    out.insert(1, "date",       df["date"].values)
    out["top_emotion_id"]         = top_id
    out["top_emotion_confidence"] = top_conf
    out["emotion_entropy"]        = ent
    path = EMO / "comment_kote_probs.parquet"
    out.to_parquet(path)
    print(f"[kote] saved {path} shape {out.shape}")
    return path


def daily_emotion():
    """03: 댓글별 감정 → 일별 집계 → cache/emotion/daily_kote_features.parquet"""
    if not (EMO / "comment_kote_probs.parquet").exists():
        raise FileNotFoundError(
            "\n\033[31m[ERROR] 전처리 파일이 없습니다. 먼저 kote_inference를 실행하세요:\033[0m\n"
            "\033[33m  ./suicide_run.sh preprocess\033[0m"
        )
    df = pd.read_parquet(EMO / "comment_kote_probs.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    ecols = [f"emotion_{i}" for i in range(N_EMO)]
    df[ecols] = df[ecols].astype(np.float32)
    g = df.groupby("date")

    mean = g[ecols].mean(); mean.columns = [f"emo_mean_{i}" for i in range(N_EMO)]
    s    = g[ecols].sum();  s.columns    = [f"emo_sum_{i}"  for i in range(N_EMO)]
    mx   = g[ecols].max();  mx.columns   = [f"emo_max_{i}"  for i in range(N_EMO)]
    std  = g[ecols].std().fillna(0.0); std.columns = [f"emo_std_{i}" for i in range(N_EMO)]
    ent  = g["emotion_entropy"].agg(["mean", "std"]).fillna(0.0)
    ent.columns = ["ent_mean", "ent_std"]

    out = pd.concat([pd.DataFrame({"comment_count": g.size()}), mean, s, mx, std, ent], axis=1)
    out = out.reset_index().sort_values("date").reset_index(drop=True)
    path = EMO / "daily_kote_features.parquet"
    out.to_parquet(path)
    print(f"[daily-emo] {out.shape} days {out.date.min().date()}..{out.date.max().date()}")
    return path


def topic_features():
    """04: 뉴스 제목 TF-IDF + KMeans → cache/topic/daily_topic_features.parquet"""
    if not (RAW / "youtube_news_comments.parquet").exists():
        raise FileNotFoundError(
            "\n\033[31m[ERROR] 전처리 파일이 없습니다. 먼저 build_comments를 실행하세요:\033[0m\n"
            "\033[33m  ./suicide_run.sh preprocess\033[0m"
        )
    TOP.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "youtube_news_comments.parquet", columns=["date", "title"])
    df["date"]  = pd.to_datetime(df["date"]).dt.normalize()
    df["title"] = df["title"].fillna("").astype(str)

    uniq = df.drop_duplicates("title")[["date", "title"]].reset_index(drop=True)
    topic_train_end = load_splits()["train"][1]
    train_titles = uniq.loc[uniq.date <= topic_train_end, "title"]
    train_titles = train_titles[train_titles.str.len() > 0]
    print(f"[topic] {len(uniq)} unique titles, {len(train_titles)} train titles")

    vec = TfidfVectorizer(max_features=20000, min_df=3, token_pattern=r"(?u)\b\w\w+\b")
    Xtr = vec.fit_transform(train_titles)
    km  = KMeans(n_clusters=K_TOPIC, n_init=5, random_state=42).fit(Xtr)

    Xall = vec.transform(uniq["title"])
    uniq["topic"] = km.predict(Xall)
    title2topic = dict(zip(uniq["title"], uniq["topic"]))
    df["topic"] = df["title"].map(title2topic).fillna(0).astype(int)

    ct    = df.groupby(["date", "topic"]).size().unstack(fill_value=0).reindex(columns=range(K_TOPIC), fill_value=0)
    total = ct.sum(1).clip(lower=1)
    ratio = ct.div(total, axis=0)
    ratio.columns = [f"topic_ratio_{k}" for k in range(K_TOPIC)]

    out = ratio.copy()
    out["dominant_topic"]       = ct.idxmax(1).values
    out["dominant_topic_ratio"] = ratio.max(1).values
    out["topic_entropy"]        = -(ratio.values * np.log(ratio.values + 1e-8)).sum(1)
    out = out.reset_index()
    path = TOP / "daily_topic_features.parquet"
    out.to_parquet(path)
    print(f"[topic] {out.shape} days {out.date.min().date()}..{out.date.max().date()}")
    return path


# ════════════════════════════════════════════════════════════════════════
# 1.  DATA / DATASET
# ════════════════════════════════════════════════════════════════════════

def _calendar(dates):
    kr = holidays.SouthKorea(years=range(2019, 2025))
    d = pd.DatetimeIndex(dates)
    wd    = np.eye(7)[d.weekday.values]
    mo    = np.eye(12)[d.month.values - 1]
    hol   = np.array([1.0 if x in kr else 0.0 for x in d.date]).reshape(-1, 1)
    doy   = d.dayofyear.values
    sin_y = np.sin(2 * np.pi * doy / 365.25).reshape(-1, 1)
    cos_y = np.cos(2 * np.pi * doy / 365.25).reshape(-1, 1)
    sin_w = np.sin(2 * np.pi * d.weekday.values / 7).reshape(-1, 1)
    cos_w = np.cos(2 * np.pi * d.weekday.values / 7).reshape(-1, 1)
    return np.concatenate([wd, mo, hol, sin_y, cos_y, sin_w, cos_w], axis=1).astype(np.float32)


def build_frame(emotion_mode="group"):
    missing = [p for p in [
        EMO / "emotion_labels.json",
        RAW / "target_call_counts.parquet",
        EMO / "daily_kote_features.parquet",
        TOP / "daily_topic_features.parquet",
        RAW / "splits.json",
    ] if not p.exists()]
    if missing:
        names = "\n  ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"\n\033[31m[ERROR] 전처리 파일이 없습니다. 먼저 preprocess를 실행하세요:\033[0m\n"
            f"\033[33m  ./suicide_run.sh preprocess\033[0m\n\n"
            f"  누락된 파일:\n  {names}"
        )
    labels = json.load(open(EMO / "emotion_labels.json"))
    target = pd.read_parquet(RAW / "target_call_counts.parquet")
    target["date"] = pd.to_datetime(target["date"]).dt.normalize()
    emo = pd.read_parquet(EMO / "daily_kote_features.parquet")
    emo["date"] = pd.to_datetime(emo["date"]).dt.normalize()
    top = pd.read_parquet(TOP / "daily_topic_features.parquet")
    top["date"] = pd.to_datetime(top["date"]).dt.normalize()

    start, end = emo["date"].min(), emo["date"].max()
    frame = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    frame = frame.merge(target, on="date", how="left")
    frame = frame.merge(emo,    on="date", how="left")
    frame = frame.merge(top,    on="date", how="left")

    frame["comment_count"] = frame["comment_count"].fillna(0.0)
    emean = [f"emo_mean_{i}" for i in range(N_EMO)]
    for c in emean:
        frame[c] = frame[c].fillna(0.0)
    tcols = [f"topic_ratio_{k}" for k in range(top.shape[1]) if f"topic_ratio_{k}" in top.columns]
    for c in tcols:
        frame[c] = frame[c].fillna(1.0 / len(tcols))
    frame["y"] = frame["y"].interpolate().ffill().bfill()
    K = len(tcols)

    if emotion_mode == "group":
        mapping = EMO_GROUPS
        gcols = []
        for g in GROUP_ORDER:
            members = [labels.index(e) for e in mapping[g] if e in labels]
            col = f"emogrp_{g}"
            frame[col] = frame[[f"emo_mean_{i}" for i in members]].mean(axis=1).values
            gcols.append(col)
        ecols = gcols
        sel_idx = list(range(len(GROUP_ORDER)))
    else:
        ecols = emean
        sel_idx = [labels.index(e) for e in SELECTED_EMO if e in labels]

    inter_cols = []
    new_cols = {}
    for j, ei in enumerate(sel_idx):
        base_col = ecols[ei] if emotion_mode == "group" else f"emo_mean_{ei}"
        for k in range(K):
            col = f"inter_{j}_{k}"
            new_cols[col] = frame[base_col].values * frame[f"topic_ratio_{k}"].values
            inter_cols.append(col)
    frame = pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1)

    return frame, labels if emotion_mode != "group" else GROUP_ORDER, ecols, tcols, inter_cols, K


def _standardize_fit(arr):
    mu = arr.mean(0); sd = arr.std(0); sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def make_datasets(lookback=56, horizon=1, emotion_mode="group"):
    frame, labels, ecols, tcols, inter_cols, K = build_frame(emotion_mode)
    dates = frame["date"]
    y     = frame["y"].values.astype(np.float32)
    logy  = np.log1p(y)
    vol   = np.log1p(frame["comment_count"].values.astype(np.float32))
    E     = frame[ecols].values.astype(np.float32)
    T     = frame[tcols].values.astype(np.float32)
    INT   = frame[inter_cols].values.astype(np.float32)
    CAL   = _calendar(dates)

    n = len(frame)

    splits = load_splits()
    def split_of(d):
        for name, (a, b) in splits.items():
            if pd.Timestamp(a) <= d <= pd.Timestamp(b):
                return name
        return None

    samples = {"train": [], "valid": [], "test": []}
    for t in range(lookback, n - horizon + 1):
        s = split_of(dates.iloc[t])
        if s is not None:
            samples[s].append(t)

    train_rows = set()
    for t in samples["train"]:
        train_rows.update(range(t - lookback, t))
    tr = sorted(train_rows)
    y_mu, y_sd = _standardize_fit(logy[tr].reshape(-1, 1))
    v_mu, v_sd = _standardize_fit(vol[tr].reshape(-1, 1))
    e_mu, e_sd = _standardize_fit(E[tr])

    logy_n = (logy.reshape(-1, 1) - y_mu) / y_sd
    vol_n  = (vol.reshape(-1, 1)  - v_mu) / v_sd
    E_n    = (E - e_mu) / e_sd

    def pack(idxs):
        out = {k: [] for k in ["y_past", "v_past", "e_past", "t_past", "inter",
                                "c_future", "y_future", "y_base", "tdate"]}
        for t in idxs:
            sl = slice(t - lookback, t)
            out["y_past"].append(logy_n[sl])
            out["v_past"].append(vol_n[sl])
            out["e_past"].append(E_n[sl])
            out["t_past"].append(T[sl])
            out["inter"].append(INT[sl])
            out["c_future"].append(CAL[t:t + horizon])
            out["y_future"].append(y[t:t + horizon])
            out["y_base"].append([max(1.0, float(y[t - 7:t].mean()))])
            out["tdate"].append(dates.iloc[t].value)
        ds = {k: np.asarray(v, dtype=np.float32) for k, v in out.items() if k != "tdate"}
        ds["tdate"] = np.asarray(out["tdate"], dtype=np.int64)
        return ds

    data = {s: pack(samples[s]) for s in samples}
    meta = dict(lookback=lookback, horizon=horizon, n_emotion=len(ecols), n_topic=K,
                n_inter=len(inter_cols), n_calendar=CAL.shape[1], labels=labels,
                emotion_mode=emotion_mode, y_mu=float(y_mu[0]), y_sd=float(y_sd[0]),
                counts={s: len(samples[s]) for s in samples})
    return data, meta


# ════════════════════════════════════════════════════════════════════════
# 2.  MODELS
# ════════════════════════════════════════════════════════════════════════

def nb_nll(y, mu, r, eps=1e-8):
    y  = y.float()
    mu = torch.clamp(mu, min=eps)
    r  = torch.clamp(r,  min=eps)
    log_prob = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
                + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu)))
    return -log_prob.mean()


def nb_stats(mu, r):
    var = mu + mu ** 2 / torch.clamp(r, min=1e-8)
    return mu, torch.sqrt(var)


class MovingAverage(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k = k; self.pad = k // 2

    def forward(self, x):
        return F.avg_pool1d(x, self.k, 1, self.pad)


class DLinearBlock(nn.Module):
    """Per-channel trend+seasonal linear projection L→H. x:(B,L,C)→(B,H,C)."""
    def __init__(self, lookback, horizon, kernel=7):
        super().__init__()
        self.ma    = MovingAverage(kernel)
        self.trend = nn.Linear(lookback, horizon)
        self.seas  = nn.Linear(lookback, horizon)

    def forward(self, x):
        x  = x.transpose(1, 2)
        tr = self.ma(x); se = x - tr
        return (self.trend(tr) + self.seas(se)).transpose(1, 2)


class KOTEExoDLinearNB(nn.Module):
    def __init__(self, meta, hidden=128, kernel=7, dropout=0.1,
                 use_calendar=True, use_volume=True, use_emotion=True,
                 use_topic=True, use_interaction=False,
                 dynamic_gate=False, emotion_dropout=0.1, head="nb"):
        super().__init__()
        L, H = meta["lookback"], meta["horizon"]
        E, K, Fi, Fc = meta["n_emotion"], meta["n_topic"], meta["n_inter"], meta["n_calendar"]
        self.H, self.E, self.K = H, E, K
        self.use_calendar = use_calendar; self.use_volume = use_volume
        self.use_emotion  = use_emotion;  self.use_topic  = use_topic
        self.use_interaction = use_interaction
        self.dynamic_gate    = dynamic_gate
        self.emotion_dropout = emotion_dropout
        self.head = head

        self.target_block = DLinearBlock(L, H, kernel)
        fdim = H
        if use_volume:
            self.volume_block = DLinearBlock(L, H, kernel); fdim += H
        if use_emotion:
            self.emotion_block = DLinearBlock(L, H, kernel)
            self.emotion_gate  = nn.Parameter(torch.zeros(E))
            if dynamic_gate:
                self.gate_mlp = nn.Sequential(nn.Linear(E, E), nn.ReLU(), nn.Linear(E, E))
            fdim += H * E
        if use_topic:
            self.topic_block = DLinearBlock(L, H, kernel)
            self.topic_gate  = nn.Parameter(torch.zeros(K))
            fdim += H * K
        if use_interaction:
            self.inter_block = DLinearBlock(L, H, kernel); fdim += H * Fi
        if use_calendar:
            self.cal_mlp = nn.Sequential(nn.Linear(Fc, 32), nn.ReLU()); fdim += H * 32

        self.fusion = nn.Sequential(
            nn.Linear(fdim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        if head == "quantile":
            self.q_levels = [0.1, 0.5, 0.9]
            self.out_head  = nn.Linear(hidden, H * len(self.q_levels))
        else:
            self.mu_head = nn.Linear(hidden, H)
            self.r_head  = nn.Linear(hidden, H)

    def _emo_dropout(self, e):
        if self.training and self.emotion_dropout > 0:
            mask = (torch.rand(e.shape[-1], device=e.device) > self.emotion_dropout).float()
            e = e * mask
        return e

    def forward(self, batch):
        B     = batch["y_past"].shape[0]
        parts = [self.target_block(batch["y_past"]).reshape(B, -1)]
        if self.use_volume:
            parts.append(self.volume_block(batch["v_past"]).reshape(B, -1))
        if self.use_emotion:
            e  = self._emo_dropout(batch["e_past"])
            eo = self.emotion_block(e)
            g  = torch.sigmoid(self.emotion_gate)
            if self.dynamic_gate:
                g  = g * torch.sigmoid(self.gate_mlp(e.mean(1)))
                eo = eo * g.unsqueeze(1)
            else:
                eo = eo * g.view(1, 1, -1)
            self._last_gate = g
            parts.append(eo.reshape(B, -1))
        if self.use_topic:
            to = self.topic_block(batch["t_past"]) * torch.sigmoid(self.topic_gate).view(1, 1, -1)
            parts.append(to.reshape(B, -1))
        if self.use_interaction:
            parts.append(self.inter_block(batch["inter"]).reshape(B, -1))
        if self.use_calendar:
            parts.append(self.cal_mlp(batch["c_future"]).reshape(B, -1))
        h    = self.fusion(torch.cat(parts, dim=-1))
        base = batch["y_base"]
        if self.head == "quantile":
            d = self.out_head(h).view(B, self.H, len(self.q_levels)).clamp(-1.0, 1.0)
            return {"quantiles": base.unsqueeze(-1) * torch.exp(d)}
        mu = base * torch.exp(self.mu_head(h).clamp(-1.0, 1.0))
        r  = F.softplus(self.r_head(h)) + 1e-6
        return {"mu": mu, "r": r}


class KOTEMLPNB(nn.Module):
    def __init__(self, meta, hidden=128, dropout=0.1, use_calendar=True, use_volume=True,
                 use_emotion=True, use_topic=True, use_interaction=False, head="nb", **kw):
        super().__init__()
        L, H = meta["lookback"], meta["horizon"]
        E, K, Fi, Fc = meta["n_emotion"], meta["n_topic"], meta["n_inter"], meta["n_calendar"]
        self.H     = H
        self.flags = dict(use_calendar=use_calendar, use_volume=use_volume,
                          use_emotion=use_emotion, use_topic=use_topic,
                          use_interaction=use_interaction)
        dim = L
        if use_volume:      dim += L
        if use_emotion:     dim += L * E
        if use_topic:       dim += L * K
        if use_interaction: dim += L * Fi
        if use_calendar:    dim += H * Fc
        self.net     = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(hidden, H)
        self.r_head  = nn.Linear(hidden, H)
        self.head    = "nb"

    def forward(self, batch):
        B     = batch["y_past"].shape[0]
        parts = [batch["y_past"].reshape(B, -1)]
        f = self.flags
        if f["use_volume"]:      parts.append(batch["v_past"].reshape(B, -1))
        if f["use_emotion"]:     parts.append(batch["e_past"].reshape(B, -1))
        if f["use_topic"]:       parts.append(batch["t_past"].reshape(B, -1))
        if f["use_interaction"]: parts.append(batch["inter"].reshape(B, -1))
        if f["use_calendar"]:    parts.append(batch["c_future"].reshape(B, -1))
        h  = self.net(torch.cat(parts, -1))
        mu = batch["y_base"] * torch.exp(self.mu_head(h).clamp(-1.0, 1.0))
        r  = F.softplus(self.r_head(h)) + 1e-6
        return {"mu": mu, "r": r}


class TiDENB(nn.Module):
    def __init__(self, meta, hidden=128, dropout=0.1, use_calendar=True, use_volume=True,
                 use_emotion=True, use_topic=True, use_interaction=False, head="nb", **kw):
        super().__init__()
        L, H = meta["lookback"], meta["horizon"]
        E, K, Fi, Fc = meta["n_emotion"], meta["n_topic"], meta["n_inter"], meta["n_calendar"]
        self.H     = H
        self.flags = dict(use_calendar=use_calendar, use_volume=use_volume,
                          use_emotion=use_emotion, use_topic=use_topic,
                          use_interaction=use_interaction)
        dim = L
        if use_volume:      dim += L
        if use_emotion:     dim += L * E
        if use_topic:       dim += L * K
        if use_interaction: dim += L * Fi
        if use_calendar:    dim += H * Fc
        self.proj = nn.Linear(dim, hidden)
        self.enc1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.enc2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.dec  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(hidden, H)
        self.r_head  = nn.Linear(hidden, H)
        self.head    = "nb"

    def forward(self, batch):
        B     = batch["y_past"].shape[0]
        parts = [batch["y_past"].reshape(B, -1)]
        f = self.flags
        if f["use_volume"]:      parts.append(batch["v_past"].reshape(B, -1))
        if f["use_emotion"]:     parts.append(batch["e_past"].reshape(B, -1))
        if f["use_topic"]:       parts.append(batch["t_past"].reshape(B, -1))
        if f["use_interaction"]: parts.append(batch["inter"].reshape(B, -1))
        if f["use_calendar"]:    parts.append(batch["c_future"].reshape(B, -1))
        h = self.proj(torch.cat(parts, -1))
        h = h + self.enc1(h); h = h + self.enc2(h); h = self.dec(h)
        mu = batch["y_base"] * torch.exp(self.mu_head(h).clamp(-1.0, 1.0))
        r  = F.softplus(self.r_head(h)) + 1e-6
        return {"mu": mu, "r": r}


MODELS = {"exodlinear": KOTEExoDLinearNB, "mlp": KOTEMLPNB, "tide": TiDENB}


# ════════════════════════════════════════════════════════════════════════
# 3.  METRICS
# ════════════════════════════════════════════════════════════════════════

def point_metrics(y, yhat, mase_denom):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    err  = yhat - y
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    wape = np.sum(np.abs(err)) / (np.sum(np.abs(y)) + 1e-8)
    smape = np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(yhat) + 1e-8))
    mase = mae / (mase_denom + 1e-8)
    return dict(MAE=mae, RMSE=rmse, MASE=mase, WAPE=wape, sMAPE=smape)


def nb_prob_metrics(y, mu, r, n_samples=2000, seed=0):
    y    = np.asarray(y, float)
    mu_t = torch.tensor(np.clip(mu, 1e-6, None), dtype=torch.float64)
    r_t  = torch.tensor(np.clip(r,  1e-6, None), dtype=torch.float64)
    dist = NegativeBinomial(total_count=r_t, logits=torch.log(mu_t / r_t))
    assert torch.allclose(dist.mean, mu_t, atol=1e-3), "NB mean != mu"
    yt   = torch.tensor(y, dtype=torch.float64)
    nll  = -dist.log_prob(yt).mean().item()
    torch.manual_seed(seed)
    samp = dist.sample((n_samples,)).numpy()
    crps = ps.crps_ensemble(y, samp.T).mean()
    lo80, hi80 = np.percentile(samp, [10, 90], axis=0)
    lo90, hi90 = np.percentile(samp, [5, 95],  axis=0)
    return dict(NLL=nll, CRPS=float(crps),
                Cov80=float(np.mean((y >= lo80) & (y <= hi80))),
                Cov90=float(np.mean((y >= lo90) & (y <= hi90))),
                Width80=float(np.mean(hi80 - lo80)),
                Width90=float(np.mean(hi90 - lo90)))


def nb_mean_std(mu, r):
    var = mu + mu ** 2 / np.clip(r, 1e-8, None)
    return mu, np.sqrt(var)


# ════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ════════════════════════════════════════════════════════════════════════

def get_data(lookback, horizon, emotion_mode="group"):
    key = (lookback, horizon, emotion_mode)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = make_datasets(lookback, horizon, emotion_mode)
    return _DATA_CACHE[key]


def to_dev(ds):
    return {k: torch.tensor(v).to(DEVICE) for k, v in ds.items() if k != "tdate"}


def mase_denominator(horizon):
    t = pd.read_parquet(RAW / "target_call_counts.parquet")
    t["date"] = pd.to_datetime(t["date"])
    a, b = load_splits()["train"]
    y = t[(t.date >= a) & (t.date <= b)].sort_values("date").y.values.astype(float)
    return np.mean(np.abs(y[7:] - y[:-7]))


def _gauss_nll(y, mu, sigma):
    sigma = torch.clamp(sigma, min=1e-3)
    return (0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma) + 0.918938).mean()


def _pinball(y, q, levels):
    loss = 0.0
    for i, a in enumerate(levels):
        e    = y - q[..., i]
        loss = loss + torch.maximum(a * e, (a - 1) * e).mean()
    return loss / len(levels)


def run(cfg, seed=42, epochs=200, patience=20, lr=1e-3, wd=1e-4, batch=32, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    data, meta = get_data(cfg.get("lookback", 56), cfg.get("horizon", 1),
                          cfg.get("emotion_mode", "group"))
    H    = meta["horizon"]; head = cfg.get("head", "nb")
    tens = {s: to_dev(data[s]) for s in ["train", "valid", "test"]}

    ModelCls = MODELS[cfg["model"]]
    model = ModelCls(
        meta, hidden=cfg.get("hidden", 128), dropout=cfg.get("dropout", 0.1),
        use_calendar=cfg.get("use_calendar", True), use_volume=cfg.get("use_volume", True),
        use_emotion=cfg.get("use_emotion", True),   use_topic=cfg.get("use_topic", True),
        use_interaction=cfg.get("use_interaction", False),
        dynamic_gate=cfg.get("dynamic_gate", False),
        emotion_dropout=cfg.get("emotion_dropout", 0.1),
        head=head,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    tr = tens["train"]; n = tr["y_past"].shape[0]
    best_val = float("inf"); best_state = None; bad = 0
    point_loss = cfg.get("point_loss", "huber")

    def loss_fn(out, y):
        if head == "nb":       return nb_nll(y, out["mu"], out["r"])
        if head == "gaussian": return _gauss_nll(y, out["mu"], out["r"])
        if head == "quantile": return _pinball(y, out["quantiles"], model.q_levels)
        # point
        if point_loss == "mse": return F.mse_loss(out["mu"], y)
        if point_loss == "mae": return F.l1_loss(out["mu"], y)
        return F.huber_loss(out["mu"], y, delta=20.0)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            mb  = {k: tr[k][idx] for k in tr}
            out = model(mb)
            loss = loss_fn(out, mb["y_future"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(tens["valid"]), tens["valid"]["y_future"]).item()
        if vloss < best_val - 1e-5:
            best_val = vloss; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)

    mden = mase_denominator(H)
    res  = {"_best_val_loss": best_val, "_epochs": ep + 1}
    preds = {}
    model.eval()

    if head == "point":
        with torch.no_grad():
            vmu = model(tens["valid"])["mu"].cpu().numpy().reshape(-1)
        vres = np.abs(data["valid"]["y_future"].reshape(-1) - vmu)
        nq   = len(vres); svr = np.sort(vres)
        def conf_q(cov):
            k = min(nq, int(np.ceil((nq + 1) * cov)))
            return float(svr[k - 1])
        q80, q90 = conf_q(0.8), conf_q(0.9)
        res["_q80"] = q80; res["_q90"] = q90

    for split in ["valid", "test"]:
        with torch.no_grad():
            out = model(tens[split])
        y = data[split]["y_future"].reshape(-1)
        if head == "quantile":
            q    = out["quantiles"].cpu().numpy().reshape(-1, len(model.q_levels))
            yhat = q[:, 1]
            pm   = point_metrics(y, yhat, mden)
            lo, hi = q[:, 0], q[:, 2]
            prob = {"NLL": np.nan, "CRPS": np.nan,
                    "Cov80": float(np.mean((y >= lo) & (y <= hi))),
                    "Cov90": np.nan,
                    "Width80": float(np.mean(hi - lo)), "Width90": np.nan}
            std  = (hi - lo) / 2.56
        elif head == "gaussian":
            mu    = out["mu"].cpu().numpy().reshape(-1)
            sigma = out["r"].cpu().numpy().reshape(-1)
            yhat  = mu; pm = point_metrics(y, yhat, mden)
            nll   = -norm.logpdf(y, mu, np.clip(sigma, 1e-3, None)).mean()
            lo80, hi80 = norm.ppf(0.1, mu, sigma), norm.ppf(0.9, mu, sigma)
            lo90, hi90 = norm.ppf(0.05, mu, sigma), norm.ppf(0.95, mu, sigma)
            prob  = {"NLL": float(nll), "CRPS": np.nan,
                     "Cov80": float(np.mean((y >= lo80) & (y <= hi80))),
                     "Cov90": float(np.mean((y >= lo90) & (y <= hi90))),
                     "Width80": float(np.mean(hi80 - lo80)),
                     "Width90": float(np.mean(hi90 - lo90))}
            std   = sigma
        elif head == "point":
            mu   = out["mu"].cpu().numpy().reshape(-1)
            yhat = mu; pm = point_metrics(y, yhat, mden)
            prob = {"NLL": np.nan, "CRPS": np.nan,
                    "Cov80": float(np.mean(np.abs(y - mu) <= q80)),
                    "Cov90": float(np.mean(np.abs(y - mu) <= q90)),
                    "Width80": float(2 * q80), "Width90": float(2 * q90)}
            std  = np.full_like(mu, q80 / 1.2816)
        else:  # nb
            mu   = out["mu"].cpu().numpy().reshape(-1)
            r    = out["r"].cpu().numpy().reshape(-1)
            yhat = mu; pm = point_metrics(y, yhat, mden)
            prob = nb_prob_metrics(y, mu, r)
            std  = np.sqrt(mu + mu ** 2 / np.clip(r, 1e-8, None))
        res[split]   = {**pm, **prob}
        preds[split] = pd.DataFrame({
            "tdate":     np.repeat(pd.to_datetime(data[split]["tdate"]), H),
            "y":         y,
            "pred_mean": yhat,
            "pred_std":  std,
        })
    return res, preds, model, meta, data


# ════════════════════════════════════════════════════════════════════════
# 5.  SUITE RUNNER
# ════════════════════════════════════════════════════════════════════════

def _cfg(*dicts):
    out = dict(COMMON)
    for d in dicts:
        out.update(d)
    return out


def experiments(mode="nb"):
    if mode == "point":
        h = dict(head="point")
        e = []
        for s in ["A", "B", "C", "D", "E", "F"]:
            e.append((f"ExoDLinear-{s}", ABL_DESC[s], _cfg(dict(model="exodlinear", lookback=56), ABL[s], h)))
        e.append(("MLP-point-C",  "+volume",  _cfg(dict(model="mlp",  lookback=56), ABL["C"], h)))
        e.append(("MLP-point-D",  "+emotion", _cfg(dict(model="mlp",  lookback=56), ABL["D"], h)))
        e.append(("TiDE-point-D", "+emotion", _cfg(dict(model="tide", lookback=56), ABL["D"], h)))
        for L in [28, 84, 112]:
            e.append((f"ExoDLinear-E-L{L}", f"L={L}", _cfg(dict(model="exodlinear", lookback=L), ABL["E"], h)))
        e.append(("ExoDLinear-D-NB(ref)",   "NB head",  _cfg(dict(model="exodlinear", lookback=56, head="nb"),       ABL["D"])))
        e.append(("ExoDLinear-D-Gauss(ref)", "Gaussian", _cfg(dict(model="exodlinear", lookback=56, head="gaussian"), ABL["D"])))
        return e

    e = []
    for s in ["A", "B", "C", "D", "E", "F"]:
        e.append((f"ExoDLinear-{s}", ABL_DESC[s], _cfg(dict(model="exodlinear", lookback=56), ABL[s])))
    e.append(("MLP-NB-C",  "+volume",  _cfg(dict(model="mlp",  lookback=56), ABL["C"])))
    e.append(("MLP-NB-D",  "+emotion", _cfg(dict(model="mlp",  lookback=56), ABL["D"])))
    e.append(("TiDE-NB-D", "+emotion", _cfg(dict(model="tide", lookback=56), ABL["D"])))
    for L in [28, 84, 112]:
        e.append((f"ExoDLinear-E-L{L}", f"L={L}", _cfg(dict(model="exodlinear", lookback=L), ABL["E"])))
    e.append(("ExoDLinear-D-dyngate",  "dyn-gate",  _cfg(dict(model="exodlinear", lookback=56, dynamic_gate=True), ABL["D"])))
    e.append(("ExoDLinear-D-gauss",    "gaussian",  _cfg(dict(model="exodlinear", lookback=56, head="gaussian"),   ABL["D"])))
    e.append(("ExoDLinear-D-quantile", "quantile",  _cfg(dict(model="exodlinear", lookback=56, head="quantile"),   ABL["D"])))
    return e


def run_one(name, desc, c, save_artifacts=True):
    per = {sp: {m: [] for m in METRICS} for sp in ["valid", "test"]}
    ep  = []
    for sd in SEEDS:
        res, preds, model, meta, data = run(c, seed=sd, lr=LR, wd=WD)
        ep.append(res["_epochs"])
        for sp in ["valid", "test"]:
            for m in METRICS:
                per[sp][m].append(res[sp].get(m, np.nan))
        if save_artifacts and sd == SEEDS[0]:
            preds["test"].to_parquet(OUT / "predictions" / f"{name}_test.parquet")
            if name in ("ExoDLinear-D",):
                tag = "point" if c.get("head") == "point" else "D"
                torch.save({"state_dict": model.state_dict(), "cfg": c, "meta": meta},
                           CK / f"exodlinear_{tag}_best.pt")
    row = dict(experiment=name, setting=desc, model=c["model"],
               lookback=c.get("lookback", 56), head=c.get("head", "nb"),
               epochs=int(np.mean(ep)))
    for sp in ["valid", "test"]:
        for m in METRICS:
            row[f"{sp}_{m}"]     = round(float(np.nanmean(per[sp][m])), 4)
            row[f"{sp}_{m}_std"] = round(float(np.nanstd(per[sp][m])),  4)
    return row


def write_tables(summ_rows, mode="nb"):
    pfx  = "" if mode == "nb" else f"{mode}_"
    summ = pd.DataFrame(summ_rows)
    order = [e[0] for e in experiments(mode)]
    summ["__o"] = summ.experiment.map({n: i for i, n in enumerate(order)})
    summ = summ.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    summ.to_csv(OUT / "metrics" / f"results_{pfx}summary.csv", index=False)

    def cell(r, m):
        return f"{r[f'test_{m}']:.3f}±{r[f'test_{m}_std']:.3f}"
    lines = ["| Experiment | Setting | MAE | RMSE | MASE | NLL | CRPS | 80%Cov | 90%Cov |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r['experiment']} | {r['setting']} | {r['test_MAE']:.2f} | "
                     f"{r['test_RMSE']:.2f} | {cell(r,'MASE')} | {r['test_NLL']:.3f} | "
                     f"{r['test_CRPS']:.2f} | {r['test_Cov80']:.3f} | {r['test_Cov90']:.3f} |")
    (OUT / "metrics" / f"results_{pfx}test.md").write_text("\n".join(lines))
    return summ


# ════════════════════════════════════════════════════════════════════════
# 6.  MONTHLY SUICIDE  (utility functions)
# ════════════════════════════════════════════════════════════════════════

def load_socio():
    p  = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=BASE_CSV)
    df = pd.read_csv(p)
    df["month"] = pd.PeriodIndex(df["date"], freq="M")
    df = df.rename(columns={"자살자수": "y"})
    socio = [c for c in df.columns if c not in ("date", "month", "y")]
    return df[["month", "y"] + socio], socio


def monthly_emotion():
    if not (EMO / "comment_kote_probs.parquet").exists():
        raise FileNotFoundError(
            "\n\033[31m[ERROR] 전처리 파일이 없습니다. 먼저 preprocess를 실행하세요:\033[0m\n"
            "\033[33m  ./suicide_run.sh preprocess\033[0m"
        )
    df = pd.read_parquet(EMO / "comment_kote_probs.parquet",
                         columns=["date"] + [f"emotion_{i}" for i in range(44)])
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    labels  = json.load(open(EMO / "emotion_labels.json"))
    mapping = EMO_GROUPS
    groups  = ["기쁨", "슬픔", "분노", "중립"]
    msum    = df.groupby("month")[[f"emotion_{i}" for i in range(44)]].sum()
    feats   = {}
    for g in groups:
        members   = [labels.index(e) for e in mapping[g] if e in labels]
        feats[f"E_{g}"] = np.log1p(msum[[f"emotion_{i}" for i in members]].sum(axis=1))
    return pd.DataFrame(feats).reset_index()


def monthly_topic():
    df = pd.read_parquet(RAW / "youtube_news_comments.parquet", columns=["date", "title"])
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    df["title"] = df["title"].fillna("").astype(str)
    uniq = df.drop_duplicates("title")["title"]
    uniq = uniq[uniq.str.len() > 0]
    vec  = TfidfVectorizer(max_features=20000, min_df=3, token_pattern=r"(?u)\b\w\w+\b")
    X    = vec.fit_transform(uniq)
    km   = KMeans(n_clusters=K_TOPIC, n_init=5, random_state=42).fit(X)
    t2c  = dict(zip(uniq, km.predict(X)))
    df["topic"] = df["title"].map(t2c)
    df = df.dropna(subset=["topic"]); df["topic"] = df["topic"].astype(int)
    ct  = (df.groupby(["month", "topic"]).size()
             .unstack(fill_value=0)
             .reindex(columns=range(K_TOPIC), fill_value=0))
    ct.columns = [f"T_{k}" for k in range(K_TOPIC)]
    return np.log1p(ct).reset_index()


def build_monthly():
    socio, S = load_socio()
    E = monthly_emotion(); T = monthly_topic()
    df = (socio.merge(E, on="month", how="inner")
               .merge(T, on="month", how="inner")
               .sort_values("month").reset_index(drop=True))
    Ecols = [c for c in df.columns if c.startswith("E_")]
    Tcols = [c for c in df.columns if c.startswith("T_")]
    return df, S, Ecols, Tcols


def build_monthly_infer(target_month):
    """
    학습 데이터(y 있는 과거) + target_month 피처 벡터 반환.
    S/E/T 가용 여부를 각각 체크해 사용 가능한 조합만 반환.
    """
    target_period = pd.Period(target_month, freq="M")

    # train: y가 있는 과거 데이터
    train_df, S, Ecols, Tcols = build_monthly()
    train_df = train_df[train_df["y"].notna()].copy()

    # S — socio에서 target_month 행 직접 조회 (y 없어도 됨)
    socio_raw, _ = load_socio()
    s_row = socio_raw[socio_raw["month"] == target_period]
    has_S = len(s_row) > 0 and not s_row[S].isnull().any().any()

    # E — 로컬 comments에서 계산
    E_df = monthly_emotion()
    e_row = E_df[E_df["month"] == target_period]
    _Ecols = [c for c in E_df.columns if c != "month"]
    has_E = len(e_row) > 0

    # T — 로컬 comments에서 계산
    T_df = monthly_topic()
    t_row = T_df[T_df["month"] == target_period]
    _Tcols = [c for c in T_df.columns if c != "month"]
    has_T = len(t_row) > 0

    # target feature 벡터 조립
    target_feats = {}
    if has_S:
        for c in S:
            target_feats[c] = float(s_row[c].values[0])
    if has_E:
        for c in _Ecols:
            target_feats[c] = float(e_row[c].values[0])
    if has_T:
        for c in _Tcols:
            target_feats[c] = float(t_row[c].values[0])

    return train_df, target_feats, S, Ecols, Tcols, has_S, has_E, has_T


def monthly_infer_predict(train_df, target_feats, cols):
    """train_df[cols] 전체 학습 → target_feats 단일 예측."""
    X_train = train_df[cols].values
    y_train = train_df["y"].values
    X_pred  = np.array([[target_feats[c] for c in cols]])
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    pipe.fit(X_train, y_train)
    return float(pipe.predict(X_pred)[0])


def expanding_eval(df, cols, start="2022-01", min_train=18):
    X = df[cols].values; y = df["y"].values
    m = df["month"].astype(str).values
    ys, ps, ms = [], [], []
    for i in range(len(df)):
        if i < min_train or m[i] < start:
            continue
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        pipe.fit(X[:i], y[:i])
        ps.append(pipe.predict(X[i:i + 1])[0]); ys.append(y[i]); ms.append(m[i])
    ys, ps = np.array(ys), np.array(ps)
    r2 = 1 - ((ys - ps) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    metrics = dict(R2=r2, MAE=np.abs(ys - ps).mean(),
                   RMSE=np.sqrt(((ys - ps) ** 2).mean()), n=len(ys))
    preds = pd.DataFrame({"month": ms, "y_true": ys.round(1), "y_pred": ps.round(1)})
    return metrics, preds


def baseline_eval(df, kind, start="2022-01", min_train=18):
    y = df["y"].values; mo = df["month"].dt.month.values
    m = df["month"].astype(str).values
    ys, ps = [], []
    for i in range(len(df)):
        if i < min_train or m[i] < start:
            continue
        if kind == "seasonal_mean":
            same = y[:i][mo[:i] == mo[i]]
            ps.append(same.mean() if len(same) else y[:i].mean())
        elif kind == "lag12":
            ps.append(y[i - 12] if i >= 12 else y[:i].mean())
        ys.append(y[i])
    ys, ps = np.array(ys), np.array(ps)
    return dict(R2=1 - ((ys - ps) ** 2).sum() / ((ys - ys.mean()) ** 2).sum(),
                MAE=np.abs(ys - ps).mean(),
                RMSE=np.sqrt(((ys - ps) ** 2).mean()), n=len(ys))
