"""Full execution pipelines.

Usage
-----
# 일별 예측기: Ablation suite (Setting A~F, 4 seeds)
python suicide_pipeline.py --run ablation [--mode nb|point]

# 일별 예측기: 메인 모델 확정 및 저장
python suicide_pipeline.py --run main

# 월별 예측기: 자살자수 × 7 피처 조합 RidgeCV
python suicide_pipeline.py --run monthly

# 병렬 실행 지원 (idx/aggregate/list)
python suicide_pipeline.py --run parallel --idx N
python suicide_pipeline.py --run parallel --aggregate [--mode nb|point]
python suicide_pipeline.py --run parallel --list
"""
# ── 경로 설정 (config/ + utils/ 폴더를 sys.path에 추가) ─────────────────────
import sys as _sys
from pathlib import Path as _Path
_BASE = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_BASE / "config"))
_sys.path.insert(0, str(_BASE / "src"))
# ────────────────────────────────────────────────────────────────────────────
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import torch

from suicide_config import (
    CK, DELIVERABLE, METRICS, OUT, SEEDS, FONT_PATH,
    MAIN_CFG,
)
from suicide_utils import (
    # preprocessing
    build_target, build_comments, kote_inference, daily_emotion, topic_features,
    # suite
    experiments, run_one, write_tables,
    # training
    run,
    # monthly
    build_monthly, expanding_eval, baseline_eval,
)

# ── 색상 출력 ────────────────────────────────────────────────────────────────
def _c(code, s): return f"\033[{code}m{s}\033[0m"
def _ts():   return __import__("datetime").datetime.now().strftime("%H:%M:%S")
def log(s):  print(_c("36", f"[{_ts()}] {s}"))
def ok(s):   print(_c("32", f"[{_ts()}] ✓ {s}"))
def die(s):  print(_c("31", f"[ERROR]  {s}"), file=sys.stderr); sys.exit(1)
# ────────────────────────────────────────────────────────────────────────────

DELIVERABLE.mkdir(exist_ok=True)

# Korean font setup for matplotlib
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"]       = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════════════════════
# Pipeline 0 — 전처리 (01~04번 스크립트)
# ════════════════════════════════════════════════════════════════════════

def run_preprocess(gpu=None):
    log("=== [전처리 1/4] target 데이터 준비 ===")
    build_target()

    log("=== [전처리 2/4] HuggingFace 댓글 다운로드 ===")
    build_comments()

    log("=== [전처리 3/4] KOTE 감정 추론 (GPU 필요) ===")
    kote_inference(device=f"cuda:{gpu}" if gpu is not None else None)

    log("=== [전처리 4/4] 일별 감정 집계 + 토픽 클러스터링 ===")
    daily_emotion()
    topic_features()

    ok("=== 전처리 완료 ===")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 1 — 일별 예측기: Ablation Suite (05_run_suite.py)
# ════════════════════════════════════════════════════════════════════════

def run_ablation_suite(mode="nb"):
    raw_rows, summ_rows = [], []
    for name, desc, c in experiments(mode):
        per = {sp: {m: [] for m in METRICS} for sp in ["valid", "test"]}
        ep  = []
        for sd in SEEDS:
            res, preds, model, meta, data = run(c, seed=sd, lr=5e-4, wd=5e-4)
            ep.append(res["_epochs"])
            for sp in ["valid", "test"]:
                for m in METRICS:
                    per[sp][m].append(res[sp].get(m, np.nan))
                raw_rows.append(dict(experiment=name, split=sp, seed=sd,
                                     **{m: res[sp].get(m, np.nan) for m in METRICS}))
            if sd == SEEDS[0]:
                preds["test"].to_parquet(OUT / "predictions" / f"{name}_test.parquet")
                if name == "ExoDLinear-D":
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
        summ_rows.append(row)
        print(f"[{name:22s}] test MASE={row['test_MASE']:.3f}±{row['test_MASE_std']:.3f} "
              f"MAE={row['test_MAE']:.2f} NLL={row['test_NLL']:.2f} "
              f"Cov80={row['test_Cov80']:.2f} (ep~{row['epochs']})")

    pfx = "" if mode == "nb" else f"{mode}_"
    pd.DataFrame(raw_rows).to_csv(OUT / "metrics" / f"results_{pfx}per_seed.csv", index=False)
    summ = write_tables(summ_rows, mode)
    print("\n===== TEST (mean over seeds) =====")
    print(summ[["experiment", "setting", "test_MAE", "test_MASE",
                "test_NLL", "test_Cov80"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════
# Pipeline 2 — 일별 예측기: 메인 모델 확정 (11_main_model.py)
# ════════════════════════════════════════════════════════════════════════

def lock_main_model():
    agg = {sp: {m: [] for m in ["MAE", "RMSE", "MASE", "WAPE", "sMAPE",
                                  "Cov80", "Cov90", "Width80", "Width90"]}
           for sp in ["valid", "test"]}
    preds0 = model0 = meta0 = None

    for sd in SEEDS:
        res, preds, model, meta, data = run(MAIN_CFG, seed=sd, lr=5e-4, wd=5e-4)
        for sp in ["valid", "test"]:
            for m in agg[sp]:
                agg[sp][m].append(res[sp][m])
        if sd == 42:
            preds0, model0, meta0 = preds, model, meta

    summary = {sp: {m: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
                    for m, v in agg[sp].items()} for sp in ["valid", "test"]}

    torch.save({"state_dict": model0.state_dict(), "cfg": MAIN_CFG, "meta": meta0},
               CK / "main_model.pt")
    preds0["test"].to_parquet(OUT / "predictions" / "MAIN_test.parquet")

    # figure: valid(2022) + test(2023)
    fig, ax = plt.subplots(figsize=(14, 4.6))
    for sp, col in [("valid", "#dd8452"), ("test", "#55a868")]:
        p = preds0[sp].copy()
        p["tdate"] = pd.to_datetime(p["tdate"]); p = p.sort_values("tdate")
        q80 = p.pred_std * 1.2816
        ax.fill_between(p.tdate, p.pred_mean - q80, p.pred_mean + q80, alpha=0.22, color=col)
        ax.plot(p.tdate, p.pred_mean, color="#c44e52", lw=1.0)
        ax.plot(p.tdate, p.y,         color="#333",    lw=0.9)
    ax.legend(handles=[
        Line2D([], [], color="#333",    label="실제"),
        Line2D([], [], color="#c44e52", label="예측(평균)"),
        Patch(fc="#dd8452", alpha=0.3,  label="검증 80% 구간"),
        Patch(fc="#55a868", alpha=0.3,  label="테스트 80% 구간"),
    ], loc="upper left", fontsize=9, ncol=2)
    t = summary["test"]
    ax.set_title(f"메인 모델 — 다음날 상담건수 예측 (ExoDLinear point+conformal) | "
                 f"TEST MAE {t['MAE'][0]:.1f}, MASE {t['MASE'][0]:.3f}, 80%커버 {t['Cov80'][0]:.2f}")
    ax.set_ylabel("일별 상담건수"); ax.margins(x=0.01); fig.tight_layout()
    fig.savefig(DELIVERABLE / "00_MAIN_prediction.png", dpi=140); plt.close(fig)

    def line(sp):
        s = summary[sp]
        return (f"| {sp} | {s['MAE'][0]:.2f} | {s['RMSE'][0]:.2f} | "
                f"{s['MASE'][0]:.3f}±{s['MASE'][1]:.3f} | {s['WAPE'][0]:.3f} | "
                f"{s['Cov80'][0]:.3f} | {s['Cov90'][0]:.3f} | {s['Width80'][0]:.0f} |")

    card = f"""# 메인 모델 카드 — 자살예방상담전화 다음날 상담건수 예측

## 과제
입력 = 과거 56일 일별 상담건수 → 출력 = **다음날 raw 상담건수 + 80/90% 예측구간**.

## 모델: KOTE-ExoDLinear (point + conformal)
- per-channel DLinear(추세+계절 분해)로 과거 56일 상담건수 인코딩 → Fusion MLP(hidden 64, dropout 0.5)
- **anchored mean**: μ = (최근 7일 평균) · exp(δ)
- 손실: Huber (point). 구간: split-conformal (검증 잔차 분위수)
- 검증 기준 calendar/volume/감정 추가 이득 없음 → Setting A 선택

## 데이터
- Train 2020-02-26~2021-12-31 (675) / Valid 2022 (365) / Test 2023-01~10 (296)

## 성능 (4-seed 평균±std)
| split | MAE | RMSE | MASE | WAPE | 80%Cov | 90%Cov | 80%폭 |
|---|---|---|---|---|---|---|---|
{line('valid')}
{line('test')}
"""
    (DELIVERABLE / "MAIN_MODEL.md").write_text(card)
    print(card)
    print("saved deliverable/00_MAIN_prediction.png, MAIN_MODEL.md, outputs/checkpoints/main_model.pt")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 3 — 월별 예측기: 자살자수 × 7 피처 조합 (13_monthly_suicide.py)
# ════════════════════════════════════════════════════════════════════════

def run_monthly_suicide():
    df, S, E, T = build_monthly()
    print(f"[data] {len(df)} months {df.month.min()}..{df.month.max()} | "
          f"S={len(S)} E={len(E)} T={len(T)} feats | "
          f"target mean {df.y.mean():.0f} std {df.y.std():.0f}")

    combos = {
        "1.사회경제(S)": S,       "2.감정(E)": E,         "3.토픽(T)": T,
        "4.S+E":        S + E,   "5.S+T":    S + T,      "6.E+T":    E + T,
        "7.S+E+T":      S + E + T,
    }
    rows = []
    for name, cols in combos.items():
        r = expanding_eval(df, cols)
        rows.append(dict(model=name, n_feat=len(cols), **{k: round(v, 3) for k, v in r.items()}))
    for bn, bk in [("기준:계절평균", "seasonal_mean"), ("기준:lag-12", "lag12")]:
        r = baseline_eval(df, bk)
        rows.append(dict(model=bn, n_feat=0, **{k: round(v, 3) for k, v in r.items()}))

    res = pd.DataFrame(rows)
    res.to_csv(DELIVERABLE / "monthly_suicide_7models.csv", index=False)

    print(f"\n=== 월별 자살자수 예측 — 확장윈도우 1-step "
          f"(test {rows[0]['n']}개월: 2022-01~2023-10) ===")
    print(res.to_string(index=False))

    s = res.set_index("model")
    print("\n=== 댓글 데이터가 유의미한가? (사회경제 S 대비 증분) ===")
    for k in ["1.사회경제(S)", "4.S+E", "5.S+T", "7.S+E+T", "6.E+T"]:
        print(f"  {k:14s} R²={s.loc[k,'R2']:+.3f}  MAE={s.loc[k,'MAE']:.1f}")
    dE  = s.loc["4.S+E",    "MAE"] - s.loc["1.사회경제(S)", "MAE"]
    dT  = s.loc["5.S+T",    "MAE"] - s.loc["1.사회경제(S)", "MAE"]
    dET = s.loc["7.S+E+T",  "MAE"] - s.loc["1.사회경제(S)", "MAE"]
    print(f"\n  S→S+E ΔMAE {dE:+.1f} | S→S+T ΔMAE {dT:+.1f} | S→S+E+T ΔMAE {dET:+.1f}  (음수=댓글이 개선)")

    cols7 = ["#4c72b0","#dd8452","#55a868","#8172b3","#937860","#c44e52","#333","#bbb","#bbb"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].bar(res.model, res.R2,  color=cols7[:len(res)])
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_ylabel("R² (확장윈도우 1-step)")
    ax[0].set_title("월별 자살자수 예측 — 설명력 (높을수록 좋음)")
    ax[0].set_xticklabels(res.model, rotation=30, ha="right", fontsize=8)
    ax[1].bar(res.model, res.MAE, color=cols7[:len(res)])
    ax[1].set_ylabel("MAE (명)"); ax[1].set_title("예측오차 (낮을수록 좋음)")
    ax[1].set_xticklabels(res.model, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DELIVERABLE / "14_monthly_suicide_7models.png", dpi=140)
    print("\nsaved deliverable/14_monthly_suicide_7models.png, monthly_suicide_7models.csv")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 4 — 병렬 실행 (run_parallel.py)
# ════════════════════════════════════════════════════════════════════════

def run_parallel_worker(idx, mode="nb"):
    exps = experiments(mode)
    partdir = OUT / ("parts" if mode == "nb" else f"parts_{mode}")
    partdir.mkdir(parents=True, exist_ok=True)
    name, desc, cfg = exps[idx]
    row = run_one(name, desc, cfg, save_artifacts=True)
    json.dump(row, open(partdir / f"_{idx:02d}.json", "w"))
    print(f"[{idx:02d} {name}] test MASE={row['test_MASE']:.3f}±{row['test_MASE_std']:.3f} "
          f"MAE={row['test_MAE']:.2f} NLL={row['test_NLL']:.3f}")


def run_parallel_aggregate(mode="nb"):
    partdir = OUT / ("parts" if mode == "nb" else f"parts_{mode}")
    rows = [json.load(open(p)) for p in sorted(partdir.glob("_*.json"))]
    summ = write_tables(rows, mode)
    print(summ[["experiment", "setting", "test_MAE",
                "test_MASE", "test_Cov80", "test_Cov90"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Suicide forecasting pipelines")
    ap.add_argument("--run", required=True,
                    choices=["preprocess", "ablation", "main", "monthly", "parallel"],
                    help="Which pipeline to execute")
    ap.add_argument("--mode",      default="nb",  help="Suite mode: nb | point")
    ap.add_argument("--gpu",       type=int, default=None, help="[preprocess] KOTE 추론 GPU 번호")
    ap.add_argument("--idx",       type=int, default=-1,   help="[parallel] worker index")
    ap.add_argument("--aggregate", action="store_true",    help="[parallel] aggregate parts")
    ap.add_argument("--list",      action="store_true",    help="[parallel] list experiment count")
    a = ap.parse_args()

    if a.run == "preprocess":
        run_preprocess(gpu=a.gpu)
    elif a.run == "ablation":
        run_ablation_suite(mode=a.mode)
    elif a.run == "main":
        lock_main_model()
    elif a.run == "monthly":
        run_monthly_suicide()
    elif a.run == "parallel":
        exps = experiments(a.mode)
        if a.list:
            print(len(exps)); return
        if a.aggregate:
            run_parallel_aggregate(mode=a.mode); return
        if a.idx < 0:
            ap.error("--idx required for parallel worker")
        run_parallel_worker(a.idx, mode=a.mode)


if __name__ == "__main__":
    main()
