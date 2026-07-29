"""전체 실행 파이프라인.

사용법
------
# 표현 생성: 정제·kiwi 키프레이즈·ko-sroberta 임베딩 (GPU 권장)
python event_online_pipeline.py --run prep --month 2025-09 --source hf [--gpu 0]
python event_online_pipeline.py --run prep --month 2025-09 --source local

# 온라인 autoregressive 트래킹 (인과적, 미래 데이터 없이 하루씩)
python event_online_pipeline.py --run track --month 2025-09

# 스트리밍 타임라인 시각화 (birth + revival per day)
python event_online_pipeline.py --run viz --month 2025-09

# 일별 리플레이 HTML 생성 (config MONTHS 전체 or 지정 월)
python event_online_pipeline.py --run export
python event_online_pipeline.py --run export --months 2025-09,2025-10,2025-11

# 로컬 덤프 → hf_staging/*.parquet PII 가드 → HF private 업로드
python event_online_pipeline.py --run upload [--dry-run]

# 로컬 덤프 → data/*.parquet 캐시 (선택)
python event_online_pipeline.py --run extract [--with-comments]

# prep → track → viz → export 한번에 (단일 월)
python event_online_pipeline.py --run all --month 2025-09 --source hf [--gpu 0]
"""
# ── 경로 설정 (config/ + utils/ 폴더를 sys.path에 추가) ─────────────────────
import sys as _sys
from pathlib import Path as _Path
_BASE = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_BASE / "config"))
_sys.path.insert(0, str(_BASE / "src"))
# ────────────────────────────────────────────────────────────────────────────
import argparse, os, sys, time
from pathlib import Path

# ── 색상 출력 ────────────────────────────────────────────────────────────────
def _c(code, s): return f"\033[{code}m{s}\033[0m"
def log(s):  print(_c("36", f"[{_ts()}] {s}"))
def ok(s):   print(_c("32", f"[{_ts()}] ✓ {s}"))
def die(s):  print(_c("31", f"[ERROR]  {s}"), file=sys.stderr); sys.exit(1)
def _ts():   return __import__("datetime").datetime.now().strftime("%H:%M:%S")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 0 — 표현 생성 (prep)
#   · 뉴스 제목 정제 (괄호·날짜·채널명 제거)
#   · kiwi 명사 추출 (NNP/NNG ≥ 2글자)
#   · 해시태그 추출 (BOILER 필터 적용)
#   · ko-sroberta-multitask 임베딩 (N×768 float32, L2 정규화)
#   입력: video_video  (source=hf|local)
#   출력: data/posts_<month>.parquet
#         data/emb_<month>.npy
# ════════════════════════════════════════════════════════════════════════

def run_prep(month, source="hf", gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["MINDCAST_DEVICE"]      = "cuda"
    from event_online_utils import prep_month
    log(f"[prep] month={month}  source={source}  gpu={gpu or 'auto'}")
    t0 = time.time()
    prep_month(month, source=source)
    ok(f"[prep] {time.time()-t0:.0f}s → data/posts_{month}.parquet, data/emb_{month}.npy")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 1 — 온라인 트래킹 (track)
#   · 매일 인과적(causal)으로 실행 — 미래 데이터 없음
#   · IDF 갱신 → 포스트 배정 → residual HDBSCAN 신규 발견
#   · 트랙 흡수(consolidation) + 라이프사이클 롤 (active/dormant/dead/revived)
#   · provisional → confirmed (2일 지속 or 3000 댓글)
#   입력: data/posts_<month>.parquet, data/emb_<month>.npy
#   출력: outputs/tracks_online[_<month>].pkl
#         outputs/events_online[_<month>].csv
# ════════════════════════════════════════════════════════════════════════

def run_track(month):
    from event_online_utils import run_tracking
    log(f"[track] month={month}")
    t0 = time.time()
    tracks, summary = run_tracking(month)
    ok(f"[track] {time.time()-t0:.0f}s  "
       f"확정={summary['n_events']}  커버리지={summary['coverage']:.1%}  "
       f"부활={summary['revived']}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 2 — 시각화 (viz)
#   · 일별 birth + revival 막대그래프 + 누적 이벤트 수
#   · 오프라인 pkl이 있으면 온/오프라인 비교 텍스트 추가
#   입력: outputs/tracks_online[_<month>].pkl
#   출력: figures/06_online_dynamics[_<month>].png
# ════════════════════════════════════════════════════════════════════════

def run_viz(month):
    from event_online_utils import viz_month
    log(f"[viz] month={month}")
    viz_month(month)
    sfx = "" if month == "2025-09" else f"_{month}"
    ok(f"[viz] → figures/06_online_dynamics{sfx}.png")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 3 — HTML 리플레이 (export)
#   · 이벤트 응집도(cohesion) 계산 + AND 라벨 생성
#   · 일별 상태(new/active/decaying/dormant/revived) 재현
#   · 뉴스 제목·댓글량 임베딩 (자립형 HTML, 외부 의존 없음)
#   입력: outputs/tracks_online[_<month>].pkl
#         data/posts_<month>.parquet, data/emb_<month>.npy  (모든 월)
#   출력: online_replay.html
# ════════════════════════════════════════════════════════════════════════

def run_export(months=None):
    from event_online_config import MONTHS as DEFAULT_MONTHS
    from event_online_utils import html_export
    mlist = months or DEFAULT_MONTHS
    log(f"[export] months={mlist}")
    t0 = time.time()
    html_export(mlist)
    ok(f"[export] {time.time()-t0:.0f}s → online_replay.html")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 4 — HF 업로드 (upload)
#   · 로컬 덤프에서 news 3개 테이블만 추출 → hf_staging/*.parquet
#   · PII 가드: 계정/비밀번호 컬럼 발견 시 중단
#              댓글 author가 전부 16-hex 해시인지 검증
#   · 가드 통과 후 private HF 데이터셋에 업로드
#   입력: $MINDCAST_DUMP (로컬 sql.gz)
#   출력: HuggingFace private dataset ($MINDCAST_HF_REPO)
# ════════════════════════════════════════════════════════════════════════

def run_upload(dry_run=False):
    from event_online_utils import build_staging, pii_guard, upload_to_hf
    from event_online_config import HF_REPO
    log(f"[upload] dry_run={dry_run}  target={HF_REPO}")
    build_staging()
    pii_guard()
    if dry_run:
        ok("[upload] dry-run 완료 — PII 가드 통과, 업로드 생략")
        return
    upload_to_hf()
    ok(f"[upload] 완료 → https://huggingface.co/datasets/{HF_REPO}")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 5 — 로컬 덤프 캐시 (extract)
#   · mysqldump에서 news 테이블만 파싱 → parquet 캐시
#   · 파이프라인 자체는 이 캐시가 없어도 동작하지만,
#     로컬 덤프를 자주 쓸 때 속도 향상용으로 사용
#   입력: $MINDCAST_DUMP
#   출력: data/video_video.parquet, data/video_channel.parquet
#         (--with-comments 시 data/video_comment.parquet 추가)
# ════════════════════════════════════════════════════════════════════════

def run_extract(with_comments=False):
    from event_online_utils import extract_db
    log(f"[extract] with_comments={with_comments}")
    extract_db(with_comments=with_comments)
    ok("[extract] 완료 → data/video_video.parquet (+ channel, comment)")


# ════════════════════════════════════════════════════════════════════════
# Pipeline 6 — 전체 (all)
#   prep → track → viz → export  (단일 월 end-to-end)
# ════════════════════════════════════════════════════════════════════════

def run_all(month, source="hf", gpu=None):
    log(f"[all] month={month}  source={source}  gpu={gpu or 'auto'}")
    run_prep(month, source=source, gpu=gpu)
    run_track(month)
    run_viz(month)
    run_export([month])
    ok(f"[all] 전체 파이프라인 완료 (month={month})")


# ════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="mindcast-online 파이프라인 실행기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--run", required=True,
                    choices=["prep","track","viz","export","upload","extract","all"],
                    help="실행할 파이프라인")
    ap.add_argument("--month",  default="2025-09",
                    help="대상 월  YYYY-MM  (기본: 2025-09)")
    ap.add_argument("--source", choices=["hf","local"], default="hf",
                    help="video_video 소스  hf|local  (기본: hf)")
    ap.add_argument("--gpu",    default=None,
                    help="GPU 번호  (prep/all 에서 CUDA_VISIBLE_DEVICES 설정)")
    ap.add_argument("--months", default=None,
                    help="export 대상 월 목록  쉼표 구분  (기본: config MONTHS 전체)")
    ap.add_argument("--dry-run",       action="store_true",
                    help="upload: PII 가드만 실행, 실제 업로드 생략")
    ap.add_argument("--with-comments", action="store_true",
                    help="extract: video_comment 테이블도 포함")
    a = ap.parse_args()

    if a.run == "prep":
        run_prep(a.month, source=a.source, gpu=a.gpu)
    elif a.run == "track":
        run_track(a.month)
    elif a.run == "viz":
        run_viz(a.month)
    elif a.run == "export":
        months = a.months.split(",") if a.months else None
        run_export(months)
    elif a.run == "upload":
        run_upload(dry_run=a.dry_run)
    elif a.run == "extract":
        run_extract(with_comments=a.with_comments)
    elif a.run == "all":
        run_all(a.month, source=a.source, gpu=a.gpu)


if __name__ == "__main__":
    main()
