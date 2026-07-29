# suicide_predict — 뉴스댓글 기반 자살 관련 지표 예측 모델

## 개요

**두 개의 독립적인 예측기**로 구성됩니다.

```
┌──────────────────────────────────────────────────────────────────┐
│  예측기 1: 일별 상담전화 건수 예측                                   │
│                                                                    │
│  입력: 과거 56일 상담건수 이력 (+ 감정/토픽/캘린더 ablation)         │
│  출력: 다음날 상담건수 + 80/90% 예측구간                             │
│  모델: ExoDLinear point + split-conformal (Setting A 채택)         │
│  결론: 감정·토픽 추가해도 상담 이력만 쓸 때와 성능 차이 없음           │
├──────────────────────────────────────────────────────────────────┤
│  예측기 2: 월별 자살 사망자수 예측                                    │
│                                                                    │
│  입력: 사회경제 변수(S) + KOTE 감정(E) + 뉴스 토픽(T)               │
│  출력: 해당 월 자살 사망자수                                          │
│  모델: RidgeCV × 7가지 피처 조합                                    │
│  결론: S(사회경제)만 써도 강하며, 댓글 피처 증분 효과 검증 목적         │
└──────────────────────────────────────────────────────────────────┘
```

두 예측기는 **타겟, 시간 단위, 모델, 데이터 파이프라인이 모두 별개**입니다.  
공통으로 사용하는 것은 KOTE 감정 추론 결과와 뉴스 토픽뿐입니다.

---

## 파일 구조

```
mindcast/
├── config/suicide_config.py      # 모든 상수·경로·설정
├── utils/suicide_utils.py        # 전처리·모델·학습·평가 함수 전체
├── pipeline/suicide_pipeline.py  # 실행 파이프라인 (CLI)
└── run/suicide_run.sh            # 쉘 실행 스크립트

kote_based_suicide_predict/       # 원본 프로젝트 (데이터·캐시·산출물)
├── cache/
│   ├── raw/                      # target parquet, 댓글 parquet
│   ├── emotion/                  # KOTE 추론 결과
│   └── topic/                    # 토픽 피처
├── data/                         # mapping_ver1.json 등
├── outputs/
│   ├── checkpoints/              # 모델 가중치 (.pt)
│   ├── predictions/              # 예측 결과 parquet
│   └── figures/                  # 시각화
└── deliverable/                  # 최종 산출물 (그래프·모델카드)
```

---

## 실행 순서

```bash
cd mindcast/run

# ── 전처리 (댓글 → KOTE → 감정집계 → 토픽) ──────────────────────
./suicide_run.sh -g 2 preprocess

# ── 일별 예측기: Ablation A~F ────────────────────────────────────
./suicide_run.sh -g 2 ablation          # NB head (기본)
./suicide_run.sh -g 2 ablation point    # point head

# ── 일별 예측기: 병렬 실행 ──────────────────────────────────────
./suicide_run.sh -g 2 parallel          # 백그라운드 병렬 → 집계

# ── 일별 예측기: 메인 모델 확정 ──────────────────────────────────
./suicide_run.sh -g 2 main

# ── 월별 예측기 ──────────────────────────────────────────────────
./suicide_run.sh -g 2 monthly

# ── 전체 한번에 ──────────────────────────────────────────────────
./suicide_run.sh -g 2 all
```

또는 Python 직접 실행:
```bash
cd mindcast/pipeline
python suicide_pipeline.py --run preprocess --gpu 2
python suicide_pipeline.py --run ablation --mode nb
python suicide_pipeline.py --run main
python suicide_pipeline.py --run monthly
```

---

## 파이프라인 단계

### Pipeline 0 — 전처리

| 단계 | 함수 | 입력 | 출력 |
|------|------|------|------|
| 1/4 target 준비 | `build_target()` | HuggingFace `MindCastSogang/SuicideDataset` | `cache/raw/target_call_counts_2018_2023.parquet` |
| 2/4 댓글 다운로드 | `build_comments()` | HuggingFace `Youtube_news_preprocessed_data` | `cache/raw/youtube_news_comments.parquet` |
| 3/4 KOTE 추론 | `kote_inference()` | `youtube_news_comments.parquet` (GPU 필요) | `cache/emotion/comment_kote_probs.parquet`<br>`cache/emotion/emotion_labels.json` |
| 4/4 감정 집계 | `daily_emotion()` | `comment_kote_probs.parquet` | `cache/emotion/daily_kote_features.parquet` |
| 4/4 토픽 클러스터링 | `topic_features()` | `youtube_news_comments.parquet` | `cache/topic/daily_topic_features.parquet` |

**cache 디렉터리 전체 구조:**
```
data/suicide_predict/
└── cache/
    ├── raw/
    │   ├── target_call_counts_2018_2023.parquet   ← 일별 상담건수
    │   └── youtube_news_comments.parquet           ← 댓글 원본 (date, title, comment_id, ...)
    ├── emotion/
    │   ├── comment_kote_probs.parquet              ← 댓글별 44감정 확률 (emotion_0~43)
    │   ├── emotion_labels.json                     ← KOTE 44감정 레이블 목록
    │   └── daily_kote_features.parquet             ← 날짜별 감정 집계 피처
    └── topic/
        └── daily_topic_features.parquet            ← 날짜별 토픽 비율 피처 (topic_ratio_0~9)
```

---

### Pipeline 1 — 일별 Ablation

**Setting A~F:**
```
A: 상담 이력만                    ← 최종 채택
B: A + 캘린더 피처
C: B + 댓글 볼륨 (댓글량)
D: C + KOTE 감정 (4 대분류)
E: D + 뉴스 토픽
F: E + 감정×토픽 상호작용

ablation 후 MASE가 가장 낮은 모델 채택
```

**산출물:**
```
outputs/
├── metrics/
│   ├── results_per_seed.csv          ← 실험×시드×split 전체 raw 결과
│   ├── results_summary.csv           ← 실험별 4-seed 평균/std 요약
│   └── results_test.md               ← 테스트셋 성능 마크다운 테이블
├── predictions/
│   └── {ExperimentName}_test.parquet ← 각 실험 seed=42 테스트 예측값
└── checkpoints/
    ├── exodlinear_D_best.pt          ← Setting D (NB head) 체크포인트
    └── exodlinear_point_best.pt      ← Setting D (point head) 체크포인트
```

> `--mode point`로 실행 시 파일명에 `point_` 접두사 붙음 (예: `results_point_summary.csv`)

---

### Pipeline 2 — 메인 모델 확정

검증셋 MASE 기준 **Setting A** 채택:
```python
MAIN_CFG = dict(model="exodlinear", head="point", lookback=56, hidden=64, dropout=0.5)
```

**산출물:**
```
outputs/
├── predictions/
│   └── MAIN_test.parquet             ← 메인 모델 seed=42 테스트 예측값
└── checkpoints/
    └── main_model.pt                 ← 최종 모델 가중치 (state_dict + cfg + meta)

deliverable/
├── 00_MAIN_prediction.png            ← valid(2022)+test(2023) 예측 시각화
└── MAIN_MODEL.md                     ← 모델 카드 (성능 테이블 포함)
```

---

### Pipeline 3 — 월별 예측기

**7개 피처 조합 × RidgeCV × expanding window:**
```
단독:  S,  E,  T
복합:  S+E, S+T, E+T, S+E+T
기준선: 계절평균, lag-12
```

**산출물:**
```
deliverable/
├── monthly_suicide_7models.csv       ← 7조합 + 기준선 성능 테이블 (R², MAE 등)
└── 14_monthly_suicide_7models.png    ← R²/MAE 막대그래프 비교
```

---

## 주요 설정 (suicide_config.py)

| 항목 | 값 |
|------|----|
| lookback | 56일 |
| Train | 2020-01-01 ~ 2021-12-31 |
| Valid | 2022-01-01 ~ 2022-12-31 |
| Test | 2023-01-01 ~ 2023-12-31 |
| KOTE 모델 | `searle-j/kote_for_easygoing_people` |
| 토픽 K | 10 |
| 감정 대분류 | 4개 (기쁨/슬픔/분노/중립) |
| 앵커 수준 | 최근 7일 raw 평균 |

---

## 성능 결과

**일별 예측기 — ExoDLinear point + conformal (Setting A, 4-seed)**
```
         MAE    RMSE   MASE          80%Cov  90%Cov
valid   26.59  33.28  0.515±0.006   0.803   0.904
test    23.76  30.54  0.461±0.002   0.843   0.927
```

**월별 예측기 — RidgeCV 7조합 (test: 2022-01 ~ 2023-10)**
```
S+E 조합: R²=+0.329, MAE=66.2  ← 최고
S 단독:   R²=+0.290, MAE=69.3
E 단독:   R²=-0.011, MAE=82.9
T 포함 조합: 전반적으로 성능 저하
```

---

## 데이터 흐름

```
[HuggingFace: MindCastSogang/SuicideDataset]
[HuggingFace: MindCastSogang/Youtube_news_preprocessed_data]
        │
        ▼
  build_target()     build_comments()
        │                   │
  target_call.pq     youtube_comments.pq
        │                   │
        │      kote_inference() → daily_emotion()
        │                   │
        │       daily_kote_features.pq
        │                   │
        │      topic_features()
        │                   │
        │       daily_topic_features.pq
        │                   │
   ┌────┴───────────────────┘
   │
   ├── [일별] dataset → train → ablation → main_model
   │
   └── [월별] load_socio (월별 사회경제변수) + monthly_emotion + monthly_topic → RidgeCV
```

---

## 트러블슈팅

### HuggingFace 401 / 404

```bash
# 토큰 설정
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# 또는 영구 저장
huggingface-cli login
```

`MindCastSogang` 조직의 접근 권한이 있는 계정 토큰인지 확인.

### ModuleNotFoundError: holidays / properscoring

```bash
pip install holidays properscoring
```

### CUDA Out of Memory (kote_inference)

`suicide_config.py`에서 `KOTE_BATCH = 512` → 줄이기:
```python
KOTE_BATCH = 128   # 또는 256
```

---

## 주의사항

- `13_monthly_suicide.py` 토픽 클러스터링은 전체 데이터로 fit → **누수 있음**
- 댓글 데이터는 2020-03 ~ 2023-10만 존재 (2020-01~02 없음)
- `dataset.py` 정규화 stats는 train rows로만 fit (leakage-safe)
- `04_topic_features.py` TF-IDF/KMeans는 TRAIN_END(2021-12-31) 이전으로만 fit (leakage-safe)
