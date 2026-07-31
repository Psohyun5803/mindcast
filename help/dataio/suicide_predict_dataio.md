# suicide_predict — 데이터 입출력 명세

## 전체 흐름

```
[HF: suicide_prevent_call]  [HF: Youtube_news_preprocessed_data]
          │                              │
     build_target                  build_comments
          │                              │
  cache/raw/target_call_counts.parquet   cache/raw/youtube_news_comments.parquet
                                         │
                                   kote_inference (GPU)
                                         │
                                  cache/emotion/comment_kote_probs.parquet
                                         │
                          ┌──────────────┴──────────────┐
                     daily_emotion                 topic_features
                          │                              │
              cache/emotion/daily_kote_features.parquet  cache/topic/daily_topic_features.parquet
                          └──────────────┬──────────────┘
                                         │
                                 [학습 데이터 구성]
                                    train/valid/test
                                         │
                        ┌────────────────┼────────────────┐
                     ablation           main            monthly
                        │                │                │
               outputs/metrics/     checkpoints/      deliverable/
               results_*.csv        main_model.pt     monthly_suicide_7models.csv
               outputs/predictions/ deliverable/      deliverable/
               checkpoints/         MAIN_MODEL.md     14_monthly_suicide_7models.png
               exodlinear_*.pt      00_MAIN_prediction.png
```

모든 경로는 `data/suicide_predict/` 하위.

---

## 전처리 단계별 입출력

### build_target (전처리 1/4)

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | HF `MindCastSogang/suicide_prevent_call` | csv (HF) | 자살예방상담전화 일별 상담건수 (2018~2023) |
| **출력** | `cache/raw/target_call_counts_2018_2023.parquet` | parquet | `date`, `y` (일별 상담건수) |

---

### build_comments (전처리 2/4)

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | HF `MindCastSogang/Youtube_news_preprocessed_data` `preprocessed/v1/{2020~2023}` | parquet (HF) | `comment`, `news_title`, `date` 등 |
| **출력** | `cache/raw/youtube_news_comments.parquet` | parquet | `comment`, `title`, `date` (2020~2023 병합) |

---

### kote_inference (전처리 3/4) — GPU 필요

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `cache/raw/youtube_news_comments.parquet` | parquet | `comment`, `date` |
| **출력** | `cache/emotion/comment_kote_probs.parquet` | parquet | `comment`, `date`, `emotion_0` … `emotion_43` (44개 sigmoid 확률) |
| **출력** | `cache/emotion/emotion_labels.json` | json | `["불평/불만", …, "안심/신뢰"]` 44개 라벨 순서 |

모델: `searle-j/kote_for_easygoing_people` (batch=512, max_len=128)

---

### daily_emotion (전처리 4a/4)

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `cache/emotion/comment_kote_probs.parquet` | parquet | 댓글별 44개 감정 확률 |
| **출력** | `cache/emotion/daily_kote_features.parquet` | parquet | 일별 집계 피처 |

`daily_kote_features.parquet` 컬럼:

| 컬럼 | 설명 |
|---|---|
| `date` | 날짜 |
| `n_comments` | 해당 날짜 댓글 수 |
| `emo_mean_{0~43}` | 감정별 일평균 확률 (44개) |
| `emo_sum_{0~43}` | 감정별 일합계 (44개) |
| `emo_max_{0~43}` | 감정별 일최대 (44개) |
| `emo_std_{0~43}` | 감정별 일표준편차 (44개) |

---

### topic_features (전처리 4b/4)

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `cache/raw/youtube_news_comments.parquet` | parquet | `date`, `title` |
| **출력** | `cache/topic/daily_topic_features.parquet` | parquet | 일별 토픽 비율 피처 |

`daily_topic_features.parquet` 컬럼:

| 컬럼 | 설명 |
|---|---|
| `date` | 날짜 |
| `topic_ratio_{0~9}` | KMeans 10-토픽별 일비율 (train 데이터로 fit) |

토픽 학습 기준: `TOPIC_TRAIN_END = "2021-12-31"` 이전 데이터로만 fit → 미래 데이터 누수 없음

---

## 학습 단계별 입출력

### ablation (설정 A~F, 4 seed)

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `cache/raw/target_call_counts_2018_2023.parquet` | parquet | `date`, `y` |
| **입력** | `cache/emotion/daily_kote_features.parquet` *(감정 설정 시)* | parquet | 일별 감정 피처 |
| **입력** | `cache/topic/daily_topic_features.parquet` *(토픽 설정 시)* | parquet | 일별 토픽 피처 |
| **출력** | `outputs/metrics/results_per_seed.csv` | csv | 실험별·seed별 전체 지표 |
| **출력** | `outputs/metrics/results_summary.csv` | csv | 실험별 mean±std 요약 |
| **출력** | `outputs/predictions/{name}_test.parquet` | parquet | seed=42 test 예측값 (실험별) |
| **출력** | `outputs/checkpoints/exodlinear_D_best.pt` | pt | ExoDLinear-D best state dict |
| **출력** | `outputs/checkpoints/exodlinear_point_best.pt` | pt | ExoDLinear-point best state dict |

`results_per_seed.csv` 컬럼: `experiment`, `split`, `seed`, `MAE`, `RMSE`, `MASE`, `WAPE`, `sMAPE`, `NLL`, `CRPS`, `Cov80`, `Cov90`, `Width80`, `Width90`

ablation 설정:

| 설정 | 피처 |
|---|---|
| A | target only (lookback 56일 상담건수) |
| B | A + 캘린더 (요일·공휴일·월) |
| C | B + volume (댓글량) |
| D | C + emotion (KOTE 4 그룹: 기쁨/슬픔/분노/중립) |
| E | D + topic (KMeans 10-토픽 비율) |
| F | E + emo×topic 교호작용 |

---

### main (메인 모델 확정, 4 seed)

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | ablation 동일 입력 (Setting A 구성) | parquet | `date`, `y` |
| **출력** | `outputs/checkpoints/main_model.pt` | pt | seed=42 state dict + `cfg` + `meta` |
| **출력** | `outputs/predictions/MAIN_test.parquet` | parquet | `tdate`, `y`, `pred_mean`, `pred_std` |
| **출력** | `deliverable/00_MAIN_prediction.png` | png | valid(2022) + test(2023) 예측 vs 실제 시각화 |
| **출력** | `deliverable/MAIN_MODEL.md` | md | 모델 카드 (성능 지표·데이터 분할·모델 설명) |

`MAIN_test.parquet` 컬럼:

| 컬럼 | 설명 |
|---|---|
| `tdate` | 날짜 |
| `y` | 실제 상담건수 |
| `pred_mean` | 예측 평균 (anchored mean: 최근 7일 평균 × exp(δ)) |
| `pred_std` | 예측 표준편차 (conformal 보정) |

데이터 분할: Train `2020-02-26~2021-12-31` / Valid `2022` / Test `2023-01~10`

---

### monthly (월별 자살자수 예측)

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | HF `MindCastSogang/SuicideDataset` `suicide_base_data_2020_2024_20260222.csv` | csv (HF) | `month`, `suicide_count`, 사회경제 피처 |
| **입력** | `cache/emotion/daily_kote_features.parquet` | parquet | 월별 집계용 |
| **입력** | `cache/topic/daily_topic_features.parquet` | parquet | 월별 집계용 |
| **출력** | `deliverable/monthly_suicide_7models.csv` | csv | 7개 피처 조합 × 확장윈도우 1-step 평가 결과 |
| **출력** | `deliverable/14_monthly_suicide_7models.png` | png | R²·MAE 비교 막대그래프 |

7개 피처 조합: `S(사회경제)` / `E(감정)` / `T(토픽)` / `S+E` / `S+T` / `E+T` / `S+E+T`

`monthly_suicide_7models.csv` 컬럼: `model`, `n_feat`, `R2`, `MAE`, `RMSE`, `n`

---

## 데이터 분할 기준

| split | 기간 | 행 수 |
|---|---|---|
| train | 2020-02-26 ~ 2021-12-31 | 675일 |
| valid | 2022-01-01 ~ 2022-12-31 | 365일 |
| test | 2023-01-01 ~ 2023-10-31 | 296일 |

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `HF_TOKEN` | `~/.cache/huggingface/token` | HF 인증 토큰 |
