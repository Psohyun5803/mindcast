# emotion_classifier — 데이터 입출력 명세

## 전체 흐름

```
[HF 또는 로컬 댓글]
        │
   teacher (보조)
        │
   teacher_targets.parquet
        │
      prep-a
        │
   stagea_normalized.parquet
        │
      train-a
        │
   stage_a/student_comment_distill.pt
        │
      prep-b ← [Stage B gold 데이터 (EC_STAGEB_INPUT)]
        │
   stageb_normalized.parquet
        │
      train-b
        │
   stage_b/stageB_adapter_checkpoint.pt
        │
      export
        │
   offline_bundle.pt
        │
     predict ← [입력 데이터]
        │
   prediction_*.json / .csv
```

---

## 단계별 입출력

### teacher

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | HF `MindCastSogang/Youtube_news_preprocessed_data` | parquet (HF) | `comment`, `news_title`, `date` 등 |
| **출력** | `data/emotion_classifier/teacher_targets.parquet` | parquet | `comment`, `teacher_prob_00_불평/불만` … `teacher_prob_43_안심/신뢰` (44개 소프트 확률 컬럼) |
| **출력** | `data/emotion_classifier/teacher_targets_file_stats.csv` | csv | `source_file`, `raw_rows`, `kept_rows` |
| **출력** | `data/emotion_classifier/teacher_targets_id2label.json` | json | `{"0": "불평/불만", …, "43": "안심/신뢰"}` |

---

### prep-a

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `data/emotion_classifier/teacher_targets.parquet` (기본) | parquet | `comment` + `teacher_prob_*` 44개 필수 |
| **출력** | `data/emotion_classifier/stagea_normalized.parquet` | parquet | 정규화 완료 동일 스키마, NaN 행 제거 |

---

### train-a

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `data/emotion_classifier/stagea_normalized.parquet` (기본) | parquet | `comment`, `teacher_prob_*` 44개 |
| **출력** | `models/emotion_classifier/stage_a/student_comment_distill.pt` | pt | KcELECTRA-base state dict (best val 기준) |
| **출력** | `models/emotion_classifier/stage_a/student_comment_distill.meta.json` | json | `student_model`, `rows`, `prob_columns`, `seed`, `created_at_utc` |
| **출력** | `models/emotion_classifier/stage_a/train_history.csv` | csv | `epoch`, `train_loss`, `val_loss`, `top1_match`, `prob_mae` |

---

### prep-b

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `EC_STAGEB_INPUT` (필수 지정) | xlsx / csv / parquet | `comment`, `sarcasm_label` 필수; `news_title`, `emotion_labels` 권장 |
| **출력** | `data/emotion_classifier/stageb_normalized.parquet` | parquet | 정규화 완료, `sarcasm_label` 검증 후 저장 |

---

### train-b

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `data/emotion_classifier/stageb_normalized.parquet` (기본) | parquet | `comment`, `news_title`, `sarcasm_label`, `emotion_labels` |
| **입력** | `models/emotion_classifier/stage_a/student_comment_distill.pt` (기본) | pt | Stage A student state dict |
| **출력** | `models/emotion_classifier/stage_b/stageB_adapter_checkpoint.pt` | pt | adapter state dict + `label_map` + `args` |
| **출력** | `models/emotion_classifier/stage_b/stageB_adapter_checkpoint.meta.json` | json | `rows`, `best_epoch`, `student_model`, `created_at_utc` |
| **출력** | `models/emotion_classifier/stage_b/tokenizer/` | 디렉토리 | HuggingFace tokenizer 저장본 |
| **출력** | `models/emotion_classifier/stage_b/train_history.csv` | csv | `epoch`, `train_loss`, `sarcasm_f1`, `positive_emotion_top1_hit_gain`, `negative_identity_rate` |
| **출력** | `models/emotion_classifier/stage_b/best_val_predictions.csv` | csv | best epoch val set 예측 결과 |

---

### export

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `models/emotion_classifier/stage_a/student_comment_distill.pt` | pt | Stage A state dict |
| **입력** | `models/emotion_classifier/stage_b/stageB_adapter_checkpoint.pt` *(선택)* | pt | Stage B state dict; 없으면 `stagea_only` 번들 생성 |
| **출력** | `models/emotion_classifier/offline_bundle.pt` | pt | `bundle_variant`: `stagea_only` 또는 `stagea_stageb` |
| **출력** | `models/emotion_classifier/offline_bundle.pt.meta.json` | json | `bundle_variant`, `student_model`, `num_labels`, `created_at_utc` 등 |

---

### predict

| 구분 | 경로/형식 | 스키마 / 내용 |
|---|---|---|
| **입력 (번들)** | `models/emotion_classifier/offline_bundle.pt` (기본 자동 탐색) | — |
| **입력 (데이터)** | 로컬 파일 또는 HF 소스 | `comment` 필수, `title` / `news_title` 선택 |
| **출력** | `data/emotion_classifier/outputs/prediction_<source>_<YYYYMMDD>.json` (기본) | json |

출력 컬럼 / 필드:

| 필드 | 타입 | 설명 |
|---|---|---|
| `comment` | str | 입력 댓글 |
| `title` | str | 입력 뉴스 제목 (없으면 빈 문자열) |
| `title_provided` | bool | title 입력 유무 |
| `gate_score` | float | Stage B 보정 강도 신호 (0~1); `stagea_only`면 항상 0 |
| `base_emotion_label_fine` | str | Stage A base 1위 소분류 |
| `base_emotion_label_major` | str | Stage A base 1위 대분류 |
| `base_emotion_score` | float | Stage A 1위 확률 |
| `primary_emotion_label` | str | 최종 1위 소분류 |
| `primary_emotion_major` | str | 최종 1위 대분류 |
| `primary_emotion_score` | float | 최종 1위 확률 |
| `top_5_json` | str (JSON) | 상위 5개 `[{label, major, score}]` |

---

### attach-major

| 구분 | 형식 | 스키마 / 내용 |
|---|---|---|
| **입력** | csv / parquet | 소분류 레이블 컬럼 포함 파일 |
| **출력** | csv / parquet | 기존 컬럼 + `{col}_major` 대분류 컬럼 추가 |
