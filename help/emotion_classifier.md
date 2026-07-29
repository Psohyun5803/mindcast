# emotion_classifier — 한국어 댓글 감정 분류 모델

## 개요

뉴스 댓글의 감정을 분류하는 2단계 모델입니다.

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage A: comment-only 학생 모델 (지식 증류)                        │
│                                                                    │
│  Teacher: KOTE (searle-j/kote_for_easygoing_people, 44라벨)       │
│  Student: KcELECTRA-base → 교사 확률 soft-label 증류               │
│  출력:    44개 감정 소분류 확률 + 대분류 (긍정/부정/중립)              │
├──────────────────────────────────────────────────────────────────┤
│  Stage B: 풍자 감정 어댑터 (sarcasm-emotion adapter)                │
│                                                                    │
│  입력:  comment + news_title (title 선택)                          │
│  역할:  title과 comment 간 맥락 충돌 감지 → 감정 보정               │
│  gate:  풍자(sarcasm) 여부 확률 신호로 보정 강도 결정               │
└──────────────────────────────────────────────────────────────────┘
```

- `title`이 없으면 Stage A base 결과만 반환합니다.
- `title`이 있으면 Stage B adapter가 추가 보정합니다.
- 출력 스키마는 두 경우 모두 동일합니다.

---

## 파일 구조

```
personalrepo/
├── src/emotion_classifier_utils.py     # 학습·추론 유틸리티 전체
├── pipeline/emotion_classifier_pipeline.py  # 실행 파이프라인 (CLI)
└── run/emotion_classifier_run.sh       # 쉘 실행 스크립트

emotion_classifier/
├── src/sarcasm_emotion_adapter/        # 패키지 (모델 정의·데이터 I/O·라벨)
│   ├── modeling.py                     # StageAStudent, StageBSarcasmAdapter
│   ├── dataio.py                       # load_dataset_frame, write_dataframe
│   ├── labels.py                       # label_map, major_mapping 로드
│   └── offline.py                      # OfflineSarcasmEmotionPredictor, export
├── scripts/                            # 번호 순 CLI 래퍼
│   ├── 00_normalize_dataset.py
│   ├── 01_offline_infer.py
│   ├── 10_prepare_stage_a_teacher_targets.py
│   ├── 12_train_stage_a.py
│   ├── 13_prepare_stage_b_targets.py
│   ├── 14_train_stage_b.py
│   ├── 15_export_offline_bundle.py
│   ├── 20_attach_major_labels.py
│   ├── 21_merge_gold_annotations.py
│   └── predict_emotion.py
└── docs/                               # 상세 원본 문서
```

---

## 실행 순서

### 학습 전체 (권장)

```bash
cd personalrepo/run

# EC_LABEL_MAP 필수 — assets/kote_id2label.json 경로 지정
export EC_LABEL_MAP=/path/to/kote_id2label.json

./emotion_classifier_run.sh -g 0 all
# teacher → train-a → prep-b → train-b → export 순서로 실행
```

### 단계별 실행

```bash
# 1. 교사 확률 생성 (KOTE → parquet)
./emotion_classifier_run.sh -g 0 teacher

# 2. Stage A 학습 (지식 증류)
./emotion_classifier_run.sh -g 0 train-a

# 3. Stage B 타겟 준비
export EC_LABEL_MAP=/path/to/kote_id2label.json
./emotion_classifier_run.sh prep-b

# 4. Stage B 학습 (풍자 어댑터)
./emotion_classifier_run.sh -g 0 train-b

# 5. 오프라인 번들 내보내기
./emotion_classifier_run.sh export
```

### 추론

```bash
# 오프라인 추론
./emotion_classifier_run.sh predict data/comments.json data/results.csv

# 대분류 컬럼 추가 (소분류 → 긍정/부정/중립)
./emotion_classifier_run.sh attach-major data/results.csv data/results_major.csv
```

경로 환경변수 (미지정 시 기본값 사용):

| 변수 | 기본값 |
|---|---|
| `EC_DATA_DIR` | `data/emotion_classifier` |
| `EC_MODEL_DIR` | `models/emotion_classifier` |
| `EC_LABEL_MAP` | (없음, 필수) |
| `EC_TEACHER_OUT` | `$EC_DATA_DIR/teacher_targets.parquet` |
| `EC_STAGEA_DIR` | `$EC_MODEL_DIR/stage_a` |
| `EC_STAGEB_TARGETS` | `$EC_DATA_DIR/stageb_targets.parquet` |
| `EC_STAGEB_DIR` | `$EC_MODEL_DIR/stage_b` |
| `EC_BUNDLE_OUT` | `$EC_MODEL_DIR/offline_bundle.pt` |

---

## 파이프라인 단계별 산출물

### Pipeline 0 — 교사 확률 생성 (`teacher`)

HuggingFace 댓글 데이터 → KOTE 교사 모델로 soft-label 생성

| 산출물 | 형식 | 설명 |
|---|---|---|
| `teacher_targets.parquet` | parquet | 댓글 행 + `teacher_prob_00_*` ~ `teacher_prob_43_*` 44개 확률 컬럼 + `teacher_pred_label` |
| `teacher_targets_file_stats.csv` | csv | 소스 파일별 raw/kept 행 수 통계 |
| `teacher_targets_id2label.json` | json | `{index: label_name}` 라벨 맵 |

### Pipeline 1 — Stage A 학습 (`train-a`)

교사 확률 → KcELECTRA-base student 지식 증류

| 산출물 | 형식 | 설명 |
|---|---|---|
| `stage_a/student_comment_distill.pt` | pt | 최고 val 성능 기준 best state dict |
| `stage_a/student_comment_distill.meta.json` | json | 학습 메타 (모델명·행 수·prob 컬럼 목록·seed) |
| `stage_a/train_history.csv` | csv | epoch별 train_loss / val_loss / top1_match / prob_mae |

### Pipeline 2 — Stage B 타겟 준비 (`prep-b`)

정규화 데이터 + label map → Stage B 학습용 타겟 컬럼 생성

| 산출물 | 형식 | 설명 |
|---|---|---|
| `stageb_targets.parquet` | parquet | `sarcasm_label`, `final_emotion_target`, `final_emotion_target_id`, `final_emotion_target_source` 컬럼 추가 |

### Pipeline 3 — Stage B 학습 (`train-b`)

Stage A checkpoint + gold 데이터 → 풍자 감정 어댑터 학습

| 산출물 | 형식 | 설명 |
|---|---|---|
| `stage_b/stageB_adapter_checkpoint.pt` | pt | best epoch state dict + label_map + args |
| `stage_b/stageB_adapter_checkpoint.meta.json` | json | 학습 메타 (입력 행 수·best_epoch·student_model) |
| `stage_b/tokenizer/` | 디렉터리 | HuggingFace tokenizer 저장본 |
| `stage_b/train_history.csv` | csv | epoch별 train_loss / sarcasm_f1 / positive_emotion_top1_hit_gain / negative_identity_rate |
| `stage_b/best_val_predictions.csv` | csv | best epoch val set 예측 결과 |

### Pipeline 4 — 번들 내보내기 (`export`)

Stage A + Stage B checkpoint → 단일 오프라인 추론 번들

| 산출물 | 형식 | 설명 |
|---|---|---|
| `offline_bundle.pt` | pt | 추론에 필요한 모든 정보 포함 (stagea_only 또는 stagea_stageb variant) |

### Pipeline 5 — 오프라인 추론 (`predict`)

번들 + 입력 파일 → 감정 예측 결과

| 산출물 | 형식 | 설명 |
|---|---|---|
| `results.csv` / `.json` | csv 또는 json | `base_emotion`, `primary_emotion`, `top_5`, `gate_score`, `title_provided` 포함 |

### Pipeline 6 — 대분류 컬럼 추가 (`attach-major`)

소분류 감정 레이블 → 대분류(긍정/부정/중립) 컬럼 추가

| 산출물 | 형식 | 설명 |
|---|---|---|
| `results_major.csv` | csv | 기존 컬럼 + `{col}_major` 컬럼 추가 |

---

## 데이터 I/O

### 지원 입력 형식

- 로컬 파일: `csv`, `json`, `parquet`, `xlsx`/`xls`
- HuggingFace 소스: repo root, 특정 subdirectory, 특정 파일 경로

### 원본 컬럼 자동 인식

| 표준 컬럼 | 원본에서 인식하는 이름들 |
|---|---|
| `comment` | `comment`, `comment_text`, `text`, `content`, `body` |
| `news_title` | `news_title`, `title`, `raw_title`, `headline` |
| `date` | `date`, `news_date`, `article_date`, `published_at`, `created_at` |

### 정규화 후 표준 컬럼

`candidate_id`, `comment`, `title`, `news_title`, `date`, `news_date`, `dataset_date`, `source_file`

- `comment`는 필수, `title`은 선택 (없으면 빈 문자열)
- `candidate_id`가 없으면 자동 생성

### 추론 JSON 단건 입력 예시

```json
{
  "comment": "와 정말 잘한다",
  "title": "같은 실수 반복한 정부",
  "date": "2026-07-29"
}
```

### 추론 출력 예시

```json
{
  "comment": "와 정말 잘한다",
  "title": "같은 실수 반복한 정부",
  "title_provided": true,
  "gate_score": 0.82,
  "base_emotion": { "label": "기쁨/신남", "major": "긍정", "score": 0.61 },
  "primary_emotion": { "label": "화남/분노", "major": "부정", "score": 0.73 },
  "top_5": [
    { "label": "화남/분노", "major": "부정", "score": 0.73 },
    { "label": "어이없음",  "major": "부정", "score": 0.68 }
  ]
}
```

| 필드 | 설명 |
|---|---|
| `title_provided` | title 입력 유무 |
| `gate_score` | adapter 보정 강도 신호 (0~1) |
| `base_emotion` | comment-only base 모델의 1위 감정 |
| `primary_emotion` | Stage B 보정 후 최종 1위 감정 |
| `top_5` | 최종 상위 5개 감정 + 점수 |

### Stage A 학습 입력 최소 요건

- `comment` 컬럼
- `teacher_prob_*` 컬럼들 (44개)

### Stage B 학습 입력 최소 요건 (gold 데이터 기준)

- `comment`, `sarcasm_label`, `emotion_labels`

---

## 유의사항

- Stage B 학습은 반드시 Stage A checkpoint 기반으로 합니다 (`--stagea-checkpoint` 필수).
- `EC_LABEL_MAP`을 지정하지 않으면 `prep-b`, `train-b` 단계에서 오류가 납니다. 기본 KOTE 44라벨은 `assets/kote_id2label.json`을 사용하세요.
- bundle에는 Stage B adapter 포함 여부에 따라 두 종류가 있습니다: `stagea_only` (base만), `stagea_stageb` (보정 포함).
- CUDA가 없으면 CPU fallback으로 실행되며, 실제 데이터에서는 속도가 느려집니다.
- Stage A checkpoint 자동 선택 시 여러 실험 결과가 섞여 있으면 의도치 않은 파일을 선택할 수 있으므로 재현 실험 시에는 경로를 명시하세요.
