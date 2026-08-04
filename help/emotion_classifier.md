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
├── config/emotion_classifier_config.py      # 기본 경로 설정 (환경변수로 덮어쓰기 가능)
├── src/emotion_classifier_utils.py          # 학습·추론 공통 유틸리티
├── pipeline/emotion_classifier_pipeline.py  # 실행 파이프라인 (CLI)
└── run/emotion_classifier_run.sh            # 쉘 실행 스크립트
```

```
root/data/emotion_classifier/
├── assets/                              # kote_id2label.json, mapping_ver1.json
├── cache/                               # 학습 중간 산출물
│   ├── teacher_targets.parquet          # teacher 단계 출력
│   ├── teacher_targets_file_stats.csv
│   ├── teacher_targets_id2label.json
│   └── stagea_normalized.parquet        # prep-a 출력
├── checkpoints/                         # 모델 가중치
│   ├── offline_bundle.pt                # 배포용 번들 (Stage A+B 통합)
│   ├── offline_bundle.pt.meta.json
│   ├── stage_a/                         # Stage A student 체크포인트
│   └── stage_b/                         # Stage B adapter 체크포인트
└── outputs/                             # 추론 결과
```

```
personalrepo/src/sarcasm_emotion_adapter/    # 핵심 패키지 (모델·데이터·추론)
├── modeling.py                              # StageAStudent, StageBSarcasmAdapter
├── dataio.py                                # load_dataset_frame, write_dataframe
├── labels.py                                # KOTE 44라벨 맵, 소분류→대분류 매핑
├── offline.py                               # 번들 export·로드·추론 (OfflineSarcasmEmotionPredictor)
└── __init__.py                              # 위 4개 모듈 re-export
```

```
personalrepo/src/emotion_classifier/         # 단계별 원본 스크립트 (파이프라인 내부에서 호출)
│
├── 00_normalize_dataset.py                  # 원본 데이터 → 정규화 parquet (로컬/HF 지원)
├── 01_offline_infer.py                      # 번들 체크포인트로 오프라인 감정 추론
│
├── 10_prepare_stage_a_teacher_targets.py    # HF 댓글 → KOTE 교사 확률 생성 (teacher 단계)
├── 12_train_stage_a.py                      # 교사 확률 → KcELECTRA-base student 지식 증류
├── 13_prepare_stage_b_targets.py            # Stage B gold 데이터 정규화 + 타겟 생성 (prep-b 단계)
├── 14_train_stage_b.py                      # 풍자 감정 어댑터 학습
├── 15_export_offline_bundle.py              # Stage A/B 체크포인트 → offline_bundle.pt 패키징
│
├── 20_attach_major_labels.py                # 소분류 감정 컬럼 → 대분류 컬럼 추가 (후처리)
├── 21_merge_gold_annotations.py             # Stage B 후보 + 어노테이션 CSV → gold 레이블 병합
│
└── predict_emotion.py                       # 체크포인트 직접 지정 추론 (번들 미사용, 레거시)
```

### 스크립트별 주요 인수

| 스크립트 | 필수 인수 | 주요 선택 인수 |
|---|---|---|
| `00_normalize_dataset.py` | `--output`, (`--input` 또는 `--hf-source`) | `--max-rows`, `--seed`, HF 옵션 |
| `01_offline_infer.py` | `--output`, (`--bundle-checkpoint` 또는 `--bundle-dir`), (`--input` 또는 `--input-json` 또는 `--comment`) | `--top-k`, `--batch-size`, `--local-files-only` |
| `10_prepare_stage_a_teacher_targets.py` | `--output` | `--teacher-model`, `--years`, HF 옵션 |
| `12_train_stage_a.py` | `--output-dir`, (`--input` 또는 `--hf-source`) | `--student-model`, `--epochs`, `--lr`, `--batch-size`, `--val-size` |
| `13_prepare_stage_b_targets.py` | `--label-map`, `--output`, (`--input` 또는 `--hf-source`) | `--drop-positive-without-target`, `--max-rows` |
| `14_train_stage_b.py` | `--stagea-checkpoint`, `--output-dir`, (`--input` 또는 `--hf-source`) | `--epochs`, `--lr`, `--gate-loss-weight`, `--emotion-loss-weight`, `--identity-loss-weight` |
| `15_export_offline_bundle.py` | `--base-checkpoint`, `--output` | `--stageb-checkpoint`, `--student-model`, `--label-map`, `--major-mapping` |
| `20_attach_major_labels.py` | `--input`, `--output` | `--mapping`, `--columns`, `--list-sep` |
| `21_merge_gold_annotations.py` | `--stageb-input`, `--annotation-input`, `--label-map`, `--output`, `--missing-output` | — |
| `predict_emotion.py` | `--base-checkpoint`, `--label-map`, `--output`, (`--input` 또는 `--comment`) | `--stageb-checkpoint`, `--title`, `--top-k` |

> **참고**: 일반 사용 시에는 원본 스크립트를 직접 호출할 필요 없이 `emotion_classifier_run.sh` 또는 `emotion_classifier_pipeline.py`를 통해 실행한다. 원본 스크립트는 파이프라인이 노출하지 않는 세밀한 인수 제어가 필요할 때 사용한다.

---

## 학습 흐름

기본 시작점은 이미 생성된 Stage A teacher-target 파일(`EC_TEACHER_OUT`)이다.
`teacher` 스크립트는 이 파일이 없을 때만 쓰는 보조 유틸리티다.

```
[teacher]     → teacher_targets.parquet       (보조: teacher target 없을 때만)
 prep-a        → stagea_normalized.parquet     (Stage A 정규화·검증)
 train-a       → stage_a/student_comment_distill.pt
 prep-b        → stageb_normalized.parquet     (Stage B 정규화·검증, EC_STAGEB_INPUT 필수)
 train-b       → stage_b/stageB_adapter_checkpoint.pt
 export        → offline_bundle.pt
```

`prep-a` / `prep-b`는 중간 산출물을 파일로 저장해 학습 전에 데이터를 확인할 수 있게 한다.

---

## 실행 방법

### 전체 실행 (권장)

```bash
cd personalrepo/run

# A+B 전체 학습
export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx
./emotion_classifier_run.sh all full
# prep-a → train-a → prep-b → train-b → export

# Stage A만 학습 (Stage B gold 데이터 없을 때)
./emotion_classifier_run.sh all stagea
# prep-a → train-a → export (base-only 번들)

# Stage B만 학습 (기존 Stage A 재사용)
export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx
./emotion_classifier_run.sh all stageb
# prep-b → train-b → export
```

### 단계별 실행

```bash
# Stage A 정규화 (중간 산출물 확인 가능)
./emotion_classifier_run.sh prep-a

# Stage A 학습
./emotion_classifier_run.sh train-a

# Stage B 정규화 (중간 산출물 확인 가능)
export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx
./emotion_classifier_run.sh prep-b

# Stage B 학습
./emotion_classifier_run.sh train-b

# 번들 내보내기 (Stage B checkpoint 있으면 A+B, 없으면 base-only 자동 감지)
./emotion_classifier_run.sh export
```

### 추론

```bash
# 로컬 파일로 추론
./emotion_classifier_run.sh predict data/comments.json data/results.csv

# HuggingFace 소스로 추론 (output 미지정 시 data/emotion_classifier/outputs/ 자동 저장)
./emotion_classifier_run.sh predict \
    --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020 \
    --output data/results.csv

# 대분류 컬럼 추가 (소분류 → 긍정/부정/중립)
./emotion_classifier_run.sh attach-major data/results.csv data/results_major.csv

# 추론 전체 한 번에 (predict → attach-major)
./emotion_classifier_run.sh all offline data/comments.json data/results.csv
```

### 교사 확률 생성 (보조)

```bash
# teacher-target 파일이 아직 없을 때만 사용
./emotion_classifier_run.sh teacher
```

---

## 환경변수

미지정 시 `config/emotion_classifier_config.py`의 기본값을 사용한다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `EC_DATA_DIR` | `data/emotion_classifier` | 데이터 루트 |
| `EC_MODEL_DIR` | `models/emotion_classifier` | 모델 루트 |
| `EC_TEACHER_OUT` | `$EC_DATA_DIR/teacher_targets.parquet` | teacher 단계 출력 / prep-a 입력 |
| `EC_STAGEA_NORMALIZED` | `$EC_DATA_DIR/stagea_normalized.parquet` | prep-a 출력 / train-a 입력 |
| `EC_STAGEA_DIR` | `$EC_MODEL_DIR/stage_a` | train-a 출력 디렉토리 |
| `EC_STAGEB_INPUT` | (없음, **필수**) | Stage B gold 데이터 (`sarcasm_label` 포함) |
| `EC_STAGEB_NORMALIZED` | `$EC_DATA_DIR/stageb_normalized.parquet` | prep-b 출력 / train-b 입력 |
| `EC_STAGEB_DIR` | `$EC_MODEL_DIR/stage_b` | train-b 출력 디렉토리 |
| `EC_BUNDLE_OUT` | `$EC_MODEL_DIR/offline_bundle.pt` | export 출력 번들 |
| `EC_LABEL_MAP` | 패키지 내장 KOTE 44라벨 | 별도 라벨 순서 사용 시에만 지정 |
| `HF_TOKEN` | `~/.cache/huggingface/token` 자동 로드 | HuggingFace 접근 토큰 |

---

## 단계별 산출물

### teacher (보조)

HuggingFace 댓글 데이터 → KOTE 교사 모델로 soft-label 생성

| 산출물 | 형식 | 설명 |
|---|---|---|
| `teacher_targets.parquet` | parquet | `comment` + `teacher_prob_00_*` ~ `teacher_prob_43_*` 44개 확률 컬럼 |
| `teacher_targets_file_stats.csv` | csv | 소스 파일별 raw/kept 행 수 통계 |
| `teacher_targets_id2label.json` | json | `{index: label_name}` 라벨 맵 |

### prep-a

teacher targets 정규화·검증 → 중간 산출물 저장

| 산출물 | 형식 | 설명 |
|---|---|---|
| `stagea_normalized.parquet` | parquet | 컬럼 정규화 완료, `teacher_prob_*` 유효성 검증 |

### train-a

정규화된 teacher targets → KcELECTRA-base student 지식 증류

| 산출물 | 형식 | 설명 |
|---|---|---|
| `stage_a/student_comment_distill.pt` | pt | best val 성능 기준 state dict |
| `stage_a/student_comment_distill.meta.json` | json | 학습 메타 (모델명·행 수·prob 컬럼·seed) |
| `stage_a/train_history.csv` | csv | epoch별 train_loss / val_loss / top1_match / prob_mae |

### prep-b

Stage B gold 데이터 정규화·검증 → 중간 산출물 저장

| 산출물 | 형식 | 설명 |
|---|---|---|
| `stageb_normalized.parquet` | parquet | 컬럼 정규화 완료, `sarcasm_label` 유효성 검증 |

### train-b

정규화된 gold 데이터 + Stage A checkpoint → 풍자 감정 어댑터 학습

| 산출물 | 형식 | 설명 |
|---|---|---|
| `stage_b/stageB_adapter_checkpoint.pt` | pt | best epoch state dict + label_map + args |
| `stage_b/stageB_adapter_checkpoint.meta.json` | json | 학습 메타 (입력 행 수·best_epoch·student_model) |
| `stage_b/tokenizer/` | 디렉터리 | HuggingFace tokenizer 저장본 |
| `stage_b/train_history.csv` | csv | epoch별 train_loss / sarcasm_f1 / positive_emotion_top1_hit_gain / negative_identity_rate |
| `stage_b/best_val_predictions.csv` | csv | best epoch val set 예측 결과 |

### export

Stage A + Stage B checkpoint → 단일 오프라인 추론 번들

| 산출물 | 형식 | 설명 |
|---|---|---|
| `offline_bundle.pt` | pt | Stage B 있으면 `stagea_stageb`, 없으면 `stagea_only` variant |

### predict

번들 + 입력 → 감정 예측 결과

| 산출물 | 형식 | 설명 |
|---|---|---|
| `prediction_<source>_<date>.json` | json | `base_emotion`, `primary_emotion`, `top_5`, `gate_score`, `title_provided` 포함 |

출력 경로 미지정 시 `data/emotion_classifier/outputs/prediction_<source>_<date>.json`에 자동 저장.

### attach-major

소분류 감정 레이블 → 대분류 컬럼 추가

| 산출물 | 형식 | 설명 |
|---|---|---|
| `results_major.csv` | csv | 기존 컬럼 + `{col}_major` 컬럼 추가 |

---

## 데이터 I/O

### 지원 입력 형식

- 로컬 파일: `csv`, `json`, `parquet`, `xlsx` / `xls`
- HuggingFace 소스: repo root, 특정 subdirectory, 특정 파일 경로

### 컬럼 자동 인식

| 표준 컬럼 | 인식하는 이름들 |
|---|---|
| `comment` | `comment`, `comment_text`, `text`, `content`, `body` |
| `news_title` | `news_title`, `title`, `raw_title`, `headline` |
| `date` | `date`, `news_date`, `article_date`, `published_at`, `created_at` |

정규화 후 표준 컬럼: `candidate_id`, `comment`, `title`, `news_title`, `date`, `news_date`, `dataset_date`, `source_file`

- `comment`는 필수, `title`은 선택 (없으면 빈 문자열)
- `candidate_id`가 없으면 자동 생성

### 추론 입력 예시

```json
{ "comment": "와 정말 잘한다", "title": "같은 실수 반복한 정부", "date": "2026-07-30" }
```

### 추론 출력 예시

```json
{
  "comment": "와 정말 잘한다",
  "title": "같은 실수 반복한 정부",
  "title_provided": true,
  "gate_score": 0.82,
  "base_emotion":    { "label": "기쁨/신남",  "major": "긍정", "score": 0.61 },
  "primary_emotion": { "label": "화남/분노",  "major": "부정", "score": 0.73 },
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
- `teacher_prob_*` 컬럼 44개

### Stage B 학습 입력 최소 요건 (gold 데이터)

- `comment`, `sarcasm_label`, `emotion_labels`

---

## sarcasm_emotion_adapter 패키지

`personalrepo/src/sarcasm_emotion_adapter/`에 위치하는 핵심 패키지. pipeline과 raw 스크립트 모두 이 패키지를 공유한다.

### offline.py

번들 export·로드·추론을 담당하는 파일. 외부에서 직접 쓸 일이 가장 많은 모듈이다.

#### 번들 포맷

`export_offline_bundle()`이 생성하는 `.pt` 파일에 포함되는 키:

| 키 | 설명 |
|---|---|
| `bundle_type` | 고정값 `"sarcasm_emotion_adapter_offline_bundle"` |
| `bundle_version` | 현재 `1` |
| `bundle_variant` | `"stagea_only"` 또는 `"stagea_stageb"` |
| `student_model` | HuggingFace 모델명 (e.g. `"beomi/KcELECTRA-base"`) |
| `id2label` | `{int: str}` 44개 소분류 라벨 맵 |
| `small_to_major` | `{str: str}` 소분류 → 대분류 매핑 |
| `num_labels` | 라벨 수 (44) |
| `max_length` | 토크나이저 max_length |
| `top_k` | 반환할 상위 감정 수 |
| `base_state_dict` | Stage A student state dict |
| `adapter_state_dict` | Stage B adapter state dict (`stagea_only`이면 `None`) |

번들 저장 시 `.pt.meta.json`도 함께 생성된다 (variant·경로·생성 시각 등 경량 메타).

#### 주요 함수 / 클래스

**`export_offline_bundle(...) → Path`**

Stage A/B 체크포인트를 단일 `.pt` 번들로 패키징.

```python
from sarcasm_emotion_adapter.offline import export_offline_bundle

export_offline_bundle(
    bundle_path="models/emotion_classifier/offline_bundle.pt",
    student_model="beomi/KcELECTRA-base",
    base_checkpoint="models/emotion_classifier/stage_a/student_comment_distill.pt",
    label_map_path=...,           # 생략 시 패키지 내장 KOTE 44라벨
    adapter_checkpoint=None,      # None이면 stagea_only 번들
    major_mapping_path=None,
    max_length=192,
    top_k=5,
)
```

---

**`resolve_bundle_checkpoint(...) → Path`**

번들 파일 경로를 자동 탐색. 직접 경로 지정 → 디렉토리 지정 → 기본 탐색 순으로 시도.

```python
resolve_bundle_checkpoint(
    bundle_checkpoint=None,          # 직접 경로 지정 시
    bundle_dir="models/emotion_classifier",
    pattern="offline_bundle*.pt",    # 복수 번들 중 최신 선택
)
```

여러 번들이 있으면 `created_at_utc` 기준 최신 파일을 자동 선택한다.

---

**`read_bundle_metadata(bundle_path) → dict`**

번들 옆에 있는 `.pt.meta.json`을 읽어 메타 정보 반환. 파일 없으면 빈 dict.

---

**`OfflineSarcasmEmotionPredictor`**

번들 기반 추론 클래스. **반드시 `from_bundle()`로 생성**해야 한다 (`__init__` 직접 호출 금지).

```python
from sarcasm_emotion_adapter.offline import OfflineSarcasmEmotionPredictor

predictor = OfflineSarcasmEmotionPredictor.from_bundle(
    bundle_checkpoint="models/emotion_classifier/offline_bundle.pt",
    batch_size=32,
    device=None,          # None이면 CUDA 자동 탐지
    local_files_only=False,
)

rows = [
    {"comment": "와 진짜 잘한다", "title": "정부 또 실수"},
    {"comment": "좋은 소식이네요", "title": ""},
]
df = predictor.predict(rows)  # → pd.DataFrame
```

`predict(rows)`의 입력은 `list[dict]`, 각 dict에 `comment`(필수)와 `title`(선택) 포함.  
반환 DataFrame 컬럼:

| 컬럼 | 설명 |
|---|---|
| `title_provided` | title 입력 유무 (`bool`) |
| `gate_score` | adapter 보정 강도 (0~1); `stagea_only` 번들이면 항상 0 |
| `base_emotion_label_fine` | Stage A base 모델 1위 소분류 |
| `base_emotion_label_major` | Stage A base 모델 1위 대분류 |
| `base_emotion_score` | Stage A 1위 확률 |
| `primary_emotion_label` | 최종 1위 소분류 (Stage B 보정 반영) |
| `primary_emotion_major` | 최종 1위 대분류 |
| `primary_emotion_score` | 최종 1위 확률 |
| `top_5_json` | 상위 k개 감정 JSON 문자열 (`[{label, major, score}]`) |

> `bundle_variant`가 `stagea_only`이면 `adapter_model`이 `None`이고 `gate_score`는 항상 0, `primary_emotion_*`과 `base_emotion_*`이 동일하다.

### 나머지 모듈 요약

| 모듈 | 주요 공개 심볼 |
|---|---|
| `modeling.py` | `StageAStudent`, `StageBSarcasmAdapter`, `FrozenStageAStudent`, `build_context_text`, `load_label_map` |
| `dataio.py` | `load_dataset_frame`, `write_dataframe`, `parse_hf_source` |
| `labels.py` | `KOTE_LABELS`, `load_small_to_major`, `load_major_mapping`, `get_default_label_map_path`, `get_default_major_mapping_path` |

---

## 유의사항

- Stage B 학습은 반드시 Stage A checkpoint 기반으로 한다.
- `EC_STAGEB_INPUT`은 `sarcasm_label`이 포함된 gold 데이터 경로로, `prep-b` / `train-b` 실행 전 반드시 설정해야 한다.
- `export`는 Stage B checkpoint 존재 여부를 자동 감지해 `stagea_only` 또는 `stagea_stageb` 번들을 생성한다.
- `predict`의 `--bundle` 미지정 시 `models/emotion_classifier/`, `data/emotion_classifier/outputs/` 순으로 자동 탐색한다.
- CUDA가 없으면 CPU fallback으로 실행되며 실제 데이터에서는 속도가 느려진다.
- `EC_LABEL_MAP`은 기본 KOTE 44라벨 체계와 다른 라벨 순서를 사용한 경우에만 지정한다.
