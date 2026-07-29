# Data I/O

모든 명령은 아래 경로에서 실행한다.

```bash
cd /home/yein38/mindcastlib_trainer/emotion_classifier
```

## 1. Raw 입력 데이터

이 패키지는 raw 입력 형식이 다소 유동적이어도, 내부에서 표준 컬럼으로 정규화한 뒤 처리한다.

지원 입력 소스:

- 로컬 `csv`
- 로컬 `json`
- 로컬 `parquet`
- 로컬 `xlsx`, `xls`
- Hugging Face dataset source

예:

- `MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1`
- `MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020/01/01-10/news_comments.json`

즉, Hugging Face 쪽은 다음 셋 다 가능하다.

- repo root
- 특정 subdirectory
- 특정 JSON file

## 2. Raw 입력에서 우선적으로 찾는 컬럼

### comment 계열

- `comment`
- `comment_text`
- `text`
- `content`
- `body`

### title 계열

- `news_title`
- `title`
- `raw_title`
- `headline`

### date 계열

- `date`
- `news_date`
- `article_date`
- `published_at`
- `created_at`

## 3. 정규화 후 표준 컬럼

`run/00_offline_normalize_data.sh` 또는 `run/10_online_normalize_data.sh`를 거치면 아래 표준 컬럼 중심으로 맞춰진다.

- `candidate_id`
- `comment`
- `title`
- `news_title`
- `date`
- `news_date`
- `dataset_date`
- `source_file`

추가로 원본에 있던 다른 컬럼은 가능한 범위에서 보존된다.

중요 규칙:

- `comment`는 반드시 있어야 한다.
- `title`이 없어도 된다.
- `title`이 없으면 빈 문자열로 채워진다.
- `candidate_id`가 없으면 자동 생성된다.

## 4. 오프라인 추론 입력 형식

### 파일 입력

`run/01_offline_infer.sh --input ...` 에 넣는 파일은 최소한 `comment` 컬럼이 있어야 한다.

권장 컬럼:

- `comment`
- `title`
- `date`
- `news_date`
- `dataset_date`
- `source_file`
- `candidate_id`

### JSON 단건 입력

```json
{
  "comment": "와 정말 잘한다",
  "title": "같은 실수 반복한 정부",
  "date": "2026-07-29"
}
```

### JSON 다건 입력

```json
[
  {
    "comment": "와 정말 잘한다",
    "title": "같은 실수 반복한 정부"
  },
  {
    "comment": "이건 좀 심하네"
  }
]
```

또는:

```json
{
  "items": [
    {
      "comment": "와 정말 잘한다",
      "title": "같은 실수 반복한 정부"
    }
  ]
}
```

## 5. 오프라인 추론 출력 형식

출력은 JSON, CSV, Parquet, XLSX로 저장할 수 있지만, 배포 관점에서는 JSON 사용을 권장한다.

JSON 1건 예시:

```json
{
  "date": "2026-07-29",
  "title": "같은 실수 반복한 정부",
  "comment": "와 정말 잘한다",
  "title_provided": true,
  "gate_score": 0.82,
  "base_emotion": {
    "label": "기쁨/신남",
    "major": "긍정",
    "score": 0.61
  },
  "primary_emotion": {
    "label": "화남/분노",
    "major": "부정",
    "score": 0.73
  },
  "top_5": [
    {
      "label": "화남/분노",
      "major": "부정",
      "score": 0.73
    },
    {
      "label": "어이없음",
      "major": "부정",
      "score": 0.68
    }
  ]
}
```

주요 필드 의미:

- `title_provided`: title 입력 유무
- `gate_score`: adapter 쪽 보정 강도 신호
- `base_emotion`: comment-only base model의 1위 감정
- `primary_emotion`: 최종 보정 후 1위 감정
- `top_5`: 최종 상위 5개 감정과 점수

## 6. 온라인 학습 입력 형식

### Stage A 입력

Stage A의 기본 입력은 이미 만들어진 teacher-target dataset이다. 최소한 아래가 있어야 한다.

- `comment`
- `teacher_prob_*` columns

원본 컬럼명이 달라도 정규화 단계에서 `comment`는 표준화된다. teacher target 생성 자체는 기본 학습 흐름 바깥 단계다.

### Stage B 입력

온라인에서도 Stage B raw input은 먼저 정규화하는 것을 기본 흐름으로 한다.

gold Stage B 최소 필드:

- `comment`
- `sarcasm_label`
- `emotion_labels`

권장 추가 필드:

- `title`
- `date`
- `news_date`
- `dataset_date`

Stage B는 gold 데이터만 가정한다. 즉 최소한 아래가 필요하다.

- `comment`
- `sarcasm_label`
- `emotion_labels`

## 7. 온라인 학습 산출물

- `outputs/stageB_normalized_<source>_<YYYYMMDD>.parquet`
- `outputs/pt/stageA_comment_distill_<source>_<YYYYMMDD>/`
- `outputs/pt/stageB_adapter_<source>_<YYYYMMDD>/`
- `outputs/pt/offline_bundle_stagea_only_<bundle-source>_<YYYYMMDD>.pt` 또는 `outputs/pt/offline_bundle_stagea_stageb_<bundle-source>_<YYYYMMDD>.pt`
- `outputs/pt/offline_bundle_stagea_only_<bundle-source>_<YYYYMMDD>.pt.meta.json` 또는 `outputs/pt/offline_bundle_stagea_stageb_<bundle-source>_<YYYYMMDD>.pt.meta.json`
- 기본 `<bundle-source>`는 사용된 Stage A / Stage B checkpoint 디렉터리명에서 생성된다.


## 8. 기본 라벨 맵

- 기본 fine-label 순서는 `assets/kote_id2label.json`을 사용한다.
- 현재 기본 KOTE 44 라벨 체계를 쓸 때는 toy output의 id2label 파일이 아니라 이 asset 파일을 기준으로 보면 된다.
- 특별히 다른 라벨 순서를 쓴 실험만 별도 `--label-map`을 넘기면 된다.
