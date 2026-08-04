# personalrepo — 프로젝트 전체 개요

한국 뉴스 댓글 데이터를 기반으로 한 세 개의 독립적인 분석 모듈로 구성됩니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│  emotion_classifier   한국어 댓글 감정 분류 (2-stage 지식 증류)          │
│  → 뉴스 댓글의 44개 소분류 감정 확률 + 풍자 감지 보정                     │
├─────────────────────────────────────────────────────────────────────┤
│  suicide_predict      자살 관련 지표 예측 (일별 + 월별)                   │
│  → 상담전화 건수 다음날 예측 / 월별 자살 사망자수 예측                      │
├─────────────────────────────────────────────────────────────────────┤
│  event_online         뉴스 이벤트 온라인 트래커                           │
│  → 미래 데이터 없이 매일 인과적으로 사건을 발견·추적                         │
└─────────────────────────────────────────────────────────────────────┘
```

세 모듈은 **완전히 독립적**으로 실행됩니다.  
공통 데이터 소스: `MindCastSogang/Youtube_news_preprocessed_data` (HuggingFace)

---

## 폴더 구조

```
root/
├── config/                              # 프로젝트별 설정 파일
│   ├── emotion_classifier_config.py
│   ├── event_online_config.py
│   └── suicide_config.py
│
├── src/                                 # 소스 코드
│   ├── emotion_classifier_utils.py      # 감정 분류기 학습·추론 유틸
│   ├── event_online_utils.py            # 이벤트 트래커 전체 유틸
│   ├── suicide_utils.py                 # 자살 예측기 전처리·모델·학습
│   ├── emotion_classifier_raw/              # 단계별 스크립트 + sarcasm_emotion_adapter 패키지
│   │   ├── sarcasm_emotion_adapter/     # 핵심 패키지 (modeling/dataio/labels/offline)
│   │   ├── 00_normalize_dataset.py
│   │   ├── 10_prepare_stage_a_teacher_targets.py
│   │   ├── 12_train_stage_a.py  ...
│   │   └── predict_emotion.py
│   └── __pycache__/
│
├── pipeline/                            # CLI 진입점 (실제 실행 대상)
│   ├── emotion_classifier_pipeline.py
│   ├── event_online_pipeline.py
│   └── suicide_pipeline.py
│
├── run/                                 # 쉘 실행 스크립트 (pipeline 래퍼)
│   ├── emotion_classifier_run.sh
│   ├── event_online_run.sh
│   └── suicide_run.sh
│
├── data/                                # 프로젝트별 데이터·산출물
│   ├── emotion_classifier/
│   │   ├── assets/                      # kote_id2label.json, mapping_ver1.json
│   │   ├── cache/                       # 학습 중간 산출물 (teacher_targets, stagea_normalized 등)
│   │   ├── checkpoints/                 # 모델 가중치 (offline_bundle.pt, stage_a/, stage_b/)
│   │   └── outputs/                     # 추론 결과
│   │
│   ├── event_online/
│   │   ├── data/                        # prep 출력 (posts_*.parquet, emb_*.npy)
│   │   ├── outputs/                     # track 출력 (tracks_online_*.pkl, events_online_*.csv)
│   │   ├── figures/                     # viz 출력 (online_dynamics_*.png)
│   │   ├── hf_staging/                  # upload 전 임시 parquet (업로드 후 삭제 가능)
│   │   └── online_replay.html           # export 출력
│   │
│   └── suicide_predict/
│       ├── cache/                       # 전처리 캐시 (raw/emotion/topic)
│       ├── outputs/                     # 학습 산출물 (checkpoints/metrics/predictions)
│       ├── deliverable/                 # 월별 예측기 최종 산출물
│       └── prediction/                  # inference 결과
│           ├── main/                    # 일별 inference (main-infer)
│           └── monthly/                 # 월별 inference (monthly-infer)
│
├── help/                                # 프로젝트 문서
│   ├── overview.md                      # 이 파일
│   ├── emotion_classifier.md
│   ├── event_online.md
│   ├── suicide_predict.md
│   ├── dataio/                          # 데이터 입출력 명세
│   │   ├── emotion_classifier_dataio.md
│   │   ├── event_online_dataio.md
│   │   └── suicide_predict_dataio.md
│   └── logic/
│       └── suicide_predict_logic.pdf
│
├── .env.example                         # 환경변수 템플릿
├── .gitignore
└── requirements.txt
```

---

## 환경 설정

### 설치

```bash
# PyTorch (CUDA 버전에 맞춰)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 나머지 의존성
pip install -r requirements.txt
```

### 환경변수

```bash
cp .env.example .env
# .env 파일에 값 입력
```

| 변수 | 필수 | 설명 |
|------|------|------|
| `HF_TOKEN` | O | HuggingFace 접근 토큰 (`MindCastSogang` 조직 권한 필요) |
| `MINDCAST_DUMP` | local 소스 시 | 로컬 mysqldump 경로 (event_online) |
| `EC_STAGEB_INPUT` | Stage B 학습 시 | gold 데이터 경로 (emotion_classifier) |
| `MINDCAST_FONT` | X | 한글 폰트 경로 (event_online viz, 미지정 시 자동 탐색) |

---

## 모듈별 빠른 실행

### emotion_classifier

```bash
cd run

# Stage A+B 전체 학습
export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx
./emotion_classifier_run.sh all full

# Stage A만 (gold 데이터 없을 때)
./emotion_classifier_run.sh all stagea

# 추론
./emotion_classifier_run.sh predict data/comments.json data/results.csv
```

### suicide_predict

```bash
cd run

# 1. 전처리 (HF → KOTE → 감정/토픽 집계)
./suicide_run.sh -g 2 preprocess

# 2. 일별 예측기 Ablation A~F
./suicide_run.sh -g 2 ablation

# 3. 메인 모델 확정
./suicide_run.sh main

# 4. 월별 예측기
./suicide_run.sh monthly

# inference
./suicide_run.sh main-infer 2024-01-01 2024-12-31
./suicide_run.sh monthly-infer 2024-03

# 전체 한번에
./suicide_run.sh -g 2 all
```

### event_online

```bash
cd run

# HF 소스 기준 (월 단위)
./event_online_run.sh -g 0 prep  2025-09 hf
./event_online_run.sh         track 2025-09
./event_online_run.sh         viz   2025-09
./event_online_run.sh         export

# 전체 한번에
./event_online_run.sh -g 0 all 2025-09 hf

# HF 업로드 (선택)
./event_online_run.sh upload --dry-run   # PII 가드 확인
./event_online_run.sh upload
```
---

## 각 모듈 상세 문서

| 모듈 | 개요 문서 | 데이터 I/O 명세 |
|------|-----------|----------------|
| emotion_classifier | [emotion_classifier.md](emotion_classifier.md) | [dataio/emotion_classifier_dataio.md](dataio/emotion_classifier_dataio.md) |
| suicide_predict | [suicide_predict.md](suicide_predict.md) | [dataio/suicide_predict_dataio.md](dataio/suicide_predict_dataio.md) |
| event_online | [event_online.md](event_online.md) | [dataio/event_online_dataio.md](dataio/event_online_dataio.md) |
