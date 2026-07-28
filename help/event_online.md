# event_online — 뉴스 이벤트 온라인 트래커

## 개요

뉴스 이벤트(사회적 사건)를 **미래 데이터 없이 하루씩 인과적으로** 추적하는 online 트래커.  
발견(discovery)과 배정(assignment)을 분리하며, 매일 스트리밍으로만 처리합니다.

```
┌──────────────────────────────────────────────────────────────────┐
│  데이터 입력                                                        │
│  source="hf"    → HuggingFace private 데이터셋 (권장)              │
│  source="local" → 로컬 mysqldump (.sql.gz) 직접 파싱               │
│  → 어느 쪽을 써도 이후 처리는 완전히 동일한 코드로 흐름               │
├──────────────────────────────────────────────────────────────────┤
│  파이프라인                                                          │
│  prep → track → viz → export                                      │
│                                                                    │
│  prep   : 제목 정제·kiwi 명사·ko-sroberta 임베딩 (GPU 권장)         │
│  track  : 인과적 트래킹 (HDBSCAN + EMA, 미래 데이터 없음)            │
│  viz    : 일별 birth/revival 타임라인 그림                           │
│  export : 자립형 일별 리플레이 HTML                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 파일 구조

```
mindcast/
├── config/event_online_config.py      # 모든 상수·환경변수·파라미터
├── utils/event_online_utils.py        # 전체 유틸리티 함수 (6개 섹션)
├── pipeline/event_online_pipeline.py  # 실행 파이프라인 (CLI)
└── run/event_online_run.sh            # 쉘 실행 스크립트

mindcast_online/                       # 원본 프로젝트 (데이터·산출물)
├── data/
│   ├── posts_<month>.parquet          # 정제된 영상 메타·키프레이즈
│   └── emb_<month>.npy               # ko-sroberta 임베딩 (N×768)
├── outputs/
│   ├── tracks_online[_<month>].pkl   # 트랙 상태 전체
│   └── events_online[_<month>].csv   # 이벤트 요약 표
├── figures/
│   └── 06_online_dynamics[_<month>].png
├── hf_staging/                        # HF 업로드 전 임시 parquet
└── online_replay.html                 # 일별 리플레이 HTML
```

---

## 실행 순서

```bash
cd mindcast/run

# ── 1) 표현 생성 (GPU 필요) ────────────────────────────────────
./event_online_run.sh -g 0 prep 2025-09 hf      # HuggingFace 소스
./event_online_run.sh -g 0 prep 2025-09 local   # 로컬 덤프 소스

# ── 2) 온라인 트래킹 ───────────────────────────────────────────
./event_online_run.sh track 2025-09

# ── 3) 시각화 ──────────────────────────────────────────────────
./event_online_run.sh viz 2025-09

# ── 4) HTML 리플레이 ───────────────────────────────────────────
./event_online_run.sh export                        # config MONTHS 전체
./event_online_run.sh export 2025-09,2025-10,2025-11

# ── 전체 한번에 (prep→track→viz→export) ────────────────────────
./event_online_run.sh -g 0 all 2025-09 hf

# ── 선택: 로컬 덤프 → parquet 캐시 ────────────────────────────
./event_online_run.sh extract
./event_online_run.sh extract --with-comments

# ── 선택: HF 업로드 ────────────────────────────────────────────
./event_online_run.sh upload --dry-run   # PII 가드만
./event_online_run.sh upload             # 실제 업로드
```

또는 Python 직접 실행:
```bash
cd mindcast/pipeline
python event_online_pipeline.py --run prep  --month 2025-09 --source hf --gpu 0
python event_online_pipeline.py --run track --month 2025-09
python event_online_pipeline.py --run viz   --month 2025-09
python event_online_pipeline.py --run export
python event_online_pipeline.py --run all   --month 2025-09 --source hf --gpu 0
```

---

## 환경변수

| 변수 | 기본값 | 용도 |
|------|--------|------|
| `MINDCAST_HF_REPO` | `MindCastSogang/mindcast-news-events` | HF 데이터셋 repo |
| `MINDCAST_DUMP` | (필수) | `source=local` 시 sql.gz 경로 |
| `MINDCAST_FONT` | (자동탐색) | viz 한글 폰트 .ttf/.ttc |
| `MINDCAST_DEVICE` | (자동: cuda>cpu) | prep 임베딩 디바이스 강제 |
| `HF_TOKEN` | — | private 데이터셋 접근 토큰 |

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
export MINDCAST_HF_REPO=MindCastSogang/mindcast-news-events
export MINDCAST_DUMP=/path/to/mindcast-prod-YYYYMMDD.sql.gz   # local 소스 시
```

---

## 파이프라인 단계

### Pipeline 0 — prep (표현 생성)

| 처리 | 설명 |
|------|------|
| clean_title | 괄호·날짜·채널명·뉴스 포맷어 제거 |
| hashtags | 설명란 해시태그 추출 (BOILER 필터) |
| kiwi | 제목에서 NNP/NNG 명사 추출 (≥ 2글자) |
| ko-sroberta | 제목+태그 임베딩 (N×768 float32, L2 정규화) |

입력: `video_video` 테이블 (source=hf 또는 local)  
출력: `data/posts_<month>.parquet`, `data/emb_<month>.npy`

### Pipeline 1 — track (온라인 트래킹)

`OnlineTracker.step(day, rows, embs)` 매일 순서:
```
1. 러닝 IDF 갱신
2. 오늘 포스트 → active/dormant 이벤트에 multi-label 배정 (τ=0.34, K=2, EMA)
3. 미배정 → residual buffer (최근 3일)
4. residual HDBSCAN으로 신규 발견 + backfill
5. 유사 트랙 흡수 (consolidation) + twin 일별 병합
6. 라이프사이클 롤 (active/dormant/dead, 재등장 시 revival)
   신규 → provisional → confirmed (2일 지속 or 3000 댓글)
```

입력: `data/posts_<month>.parquet`, `data/emb_<month>.npy`  
출력: `outputs/tracks_online[_<month>].pkl`, `outputs/events_online[_<month>].csv`

### Pipeline 2 — viz (시각화)

- 일별 birth + revival 막대그래프
- 누적 확정 이벤트 수 꺾은선
- 오프라인 pkl이 있으면 온/오프라인 비교 텍스트 추가

출력: `figures/06_online_dynamics[_<month>].png`

### Pipeline 3 — export (HTML 리플레이)

- 이벤트 응집도(cohesion) 계산 → 기타/저응집 분류
- AND 라벨 생성 (포스트의 FRAC 이상에서 공유된 엔티티)
- 일별 상태 재현 (new/active/decaying/dormant/revived)
- 실제 뉴스 제목·댓글량 임베딩 (자립형 HTML, 외부 의존 없음)

입력: 모든 `MONTHS`의 pkl + posts + emb  
출력: `online_replay.html`

### Pipeline 4 — upload (HF 업로드)

```
1. 로컬 덤프에서 news 3개 테이블 추출 → hf_staging/*.parquet
2. PII 가드: 계정/비밀번호 컬럼 발견 시 중단
             댓글 author가 전부 16-hex 해시인지 검증
3. 가드 통과 후 private HF 데이터셋에 업로드
```

### Pipeline 5 — extract (로컬 캐시)

로컬 덤프를 자주 쓸 때 속도 향상용 parquet 캐시 생성.  
파이프라인 자체는 이 캐시 없이도 동작합니다.

---

## 주요 파라미터 (event_online_config.py)

### 트래킹 파라미터 (P_ONLINE)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `tau` | 0.34 | 배정 임계값 (cos+엔티티 유사도) |
| `K` | 2 | 포스트당 최대 배정 이벤트 수 |
| `ema` | 0.6 | centroid 업데이트 EMA 계수 |
| `residual_window` | 3 | residual 유지 일수 |
| `active_gap` | 3 | active 유지 최대 공백일 |
| `dormant_gap` | 7 | dormant 유지 최대 공백일 |
| `archive_gap` | 30 | 배정 후보 유지 최대 공백일 |
| `confirm_days` | 2 | confirmed 조건: 지속 일수 |
| `confirm_comments` | 3000 | confirmed 조건: 누적 댓글 수 |
| `cons_cos` | 0.72 | twin 병합 cosine 임계값 |
| `merge_thr` | 0.55 | residual 흡수 임계값 |

### 대상 월 (MONTHS)

```python
MONTHS = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-03"]
```

`event_online_config.py`에서 직접 수정하거나 `--months` 인자로 지정.

---

## 검증 결과

| 월 | 확정 이벤트 | 부활 | 포스트 커버리지 | offline top-15 recall |
|----|------------|------|----------------|----------------------|
| 2025-09 | 202 (+3 prov) | 59 | 94.4% | 9/15 |
| 2025-10 | 163 | 59 | 93.8% | 9/15 |
| 2025-11 | 98 | 40 | 96% | 8/15 |
| 2025-12 | 84 | 27 | 97% | 12/15 |

---

## 트러블슈팅

### HuggingFace 404 (레포 없음 / 권한 없음)

```
RepositoryNotFoundError: 404 Client Error.
```

**원인:** 레포가 private이거나 아직 생성되지 않음. 유효한 토큰이어도 접근 권한이 없으면 404.

```bash
# 1. 토큰 설정 (현재 셸에 반드시)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# 2. 인증 확인
python -c "import huggingface_hub; print(huggingface_hub.whoami()['name'])"

# 3. 레포가 없다면 → local 소스로 먼저 업로드
export MINDCAST_DUMP=/path/to/mindcast-prod-YYYYMMDD.sql.gz
./event_online_run.sh upload --dry-run   # PII 가드 확인
./event_online_run.sh upload             # 업로드 (레포 생성됨)
```

### 한글 폰트 없음 (viz 경고)

```
[viz] warning: no Korean font found — set $MINDCAST_FONT to a .ttf/.ttc
```

```bash
# Noto CJK 설치 (Ubuntu/Debian)
sudo apt install fonts-noto-cjk

# 또는 환경변수로 직접 지정
export MINDCAST_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

### kiwipiepy 설치 오류 (prep)

```bash
pip install kiwipiepy
# 빌드 도구 필요 시:
sudo apt install build-essential
```

### prep 메모리 부족 (임베딩)

`event_online_config.py`에서 `EMBED_BATCH = 128` → 줄이기:
```python
EMBED_BATCH = 32   # 또는 64
```

### export HTML — 특정 월 데이터 없음

```
FileNotFoundError: outputs/tracks_online_2025-10.pkl
```

해당 월 track이 먼저 완료되어야 합니다:
```bash
./event_online_run.sh track 2025-10
```

---

## 주의사항

- `data/` 폴더가 `mindcast_kote/event_tracking/data`로의 심볼릭 링크인 경우 → 독립 실행 시 실제 디렉토리로 교체 후 prep부터 재실행
- `outputs/tracks_ml*.pkl` (오프라인 결과) 없어도 동작 — 해당 recall 출력 줄만 생략됨
- 원본 sql.gz는 실제 개인정보 포함 → **공유·커밋·업로드 절대 금지**
- 댓글 author는 업로드 전 SHA-256(16 hex) 가명화 필수 (`upload` 파이프라인이 자동 검증)
