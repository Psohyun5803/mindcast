# event_online — 데이터 입출력 명세

## 전체 흐름

```
[HF private 또는 로컬 mysqldump]
        │
      prep  (제목 정제 · kiwi 명사 · 임베딩)
        │
   data/posts_<month>.parquet
   data/emb_<month>.npy
        │
      track  (인과적 이벤트 트래킹)
        │
   outputs/tracks_online_<month>.pkl
   outputs/events_online_<month>.csv
        │
      viz  (타임라인 시각화)
        │
   figures/06_online_dynamics_<month>.png
        │
      export  (HTML 리플레이)
        │
   online_replay.html
```

별도 흐름:
```
[로컬 mysqldump]
      │
   upload  (PII 가드 → HF private 업로드)
   extract (로컬 parquet 캐시)
```

모든 경로는 `data/event_online/` 하위.

---

## 단계별 입출력

### extract *(선택)*

로컬 mysqldump에서 뉴스 테이블만 파싱해 parquet 캐시로 저장. 자주 사용할 때 속도 향상용.

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `$MINDCAST_DUMP` (환경변수) | sql.gz | 로컬 mysqldump 압축 파일 |
| **출력** | `data/video_video.parquet` | parquet | `id`, `video_id`, `title`, `description`, `tags`, `like_count`, `comment_count`, `view_count`, `created_at`, `channel_id` 등 |
| **출력** | `data/video_channel.parquet` | parquet | `channel_id`, `title`, `subscriber_count`, `video_count`, `is_main`, `region` 등 |
| **출력** *(--with-comments)* | `data/video_comment.parquet` | parquet | `comment_id`, `author`(SHA-256 해시), `text`, `like_count`, `created_at`, `video_id` |

---

### upload

로컬 덤프에서 뉴스 3개 테이블 추출 → PII 가드 → HuggingFace private 데이터셋 업로드.

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `$MINDCAST_DUMP` | sql.gz | 로컬 mysqldump |
| **중간** | `hf_staging/video_video.parquet` | parquet | PII 검증 통과 후 스테이징 |
| **중간** | `hf_staging/video_channel.parquet` | parquet | — |
| **중간** | `hf_staging/video_comment.parquet` | parquet | `author` = SHA-256(16 hex) 가명화 |
| **출력** | HF `$MINDCAST_HF_REPO` | HF dataset | `video_video`, `video_channel`, `video_comment` 3개 파일 |

PII 가드 검사 항목:
- `password`, `pw_display`, `userid`, `email`, `session_data`, `token`, `jti`, `first_name`, `last_name` 컬럼 발견 시 즉시 중단
- 댓글 `author` 컬럼이 전부 16자리 hex 해시인지 검증

---

### prep

뉴스 제목 정제·키프레이즈 추출·임베딩 생성.

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** (hf) | HF `MindCastSogang/mindcast-news-events` `video_video.parquet` | parquet | `title`, `created_at`, `channel_id`, `comment_count` |
| **입력** (local) | `data/video_video.parquet` (extract 결과) | parquet | 동일 |
| **출력** | `data/posts_<month>.parquet` | parquet | `post_id`, `date`, `title_clean`, `keywords` (list), `hashtags` (list), `comment_count` |
| **출력** | `data/emb_<month>.npy` | npy | shape `(N, 768)` float32, L2 정규화된 ko-sroberta 임베딩 |

`posts_<month>.parquet` 주요 컬럼:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `post_id` | int | 원본 `video_video.id` |
| `date` | date | 뉴스 게시일 |
| `title_clean` | str | 괄호·날짜·채널명 제거 후 제목 |
| `keywords` | list[str] | kiwi 추출 명사 (NNP/NNG ≥ 2글자) |
| `hashtags` | list[str] | 보일러플레이트 필터 적용 해시태그 |
| `comment_count` | int | 댓글 수 |

---

### track

인과적(causal) 온라인 이벤트 트래킹. 미래 데이터 없이 하루씩 처리.

| 구분 | 경로 | 형식 | 스키마 / 내용 |
|---|---|---|---|
| **입력** | `data/posts_<month>.parquet` | parquet | prep 출력 |
| **입력** | `data/emb_<month>.npy` | npy | prep 출력 임베딩 |
| **출력** | `outputs/tracks_online_<month>.pkl` | pkl | `OnlineTracker` 객체 전체 상태 (트랙 딕셔너리, 라이프사이클 등) |
| **출력** | `outputs/events_online_<month>.csv` | csv | 확정 이벤트 목록 |

`events_online_<month>.csv` 주요 컬럼:

| 컬럼 | 설명 |
|---|---|
| `track_id` | 이벤트 고유 ID |
| `birth_date` | 최초 발견일 |
| `confirm_date` | 확정일 (2일 지속 or 댓글 3,000건 기준) |
| `status` | `active` / `dormant` / `dead` / `revived` |
| `n_posts` | 누적 포스트 수 |
| `total_comments` | 누적 댓글 수 |
| `top_keywords` | 대표 키워드 |

트랙 생명주기: `provisional` → `confirmed` → `active` → `dormant` (3일 미활동) → `dead` (7일) / `revived`

---

### viz

일별 birth/revival 타임라인 시각화.

| 구분 | 경로 | 형식 | 내용 |
|---|---|---|---|
| **입력** | `outputs/tracks_online_<month>.pkl` | pkl | track 객체 |
| **출력** | `figures/06_online_dynamics_<month>.png` | png | 일별 신규(birth)/부활(revival) 막대 + 누적 이벤트 수 라인 |

---

### export

모든 월의 데이터를 결합해 자립형 HTML 리플레이 생성.

| 구분 | 경로 | 형식 | 내용 |
|---|---|---|---|
| **입력** | `outputs/tracks_online_<month>.pkl` (각 월) | pkl | 트랙 상태 |
| **입력** | `data/posts_<month>.parquet` (각 월) | parquet | 뉴스 제목·날짜 |
| **입력** | `data/emb_<month>.npy` (각 월) | npy | 임베딩 (응집도 계산용) |
| **출력** | `online_replay.html` | html | 외부 의존 없는 자립형 HTML |

HTML 내용:
- 일별 이벤트 상태 (new / active / decaying / dormant / revived) 재현
- 이벤트별 응집도(cohesion) 점수 — 낮으면 `기타(저응집)` 분류
- AND 라벨: 멤버의 ≥30%에서 등장한 엔티티
- 뉴스 제목·댓글량 임베딩 시각화 (최대 20개/이벤트/일)

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MINDCAST_HF_REPO` | `MindCastSogang/mindcast-news-events` | HF 업로드 대상 레포 |
| `MINDCAST_DUMP` | (없음) | 로컬 mysqldump 경로 (source=local 또는 upload 시 필수) |
| `MINDCAST_DEVICE` | PyTorch 자동 탐지 | 임베딩 디바이스 강제 지정 |
| `HF_TOKEN` | `~/.cache/huggingface/token` | HF 인증 토큰 |
