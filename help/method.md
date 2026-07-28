# method.md — MindCast 코딩 컨벤션

이 문서는 `mindcast/` 폴더의 두 파이프라인(`suicide_predict`, `event_online`)에서 공통으로 적용하는
코드 구조·스타일·네이밍 규칙을 정리합니다.

---

## 1. 폴더 구조

```
mindcast/
├── config/          # 상수·경로·하이퍼파라미터만 선언 (실행 로직 없음)
├── utils/           # 순수 함수·클래스 라이브러리 (CLI 없음)
├── pipeline/        # CLI 진입점 — argparse + run_*() 함수
├── run/             # 쉘 래퍼 스크립트 (.sh)
└── help/            # 사용법·컨벤션 문서 (.md)
```

**규칙:**

- 파일명은 `{프로젝트명}_{역할}.py` 형식:
  `suicide_config.py`, `suicide_utils.py`, `suicide_pipeline.py`
- 각 레이어는 단방향 의존: `pipeline → utils → config` (역방향 금지)
- `config`는 순수 상수만 포함; I/O, 모델 호출, 외부 패키지 import 최소화

---

## 2. sys.path 설정 패턴

config/ · utils/ · pipeline/ 폴더가 분리되어 있어 `import`가 기본으로 안 됨.  
각 파일 최상단 docstring 바로 다음에 아래 블록을 삽입한다.

### utils/ 파일 (config/ 하나만 추가)

```python
"""모듈 docstring."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
```

### pipeline/ 파일 (config/ + utils/ 둘 다 추가)

```python
"""파이프라인 docstring."""

# ── 경로 설정 (config/ + utils/ 폴더를 sys.path에 추가) ─────────────────────
import sys as _sys
from pathlib import Path as _Path
_BASE = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_BASE / "config"))
_sys.path.insert(0, str(_BASE / "utils"))
# ────────────────────────────────────────────────────────────────────────────
```

**왜 `_sys`, `_Path` (언더스코어 prefix)?**  
모듈 네임스페이스를 오염시키지 않기 위해 private 이름으로 선언.  
`from xxx import *` 시 노출되지 않음.

---

## 3. ROOT 경로 계산

데이터는 `mindcast/data/` 아래 프로젝트별로 분리해 저장한다.  
모든 데이터는 파이프라인 실행 시 HuggingFace에서 새로 다운로드된다.

`mindcast/config/` 기준 1단계 위 → `mindcast/` → `data/{project}/`

```python
# config 파일 내부
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "data" / "suicide_predict"
ROOT = Path(__file__).resolve().parents[1] / "data" / "event_online"
```

결과 폴더 구조:
```
mindcast/
└── data/
    ├── suicide_predict/
    │   ├── cache/emotion/
    │   ├── cache/raw/
    │   ├── outputs/
    │   └── deliverable/
    └── event_online/
        ├── data/
        ├── outputs/
        └── figures/
```

ROOT를 config에서만 선언하고 utils · pipeline은 config에서 import해 사용한다.  
폴더를 이동하면 이 한 줄만 수정하면 된다.

---

## 4. 모듈 docstring 형식

파일 최상단 docstring은 **한글 사용법**을 포함한다.

```python
"""전체 실행 파이프라인.

사용법
------
# 단계 설명
python pipeline_name.py --run step_name [--옵션 값]

# 전체 한번에
python pipeline_name.py --run all --month 2025-09
"""
```

utils 파일 docstring은 섹션 목록만 나열:

```python
"""pipeline_name 유틸리티.

섹션
----
1. 공통 기초 함수
2. 데이터 소스
3. 전처리
4. 핵심 알고리즘
5. 시각화
6. 내보내기
"""
```

---

## 5. 파이프라인 섹션 헤더

pipeline 파일 내 각 단계는 80자 박스 형식으로 구분:

```python
# ════════════════════════════════════════════════════════════════════════
# Pipeline N — 한글 단계명 (영문 단계명)
#   · 처리 내용 1줄 설명
#   · 처리 내용 2줄 설명
#   입력: 입력 파일 경로 또는 데이터 설명
#   출력: 출력 파일 경로
# ════════════════════════════════════════════════════════════════════════
```

utils 파일 내 섹션 구분은 짧은 선:

```python
# ── 섹션명 ────────────────────────────────────────────────────────────────
```

---

## 6. 로깅 함수

pipeline 파일에 공통으로 정의하는 4개 함수:

```python
def _c(code, s): return f"\033[{code}m{s}\033[0m"
def log(s):  print(_c("36", f"[{_ts()}] {s}"))    # 파란색: 진행 메시지
def ok(s):   print(_c("32", f"[{_ts()}] ✓ {s}"))   # 초록색: 완료
def die(s):  print(_c("31", f"[ERROR]  {s}"), file=sys.stderr); sys.exit(1)  # 빨간색: 오류
def _ts():   return __import__("datetime").datetime.now().strftime("%H:%M:%S")
```

utils에서는 `print()`를 직접 쓰지 않고 caller(pipeline)가 로깅을 담당.

---

## 7. 환경변수 처리

config 파일에서 `os.environ.get()`으로 읽고 기본값을 함께 선언:

```python
import os
HF_REPO   = os.environ.get("MINDCAST_HF_REPO", "MindCastSogang/mindcast-news-events")
DUMP_PATH = os.environ.get("MINDCAST_DUMP", "")
FONT_PATH = os.environ.get("MINDCAST_FONT", "")  # 빈 문자열이면 자동 탐색
```

GPU 설정은 pipeline에서 argparse 후 환경변수로 전달:

```python
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MINDCAST_DEVICE"] = "cuda"
```

---

## 8. argparse 패턴

모든 pipeline은 `--run {step}` 방식 단일 진입점:

```python
def main():
    p = argparse.ArgumentParser(description="파이프라인 이름")
    p.add_argument("--run", required=True,
                   choices=["step1", "step2", "all"],
                   help="실행할 단계")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--month", default=None, help="YYYY-MM")
    args = p.parse_args()

    if args.run == "step1": run_step1(...)
    elif args.run == "step2": run_step2(...)
    elif args.run == "all": run_all(...)

if __name__ == "__main__":
    main()
```

---

## 9. 쉘 스크립트 컨벤션

```bash
#!/usr/bin/env bash
set -euo pipefail
```

**필수 헬퍼 함수:**

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/../pipeline/{name}_pipeline.py"

log()  { echo -e "\033[36m[$(date +%H:%M:%S)]\033[0m $*"; }
ok()   { echo -e "\033[32m[$(date +%H:%M:%S)] ✓\033[0m $*"; }
die()  { echo -e "\033[31m[ERROR]\033[0m $*" >&2; exit 1; }
```

**GPU 플래그:**

```bash
GPU=""
while getopts ":g:" opt; do
    case $opt in
        g) GPU="--gpu $OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
```

**단계 함수 패턴:**

```bash
do_step1() {
    log "단계 설명..."
    python "$PIPELINE" --run step1 $GPU "$@"
    ok "step1 완료"
}
```

**실행 분기:**

```bash
COMMAND="${1:-}"
case "$COMMAND" in
    step1)  do_step1 "${@:2}" ;;
    all)    do_all   "${@:2}" ;;
    ""|--help|-h) usage ;;
    *) die "알 수 없는 명령: $COMMAND" ;;
esac
```

---

## 10. import 순서

```python
# 1) stdlib
import argparse, os, sys, time
from pathlib import Path

# 2) 서드파티
import numpy as np
import pandas as pd
import torch

# 3) 로컬 config
from suicide_config import (ROOT, SPLITS, SEEDS)

# 4) 로컬 utils
from suicide_utils import (build_target, run_ablation)
```

같은 그룹 내에서는 알파벳 순 정렬. 그룹 사이에 빈 줄 하나.

---

## 11. 주석 규칙

- 인라인 설명은 **한글**로 작성 (프로젝트 언어가 한국어)
- `# WHY:` 형태로 비자명한 이유만 달기; WHAT 설명 금지
- 긴 상수 블록은 오른쪽 정렬 주석:

```python
TAU          = 0.34   # 배정 임계값 (cos + 엔티티 유사도)
K            = 2      # 포스트당 최대 배정 이벤트 수
EMA          = 0.6    # centroid 업데이트 EMA 계수
```

---

## 12. 파일 이름 규칙 요약

| 레이어 | 파일명 패턴 | 예시 |
|--------|------------|------|
| config | `{project}_config.py` | `suicide_config.py` |
| utils  | `{project}_utils.py`  | `event_online_utils.py` |
| pipeline | `{project}_pipeline.py` | `suicide_pipeline.py` |
| run shell | `{project}_run.sh` | `event_online_run.sh` |
| help doc | `{project}.md` | `event_online.md` |

`project` 이름은 폴더명과 일치:  
`kote_based_suicide_predict` → `suicide`  
`mindcast_online` → `event_online`
