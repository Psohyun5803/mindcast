# Overview

## 무엇을 하는 모델인가

이 패키지는 댓글 감정 분류를 기본 목표로 하는 한국어 emotion classifier다. 핵심 아이디어는 다음과 같다.

- `comment`만 보면 충분한 경우에는 base emotion model만 사용한다.
- `title`이 함께 주어져 문맥 충돌이나 sarcasm 가능성이 생기면 adapter가 추가 보정을 수행한다.
- 사용자는 입력에 `title`을 넣을지 말지만 결정하면 되고, 최종 출력 형식은 항상 같다.

즉, 입력 인터페이스는 단순하게 유지하면서 내부적으로는 `comment-only`와 `comment+title` 두 branch를 지원한다.

## 왜 이렇게 나눴는가

- 실제 서비스 입력에는 `title`이 없는 경우가 많다.
- 반대로 뉴스 댓글처럼 `title`이 문맥 역할을 하는 경우도 있다.
- adapter를 base model과 분리해 두면, title이 없는 입력에는 base만 쓰고 title이 있는 입력에만 보정 모듈을 탈부착할 수 있다.
- 나중에 더 좋은 adapter나 bundle이 나오면 `.pt` 파일만 교체해 오프라인 추론 모델을 업데이트할 수 있다.

## 모델 구조

### 오프라인 추론

1. raw input을 표준 형식으로 정규화한다.
2. `title`이 비어 있으면 base model branch로 간다.
3. `title`이 있으면 base + adapter branch로 간다.
4. 두 경우 모두 동일한 JSON 스키마로 결과를 반환한다.

### 온라인 학습

1. Stage A: 이미 준비된 teacher-target data를 정규화한 뒤 comment-only student를 distillation한다.
2. Stage B: sarcasm 라벨과 gold emotion 라벨이 있는 데이터를 정규화하고, 필요하면 prepared target으로 변환한 뒤 adapter를 fine-tune한다.
3. Export: Stage A checkpoint와 Stage B adapter를 묶어 offline bundle `.pt`를 만든다.

## 설계 원칙

- `title`은 optional이다.
- 입력이 달라도 출력 계약은 동일하다.
- 오프라인 모드는 추론 전용이고, 온라인 모드는 학습 가능 모드다.
- `.pt` 산출물은 일반 결과 파일과 분리해 `outputs/pt/`에 저장한다.
- 자동 파일명은 데이터 출처와 생성일을 반영해 재현성을 높인다.

## 현재 배포 관점에서 중요한 점

- 서비스 추론 시에는 보통 `01_offline_infer.sh` 또는 `02_offline_pipeline.sh`만 쓰면 된다.
- 연구/재학습 시에는 `10_`, `12_`, `13_`, `14_`, `15_`, `16_` 중심으로 온라인 스크립트를 사용한다.
- `11_`은 teacher target이 아직 없을 때만 쓰는 보조 유틸리티다.
- 온라인 Stage A/Stage B 입력 모두 오프라인과 동일한 정규화 로직을 먼저 거친다.
- 가장 최신 bundle을 자동으로 고르도록 되어 있지만, 운영에서는 명시적 `--bundle-checkpoint` 지정도 가능하다.
