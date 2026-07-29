# Emotion Classifier

이 저장소는 한국어 댓글 감정 분류기를 배포형으로 정리한 패키지다. 기본적으로는 `comment`만으로 감정을 추론하고, `title`이 함께 들어오면 sarcasm-aware adapter를 추가로 적용해 문맥 보정을 시도한다.

모든 예시는 아래 경로에서 실행하는 것을 기준으로 한다.

```bash
cd ./mindcastlib_trainer/emotion_classifier
```

핵심 문서는 4개만 보면 된다.

- 개념과 전체 구조: [docs/01_OVERVIEW.md](docs/01_OVERVIEW.md)
- 오프라인 실행 스크립트: [docs/02_OFFLINE_RUN_SCRIPTS.md](docs/02_OFFLINE_RUN_SCRIPTS.md)
- 온라인 실행 스크립트: [docs/03_ONLINE_RUN_SCRIPTS.md](docs/03_ONLINE_RUN_SCRIPTS.md)
- 데이터 준비와 입출력 형식: [docs/04_DATA_IO.md](docs/04_DATA_IO.md)

## 핵심 동작

- `comment`만 있으면 base emotion model branch로 추론한다.
- `comment + title`이 있으면 base model 위에 adapter branch를 활성화한다.
- 두 branch는 같은 JSON 출력 형식을 사용한다.
- 오프라인 배포에서는 최신 `offline_bundle*.pt`만 교체하면 된다. `offline_bundle_stagea_only_...pt`는 base-only, `offline_bundle_stagea_stageb_...pt`는 adapter 포함 번들이다.
- 온라인 학습은 이미 만들어진 Stage A teacher-target data에서 시작하며, Stage B는 gold 데이터만 가정한다. `A only`, `B only`, `A+B full` 세 모드 모두 bundle export까지 가능하다.
- pt/bundle 경로는 코드 수정 대신 실행 인자로 바꾸는 방식을 기본으로 한다. 예를 들어 `--stagea-checkpoint`, `--bundle-checkpoint`, `--bundle-output`, `--source-tag`로 제어한다.

## 자동 저장 규칙

- 일반 결과 파일은 `outputs/` 아래에 저장된다.
- `.pt` checkpoint와 bundle은 `outputs/pt/` 아래에 저장된다.
- 출력 경로를 생략하면 `<kind>_<source>_<YYYYMMDD>` 형식으로 자동 생성된다. bundle의 `<source>`는 일반 입력 source가 아니라, 기본적으로 사용된 checkpoint 디렉터리명을 조합한 provenance tag다.
- 예: `outputs/normalized_hf_v1_20260729.parquet`
- 예: `outputs/stageB_normalized_stageb_gold_20260729.parquet`
- 예: `outputs/prediction_hf_onefile_normalized_20260729.json`
- 예: `outputs/pt/offline_bundle_stagea_only_from_stagea_comment_distill_stagea_normalized_xxx_20260729_20260729.pt`
- 예: `outputs/pt/offline_bundle_stagea_stageb_from_stagea_comment_distill_stagea_normalized_xxx_20260729_with_stageb_adapter_stageb_normalized_xxx_20260729_20260729.pt`

## 스크립트 이름 규칙

- 오프라인 추론: `00_`, `01_`, `02_`
- 온라인 학습: `10_` ~ `16_`
- 내부 공용 함수: `run/_common.sh`
