# Offline Run Scripts

모든 명령은 먼저 아래처럼 프로젝트 루트로 이동한 뒤 실행한다.

```bash
cd /home/yein38/mindcastlib_trainer/emotion_classifier
```

## 목적

오프라인 모드는 학습 없이 추론만 수행하는 배포용 흐름이다. 입력 raw data를 정규화한 뒤, `title` 유무에 따라 base branch 또는 base+adapter branch로 보내고, 최종 결과를 같은 JSON 형식으로 반환한다.

## 스크립트 순서

- `run/00_offline_normalize_data.sh`
- `run/01_offline_infer.sh`
- `run/02_offline_pipeline.sh`

## PT/BUNDLE 외부 인자

오프라인 추론에서는 코드 안의 경로를 수정할 필요 없이, 실행할 때 bundle 경로만 넘기면 된다.

- 특정 bundle 하나를 정확히 지정: `--bundle-checkpoint /absolute/path/to/offline_bundle....pt`
- 특정 디렉터리에서 최신 bundle 자동 선택: `--bundle-dir /absolute/path/to/outputs/pt`
- 파일명 패턴까지 제한: `--bundle-pattern 'offline_bundle_stagea_stageb*.pt'`

가장 단순한 방식은 보통 `--bundle-checkpoint` 하나만 넘기는 것이다.

## 1. 정규화만 수행

```bash
./run/00_offline_normalize_data.sh   --input /path/to/raw.json
```

Hugging Face source도 가능하다.

```bash
./run/00_offline_normalize_data.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1
```

```bash
./run/00_offline_normalize_data.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020
```

```bash
./run/00_offline_normalize_data.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020/01/01-10/news_comments.json
```

출력 파일명을 직접 정하려면:

```bash
./run/00_offline_normalize_data.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020   --output outputs/my_2020_normalized.parquet
```

출력 파일명을 직접 지정하지 않으면 아래 규칙으로 자동 저장된다.

- `outputs/normalized_<source>_<YYYYMMDD>.parquet`
- 예: `outputs/normalized_hf_2020_20260729.parquet`

## 2. 추론만 수행

정규화된 파일 입력:

```bash
./run/01_offline_infer.sh   --input outputs/my_2020_normalized.parquet
```

단건 직접 입력:

```bash
./run/01_offline_infer.sh   --comment "와 정말 대단하네요"   --title "논란 끝에 또 같은 결정"
```

동작 규칙:

- `title`이 없으면 base model만 사용한다.
- `title`이 있으면 adapter branch를 함께 사용한다.
- 출력 형식은 두 경우 모두 동일하다.

bundle 탐색 규칙:

1. `--bundle-checkpoint`가 있으면 그 파일을 사용한다.
2. 없고 `--bundle-dir`가 있으면 그 디렉터리에서 최신 bundle을 찾는다.
3. 둘 다 없으면 기본적으로 `outputs/pt/`, `outputs/` 순서로 본다.

번들을 직접 지정하는 예시:

checkpoint 파일을 바로 지정:

```bash
./run/01_offline_infer.sh   --input outputs/my_2020_normalized.parquet   --bundle-checkpoint /home/yein38/mindcastlib_trainer/emotion_classifier/outputs/pt/offline_bundle_stagea_stageb_stageb_gold_20260729.pt
```

bundle 디렉터리만 지정하고 그 안에서 최신 bundle을 자동 선택:

```bash
./run/01_offline_infer.sh   --input outputs/my_2020_normalized.parquet   --bundle-dir /home/yein38/mindcastlib_trainer/emotion_classifier/outputs/pt
```

패턴까지 직접 지정하려면:

```bash
./run/01_offline_infer.sh   --input outputs/my_2020_normalized.parquet   --bundle-dir /home/yein38/mindcastlib_trainer/emotion_classifier/outputs/pt   --bundle-pattern 'offline_bundle*.pt'
```

출력 파일명을 직접 정하려면:

```bash
./run/01_offline_infer.sh   --input outputs/my_2020_normalized.parquet   --output outputs/my_2020_prediction.json
```

출력 파일명을 직접 지정하지 않으면 아래 규칙으로 자동 저장된다.

- `outputs/prediction_<source>_<YYYYMMDD>.json`
- 예: `outputs/prediction_my_2020_normalized_20260729.json`

## 3. 정규화 + 추론을 한 번에 수행

```bash
./run/02_offline_pipeline.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020
```

repo root도 바로 넣을 수 있다.

```bash
./run/02_offline_pipeline.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1
```

내부 순서:

1. `run/00_offline_normalize_data.sh`
2. `run/01_offline_infer.sh`

출력 파일명을 직접 정하려면:

```bash
./run/02_offline_pipeline.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020   --normalized-output outputs/news2020_norm.parquet   --prediction-output outputs/news2020_pred.json
```

추론에 사용할 bundle도 같이 지정하려면:

```bash
./run/02_offline_pipeline.sh   --hf-source MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020   --normalized-output outputs/news2020_norm.parquet   --prediction-output outputs/news2020_pred.json   --bundle-checkpoint /home/yein38/mindcastlib_trainer/emotion_classifier/outputs/pt/offline_bundle_stagea_stageb_stageb_gold_20260729.pt
```

출력 파일명을 직접 지정하지 않으면 아래 규칙으로 자동 저장된다.

- normalized: `outputs/normalized_<source>_<YYYYMMDD>.parquet`
- prediction: `outputs/prediction_<source>_<YYYYMMDD>.json`

## 예외 케이스

- raw 입력이 로컬 파일이면 `--input` 사용
- raw 입력이 Hugging Face source면 `--hf-source` 사용
- `title`이 없어도 오류가 아니며 base branch로 처리됨
- base-only bundle은 보통 `offline_bundle_stagea_only_...pt`, adapter 포함 bundle은 `offline_bundle_stagea_stageb_...pt` 이름으로 저장됨
- bundle 경로를 지정하지 않으면 기본 search dir를 사용함
- CUDA가 없으면 CPU fallback으로 실행됨
