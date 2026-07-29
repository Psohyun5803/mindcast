# Online Run Scripts

모든 명령은 먼저 아래처럼 프로젝트 루트로 이동한 뒤 실행한다.

```bash
cd /home/yein38/mindcastlib_trainer/emotion_classifier
```

## 목적

온라인 모드는 학습 가능한 모드다. 이 패키지는 teacher target을 직접 만들어내는 단계부터 시작하지 않고, 이미 만들어진 Stage A teacher-target data를 받아 student 학습부터 시작하는 것을 기본 전제로 한다. 그리고 A와 B 모두 raw input은 먼저 정규화한 뒤 다음 단계로 넘긴다.

또한 이제 Stage B는 `gold` 데이터만 있다고 가정한다. legacy/proxy 분기는 기본 흐름에서 제거했다.

## 핵심 시작점

기본 시작점은 아래 2개를 이미 가지고 있는 상태다.

- Stage A teacher-target dataset
- Stage A label map json 또는 기본 asset label map

## 실행 모드

- `--mode full`: A와 B를 모두 새로 학습하고 bundle export
- `--mode stagea`: A만 학습하고 base-only bundle export
- `--mode stageb`: 기존 Stage A checkpoint를 재사용해 B만 학습하고 bundle export
  `--stagea-checkpoint`를 생략하면 `outputs/pt` 아래 최신 `student_comment_distill.pt`를 자동 사용

## PT/BUNDLE 외부 인자

온라인 모드도 마찬가지로, 코드 안의 경로를 고치지 않고 실행 인자로 checkpoint 경로를 넘기는 방식을 기본으로 한다.

핵심 인자:

- Stage A 재사용 pt 지정: `--stagea-checkpoint /absolute/path/to/student_comment_distill.pt`
- Stage B 결과 저장 위치 지정: `--stageb-output-dir /absolute/path/to/output_dir`
- bundle 최종 저장 파일명 직접 지정: `--bundle-output /absolute/path/to/offline_bundle_xxx.pt`
- bundle 이름만 짧게 덮어쓰고 싶을 때: `--source-tag my_exp_name`

즉, 보통 실험마다 바꾸는 것은 코드가 아니라 실행 명령의 인자다.

## 스크립트 순서

핵심 학습 흐름:

- `run/10_online_normalize_data.sh`
- `run/12_online_train_stage_a.sh`
- `run/14_online_train_stage_b.sh`
- `run/15_online_export_offline_bundle.sh`
- `run/16_online_pipeline.sh`

보조 유틸리티:

- `run/11_online_prepare_stage_a_teacher_targets.sh`
- `run/13_online_prepare_stage_b_targets.sh`

`11`은 teacher target이 아직 없을 때만 쓰는 보조 데이터 준비 스크립트다. `13`은 현재 기본 흐름에서는 쓰지 않는다.

## 1. Stage A teacher-target data 정규화

```bash
./run/10_online_normalize_data.sh   --kind stageA   --input /path/to/stagea_teacher_targets_raw.parquet
```

출력 파일명을 직접 지정하지 않으면 `outputs/stageA_normalized_<source>_<YYYYMMDD>.parquet` 규칙으로 자동 저장된다.

## 2. Stage A student 학습

```bash
./run/12_online_train_stage_a.sh   --input outputs/stagea_targets_norm.parquet
```

입력은 보통 `run/10_online_normalize_data.sh --kind stageA`의 출력이다.

출력 디렉터리를 직접 지정하지 않으면 `outputs/pt/stageA_comment_distill_<source>_<YYYYMMDD>/` 규칙으로 자동 생성된다.

## 3. Stage B gold input 정규화

```bash
./run/10_online_normalize_data.sh   --kind stageB   --input /path/to/stageb_gold_raw.xlsx
```

출력 파일명을 직접 지정하지 않으면 `outputs/stageB_normalized_<source>_<YYYYMMDD>.parquet` 규칙으로 자동 저장된다.

## 4. Stage B adapter 학습

```bash
./run/14_online_train_stage_b.sh   --input outputs/stageb_gold_norm.parquet   --stagea-checkpoint outputs/pt/stageA_comment_distill_xxx/student_comment_distill.pt   --label-map assets/kote_id2label.json
```

또는 최신 Stage A checkpoint 자동 사용:

```bash
./run/14_online_train_stage_b.sh   --input outputs/stageb_gold_norm.parquet
```

출력 디렉터리를 직접 지정하지 않으면 `outputs/pt/stageB_adapter_<source>_<YYYYMMDD>/` 규칙으로 자동 생성된다.

## 5. Offline bundle export

A만 학습했을 때는 base-only bundle export가 가능하다.

```bash
./run/15_online_export_offline_bundle.sh   --base-checkpoint outputs/pt/stageA_comment_distill_xxx/student_comment_distill.pt   --label-map assets/kote_id2label.json
```

B까지 학습했을 때는 Stage B checkpoint를 같이 묶는다.

```bash
./run/15_online_export_offline_bundle.sh   --base-checkpoint outputs/pt/stageA_comment_distill_xxx/student_comment_distill.pt   --stageb-checkpoint outputs/pt/stageB_adapter_xxx/stageB_adapter_checkpoint.pt   --label-map assets/kote_id2label.json
```

bundle 출력 파일명을 직접 지정하지 않으면 아래처럼 자동 저장된다.

- A only: `outputs/pt/offline_bundle_stagea_only_<bundle-source>_<YYYYMMDD>.pt`
- A+B: `outputs/pt/offline_bundle_stagea_stageb_<bundle-source>_<YYYYMMDD>.pt`
- 기본 `<bundle-source>`는 실제로 사용한 Stage A / Stage B checkpoint 디렉터리명을 조합해서 만든 provenance tag다.

## 6. 전체 파이프라인 한 번에 실행

### A+B 전체 실행

```bash
./run/16_online_pipeline.sh   --mode full   --stagea-input /path/to/stagea_teacher_targets_raw.parquet   --label-map assets/kote_id2label.json   --stageb-input /path/to/stageb_gold_raw.xlsx
```

내부 순서:

1. `run/10_online_normalize_data.sh --kind stageA`
2. `run/12_online_train_stage_a.sh`
3. `run/10_online_normalize_data.sh --kind stageB`
4. `run/14_online_train_stage_b.sh`
5. `run/15_online_export_offline_bundle.sh`

### A만 실행

```bash
./run/16_online_pipeline.sh   --mode stagea   --stagea-input /path/to/stagea_teacher_targets_raw.parquet   --label-map assets/kote_id2label.json
```

이 경우:

- Stage A pt는 새로 업데이트된다.
- base-only offline bundle도 같이 export된다.

### B만 실행

```bash
./run/16_online_pipeline.sh   --mode stageb   --stagea-checkpoint outputs/pt/stageA_comment_distill_xxx/student_comment_distill.pt   --label-map assets/kote_id2label.json   --stageb-input /path/to/stageb_gold_raw.xlsx
```

또는 최신 Stage A checkpoint 자동 사용:

```bash
./run/16_online_pipeline.sh   --mode stageb   --stagea-checkpoint /absolute/path/to/student_comment_distill.pt   --stageb-input /path/to/stageb_gold_raw.xlsx
```

이 경우:

- `--stagea-checkpoint`를 주면 그 경로를 쓴다.
- 생략하면 `outputs/pt` 아래 최신 Stage A checkpoint를 자동 사용한다.
- Stage B pt는 새로 업데이트된다.
- 그 Stage A + 새 Stage B를 묶은 offline bundle도 같이 export된다.

## 유의사항

- Stage B 학습은 반드시 Stage A checkpoint를 기반으로 한다. 즉 `B only`라도 내부적으로는 기존 `A pt`가 먼저 있어야 한다.
- 기본값은 `assets/kote_id2label.json`이다. 현재 KOTE 44라벨 고정 순서를 쓸 때는 이 asset만 써도 된다.
- 다만 teacher/annotation 단계에서 다른 라벨 순서를 의도적으로 사용했다면 그때만 `--label-map`을 명시적으로 덮어써야 한다.
- `--stagea-checkpoint`를 생략하면 `outputs/pt` 아래에서 가장 최신 `student_comment_distill.pt`를 자동 선택한다. 실험을 여러 개 병렬로 돌렸거나 다른 실험 산출물이 섞여 있으면, 이 자동 선택이 의도와 다를 수 있다.
- 따라서 재현성이나 실험 통제를 중요하게 볼 때는 `--stagea-checkpoint`를 명시적으로 주는 것을 권장한다.
- `A only` 모드는 base-only bundle을 export한다. 이 bundle에는 Stage B adapter가 포함되지 않으며 파일명도 `offline_bundle_stagea_only_...pt`로 저장된다. 기본 이름에는 어떤 Stage A pt를 썼는지도 같이 들어간다.
- `B only` 모드는 기존 Stage A checkpoint를 재사용해서 Stage B만 새로 학습하고, 마지막에는 그 Stage A + 새 Stage B를 묶은 bundle을 다시 export한다. 이 파일명은 `offline_bundle_stagea_stageb_...pt`로 저장되며, 기본 이름에 Stage A pt와 Stage B pt 출처가 같이 들어간다.
- Stage A 입력은 이미 만들어진 teacher-target dataset이어야 한다. 이 패키지의 기본 온라인 학습은 teacher target 생성부터 시작하지 않는다.
- Stage A와 Stage B raw input은 모두 먼저 정규화한다. 원본 raw 파일을 바로 train script에 넣는 것보다 normalize output을 중간 산출물로 남겨두는 편이 디버깅과 재현에 유리하다.
- 출력 경로를 생략하면 날짜가 붙은 자동 이름이 생성된다. bundle은 이제 `stagea_only`와 `stagea_stageb`가 파일명에 포함되고, 기본 source 부분도 실제 checkpoint 디렉터리명을 반영해 어떤 pt를 썼는지 더 잘 보이게 저장된다. 그래도 이름을 더 짧게 하거나 실험 표기를 통일하고 싶으면 `--source-tag`나 `--bundle-output`으로 직접 지정하면 된다.
- Stage B는 이제 gold 데이터만 가정한다. `legacy/proxy` 형식은 현재 기본 흐름에서 지원 대상으로 보지 않는다.
- CUDA가 없으면 CPU fallback이 발생할 수 있고, toy sample은 괜찮아도 실제 데이터에서는 속도가 많이 느려질 수 있다.
