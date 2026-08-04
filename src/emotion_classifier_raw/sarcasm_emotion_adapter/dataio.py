from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

COMMENT_KEYS = ("comment", "comment_text", "text", "content", "body")
TITLE_KEYS = ("news_title", "title", "raw_title", "headline")
DATE_KEYS = ("date", "news_date", "article_date", "published_at", "created_at")
SARCASTIC_POSITIVE = {"o", "1", "true", "t", "yes", "y", "sarcasm", "sarcastic"}
SARCASTIC_NEGATIVE = {"x", "0", "false", "f", "no", "n", "non_sarcasm", "non-sarcasm", "plain"}


def parse_hf_source(hf_source: str) -> tuple[str, str | None]:
    parts = [part for part in str(hf_source).strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("hf_source must look like 'org/repo/optional/subdir'")
    repo_id = "/".join(parts[:2])
    subdir = "/".join(parts[2:]) or None
    return repo_id, subdir


def _scalarize(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def _first_nonempty(mapping: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = str(mapping[key]).strip()
            if value:
                return value
    return ""


def _normalize_comment_item(comment_item) -> dict | None:
    if isinstance(comment_item, str):
        text = comment_item.strip()
        return {"comment": text} if text else None
    if not isinstance(comment_item, dict):
        return None

    row = {key: _scalarize(value) for key, value in comment_item.items()}
    text = _first_nonempty(row, COMMENT_KEYS)
    if not text:
        return None
    row["comment"] = text
    return row


def iter_mindcast_posts(payload: dict):
    for day_item in payload.get("data", []) or []:
        if isinstance(day_item.get("posts"), list):
            dataset_date = day_item.get("date")
            for post_idx, post in enumerate(day_item.get("posts", []) or []):
                yield dataset_date, post_idx, post
            continue

        for nested_day in day_item.get("dates", []) or []:
            dataset_date = nested_day.get("date")
            for post_idx, post in enumerate(nested_day.get("posts", []) or []):
                yield dataset_date, post_idx, post


def flatten_mindcast_payload(payload: dict, source_file: str = "") -> pd.DataFrame:
    rows: list[dict] = []
    for dataset_date, post_idx, post in iter_mindcast_posts(payload):
        title = _first_nonempty(post, TITLE_KEYS)
        news_date = _scalarize(post.get("news_date"))
        canonical_date = news_date if news_date is not None else dataset_date
        post_base = {
            "source_file": source_file,
            "dataset_date": dataset_date,
            "news_date": news_date,
            "date": canonical_date,
            "post_index_in_day": post_idx,
            "news_title": title,
            "title": title,
            "raw_title": _scalarize(post.get("raw_title")),
        }
        for key, value in post.items():
            if key in {"comments", "dates", "posts"}:
                continue
            if key not in post_base:
                post_base[key] = _scalarize(value)

        comments = post.get("comments", []) or []
        for comment_idx, comment_item in enumerate(comments):
            comment_row = _normalize_comment_item(comment_item)
            if comment_row is None:
                continue
            row = {**post_base, **comment_row}
            row["comment_index"] = comment_idx
            if not str(row.get("candidate_id", "")).strip():
                row["candidate_id"] = f"{source_file}::{post_idx}::{comment_idx}"
            rows.append(row)
    return pd.DataFrame(rows)


def _normalize_binary_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in SARCASTIC_POSITIVE:
        return 1
    if text in SARCASTIC_NEGATIVE:
        return 0
    return None


def _copy_first_available(df: pd.DataFrame, target: str, candidates: list[str]) -> None:
    if target in df.columns:
        return
    for candidate in candidates:
        if candidate in df.columns:
            df[target] = df[candidate]
            return


def standardize_frame(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    work = df.copy()
    _copy_first_available(work, "comment", ["comment_text", "text", "content", "body"])
    _copy_first_available(work, "news_title", ["title", "raw_title", "headline"])
    _copy_first_available(work, "title", ["news_title", "raw_title", "headline"])
    _copy_first_available(work, "news_date", ["date", "article_date", "published_at", "created_at"])
    _copy_first_available(work, "date", ["news_date", "dataset_date", "article_date", "published_at", "created_at"])
    _copy_first_available(work, "dataset_date", ["date"])

    _copy_first_available(work, "base_emotion_label", ["comment_only_pred", "emotion_label_pred", "teacher_pred_label", "predicted_emotion_label"])
    _copy_first_available(work, "actual_emotion_target", ["actual_emotion_label", "gold_emotion_label", "emotion_label", "label"])
    _copy_first_available(work, "final_emotion_target", ["actual_emotion_target", "gold_emotion_label", "emotion_label", "label"])
    _copy_first_available(work, "sarcasm_annotation", ["sarcasm", "sarcasm_gold", "sarcasm_tag"])

    if "sarcasm_label" not in work.columns and "sarcasm_annotation" in work.columns:
        work["sarcasm_label"] = work["sarcasm_annotation"].map(_normalize_binary_label)
    elif "sarcasm_label" in work.columns:
        work["sarcasm_label"] = work["sarcasm_label"].map(_normalize_binary_label)

    if "candidate_id" not in work.columns:
        if {"source_file", "post_index_in_day", "comment_index"}.issubset(work.columns):
            work["candidate_id"] = (
                work["source_file"].astype(str)
                + "::"
                + work["post_index_in_day"].astype(str)
                + "::"
                + work["comment_index"].astype(str)
            )
        else:
            work = work.reset_index(drop=True)
            work["candidate_id"] = [f"row::{idx}" for idx in range(len(work))]

    if "comment" not in work.columns:
        raise ValueError("Could not find a comment column")
    if "news_title" not in work.columns:
        work["news_title"] = ""
    if "title" not in work.columns:
        work["title"] = work["news_title"]
    if "news_date" not in work.columns:
        work["news_date"] = ""
    if "date" not in work.columns:
        work["date"] = work["news_date"]
    if "dataset_date" not in work.columns:
        work["dataset_date"] = ""

    work["comment"] = work["comment"].fillna("").astype(str)
    work["news_title"] = work["news_title"].fillna("").astype(str)
    work["title"] = work["title"].fillna("").astype(str)
    work["news_date"] = work["news_date"].fillna("").astype(str)
    work["date"] = work["date"].fillna("").astype(str)
    work["dataset_date"] = work["dataset_date"].fillna("").astype(str)
    work = work[work["comment"].str.strip().ne("")].reset_index(drop=True)
    return work.reset_index(drop=True)


def _read_json_like(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "data" in payload:
        return flatten_mindcast_payload(payload, source_file=str(path))
    if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
        return pd.DataFrame(payload["items"])
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise ValueError("Unsupported JSON structure")


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(
        "Failed to read the input as CSV. Tried encodings: " + "; ".join(errors)
    )


def load_local_dataframe(input_path: str | Path, seed: int = 42) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()
    header = path.read_bytes()[:16]
    stripped = header.lstrip()

    if suffix in {".pt", ".pth", ".bin"}:
        raise ValueError(
            f"Unsupported training input file: {path}. This looks like a checkpoint/binary file, not a dataset. "
            "For Stage B training, pass a normalized dataset such as .parquet, .csv, .xlsx, or .json."
        )

    try:
        if suffix == ".parquet" or header.startswith(b"PAR1"):
            df = pd.read_parquet(path)
        elif suffix in {".xlsx", ".xls"} or header.startswith(b"PK\x03\x04"):
            df = pd.read_excel(path)
        elif suffix == ".json" or (stripped[:1] in {b"{", b"["}):
            df = _read_json_like(path)
        else:
            df = _read_csv_with_fallback(path)
    except Exception as first_exc:
        if suffix not in {".parquet", ".xlsx", ".xls", ".json", ".csv", ".tsv", ".txt"}:
            try:
                if header.startswith(b"PAR1"):
                    df = pd.read_parquet(path)
                elif header.startswith(b"PK\x03\x04"):
                    df = pd.read_excel(path)
                elif stripped[:1] in {b"{", b"["}:
                    df = _read_json_like(path)
                else:
                    df = _read_csv_with_fallback(path)
            except Exception as fallback_exc:
                raise ValueError(
                    f"Could not infer dataset format for {path}. Primary error: {first_exc}. Fallback error: {fallback_exc}"
                ) from fallback_exc
        else:
            raise
    return standardize_frame(df, seed=seed)


def list_hf_json_files(repo_id: str, subdir: str | None = None, revision: str = "main", token: str | None = None) -> list[str]:
    repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision, token=token)
    if subdir and subdir.endswith(".json"):
        return [subdir] if subdir in repo_files else []

    prefix = None if not subdir else subdir.rstrip("/") + "/"
    matched = []
    for path in repo_files:
        if not path.endswith(".json"):
            continue
        if prefix is not None and not path.startswith(prefix):
            continue
        matched.append(path)
    return sorted(matched)


def load_hf_dataframe(
    hf_source: str | None = None,
    hf_repo_id: str | None = None,
    hf_subdir: str | None = None,
    revision: str = "main",
    token: str | None = None,
    max_files: int | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if hf_source:
        hf_repo_id, parsed_subdir = parse_hf_source(hf_source)
        hf_subdir = hf_subdir or parsed_subdir
    if not hf_repo_id:
        raise ValueError("hf_repo_id is required when loading from Hugging Face")

    repo_paths = list_hf_json_files(repo_id=hf_repo_id, subdir=hf_subdir, revision=revision, token=token)
    if max_files is not None:
        repo_paths = repo_paths[:max_files]
    if not repo_paths:
        raise ValueError("No JSON files matched the requested Hugging Face source")

    frames = []
    for repo_path in repo_paths:
        local_path = hf_hub_download(repo_id=hf_repo_id, repo_type="dataset", filename=repo_path, revision=revision, token=token)
        with open(local_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            frame = flatten_mindcast_payload(payload, source_file=repo_path)
        elif isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
            frame = pd.DataFrame(payload["items"])
            frame["source_file"] = repo_path
        elif isinstance(payload, list):
            frame = pd.DataFrame(payload)
            frame["source_file"] = repo_path
        elif isinstance(payload, dict):
            frame = pd.DataFrame([payload])
            frame["source_file"] = repo_path
        else:
            raise ValueError(f"Unsupported JSON structure in {repo_path}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = standardize_frame(df, seed=seed)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def load_dataset_frame(
    input_path: str | Path | None = None,
    hf_source: str | None = None,
    hf_repo_id: str | None = None,
    hf_subdir: str | None = None,
    hf_revision: str = "main",
    hf_token: str | None = None,
    hf_max_files: int | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if input_path:
        df = load_local_dataframe(input_path, seed=seed)
    elif hf_source or hf_repo_id:
        df = load_hf_dataframe(
            hf_source=hf_source,
            hf_repo_id=hf_repo_id,
            hf_subdir=hf_subdir,
            revision=hf_revision,
            token=hf_token,
            max_files=hf_max_files,
            max_rows=max_rows,
            seed=seed,
        )
    else:
        raise ValueError("Either input_path or hf_source/hf_repo_id must be provided")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def write_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
