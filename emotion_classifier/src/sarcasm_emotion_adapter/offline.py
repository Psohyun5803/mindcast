from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from .labels import load_small_to_major
from .modeling import StageAStudent, StageBSarcasmAdapter, build_context_text, load_label_map

BUNDLE_TYPE = "sarcasm_emotion_adapter_offline_bundle"
BUNDLE_VERSION = 1
BUNDLE_VARIANT_STAGEA_ONLY = "stagea_only"
BUNDLE_VARIANT_STAGEA_STAGEB = "stagea_stageb"
META_FIELDS = ("candidate_id", "date", "news_date", "dataset_date", "source_file")


def _extract_adapter_state_dict(raw_adapter: dict) -> dict[str, torch.Tensor]:
    if "model_state_dict" in raw_adapter:
        return raw_adapter["model_state_dict"]
    return raw_adapter


def _copy_metadata(row: dict) -> dict:
    meta = {}
    for key in META_FIELDS:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            meta[key] = text
    return meta


def _parse_created_at_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def read_bundle_metadata(bundle_path: str | Path) -> dict:
    bundle_path = Path(bundle_path)
    meta_path = bundle_path.with_suffix(bundle_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def resolve_bundle_checkpoint(
    bundle_checkpoint: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    pattern: str = "offline_bundle*.pt",
) -> Path:
    if bundle_checkpoint is not None:
        bundle_path = Path(bundle_checkpoint)
        if bundle_path.is_dir():
            bundle_dir = bundle_path
        else:
            if not bundle_path.exists():
                raise FileNotFoundError(f"Bundle checkpoint not found: {bundle_path}")
            return bundle_path.resolve()

    if bundle_dir is not None:
        search_dirs = [Path(bundle_dir)]
    else:
        base_outputs = Path(__file__).resolve().parents[2] / "outputs"
        search_dirs = [base_outputs / "pt", base_outputs]

    existing_dirs = [path for path in search_dirs if path.exists()]
    if not existing_dirs:
        joined = ", ".join(str(path) for path in search_dirs)
        hint = ""
        joined_text = str(search_dirs[0]) if search_dirs else ""
        if "/path/to/" in joined_text:
            hint = (
                " The value looks like a placeholder path from the docs. "
                "Replace it with a real bundle directory, or omit --bundle-dir to search the default outputs/pt and outputs directories."
            )
        raise FileNotFoundError(f"Bundle search directory not found. Checked: {joined}.{hint}")

    candidates: list[Path] = []
    for search_dir in existing_dirs:
        candidates.extend([path for path in search_dir.glob(pattern) if path.is_file()])

    if not candidates:
        joined = ", ".join(str(path) for path in existing_dirs)
        raise FileNotFoundError(
            f"No bundle checkpoint matched pattern '{pattern}' in directories: {joined}"
        )

    def sort_key(path: Path):
        meta = read_bundle_metadata(path)
        created_at = _parse_created_at_utc(meta.get("created_at_utc"))
        timestamp = created_at.timestamp() if created_at is not None else path.stat().st_mtime
        return (timestamp, path.name)

    return max(candidates, key=sort_key).resolve()


def export_offline_bundle(
    bundle_path: str | Path,
    student_model: str,
    base_checkpoint: str | Path,
    label_map_path: str | Path,
    adapter_checkpoint: str | Path | None = None,
    major_mapping_path: str | Path | None = None,
    max_length: int = 192,
    top_k: int = 5,
) -> Path:
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    id2label = load_label_map(label_map_path)
    small_to_major = load_small_to_major(major_mapping_path)
    base_state_dict = torch.load(base_checkpoint, map_location="cpu")
    bundle_variant = (
        BUNDLE_VARIANT_STAGEA_STAGEB if adapter_checkpoint is not None else BUNDLE_VARIANT_STAGEA_ONLY
    )

    created_at_utc = datetime.now(timezone.utc).isoformat()
    bundle = {
        "bundle_type": BUNDLE_TYPE,
        "bundle_version": BUNDLE_VERSION,
        "bundle_variant": bundle_variant,
        "created_at_utc": created_at_utc,
        "student_model": student_model,
        "base_checkpoint_path": str(base_checkpoint),
        "adapter_checkpoint_path": str(adapter_checkpoint) if adapter_checkpoint is not None else None,
        "label_map_path": str(label_map_path),
        "major_mapping_path": str(major_mapping_path) if major_mapping_path is not None else None,
        "id2label": {int(k): str(v) for k, v in id2label.items()},
        "small_to_major": dict(small_to_major),
        "num_labels": len(id2label),
        "max_length": int(max_length),
        "top_k": int(top_k),
        "base_state_dict": base_state_dict,
        "adapter_state_dict": None,
    }

    if adapter_checkpoint is not None:
        raw_adapter = torch.load(adapter_checkpoint, map_location="cpu")
        bundle["adapter_state_dict"] = _extract_adapter_state_dict(raw_adapter)

    torch.save(bundle, bundle_path)
    with open(bundle_path.with_suffix(bundle_path.suffix + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "bundle": str(bundle_path),
                "bundle_variant": bundle_variant,
                "created_at_utc": created_at_utc,
                "student_model": student_model,
                "base_checkpoint_path": str(base_checkpoint),
                "adapter_checkpoint_path": str(adapter_checkpoint) if adapter_checkpoint is not None else None,
                "label_map_path": str(label_map_path),
                "major_mapping_path": str(major_mapping_path) if major_mapping_path is not None else None,
                "num_labels": len(id2label),
                "max_length": int(max_length),
                "top_k": int(top_k),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return bundle_path


class OfflineSarcasmEmotionPredictor:
    def __init__(
        self,
        student_model: str,
        base_checkpoint: str | Path,
        label_map_path: str | Path,
        adapter_checkpoint: str | Path | None = None,
        major_mapping_path: str | Path | None = None,
        max_length: int = 192,
        batch_size: int = 32,
        top_k: int = 5,
        local_files_only: bool = False,
        device: str | None = None,
    ):
        self.id2label = load_label_map(label_map_path)
        self.small_to_major = load_small_to_major(major_mapping_path)
        self.num_labels = len(self.id2label)
        self.max_length = max_length
        self.batch_size = batch_size
        self.top_k = max(1, int(top_k))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.student_model = student_model

        self.tokenizer = AutoTokenizer.from_pretrained(student_model, local_files_only=local_files_only)
        self.base_model = StageAStudent(student_model, self.num_labels, local_files_only=local_files_only).to(self.device)
        base_state = torch.load(base_checkpoint, map_location="cpu")
        self.base_model.load_state_dict(base_state, strict=True)
        self.base_model.eval()

        self.adapter_model = None
        if adapter_checkpoint is not None:
            adapter_meta = torch.load(adapter_checkpoint, map_location="cpu")
            self.adapter_model = StageBSarcasmAdapter(
                model_name=student_model,
                stagea_checkpoint=base_checkpoint,
                stagea_state_dict=base_state,
                num_labels=self.num_labels,
                local_files_only=local_files_only,
            ).to(self.device)
            self.adapter_model.load_state_dict(_extract_adapter_state_dict(adapter_meta), strict=True)
            self.adapter_model.eval()

    @classmethod
    def from_bundle(
        cls,
        bundle_checkpoint: str | Path,
        batch_size: int = 32,
        local_files_only: bool = False,
        device: str | None = None,
    ) -> "OfflineSarcasmEmotionPredictor":
        bundle = torch.load(bundle_checkpoint, map_location="cpu")
        if bundle.get("bundle_type") != BUNDLE_TYPE:
            raise ValueError(f"Unsupported bundle type: {bundle.get('bundle_type')}")

        self = cls.__new__(cls)
        self.id2label = {int(k): str(v) for k, v in bundle["id2label"].items()}
        self.small_to_major = {str(k): str(v) for k, v in bundle["small_to_major"].items()}
        self.num_labels = int(bundle.get("num_labels", len(self.id2label)))
        self.max_length = int(bundle.get("max_length", 192))
        self.batch_size = batch_size
        self.top_k = max(1, int(bundle.get("top_k", 5)))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.student_model = str(bundle["student_model"])
        self.bundle_variant = str(
            bundle.get(
                "bundle_variant",
                BUNDLE_VARIANT_STAGEA_STAGEB
                if bundle.get("adapter_state_dict") is not None
                else BUNDLE_VARIANT_STAGEA_ONLY,
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.student_model, local_files_only=local_files_only)
        self.base_model = StageAStudent(self.student_model, self.num_labels, local_files_only=local_files_only).to(self.device)
        self.base_model.load_state_dict(bundle["base_state_dict"], strict=True)
        self.base_model.eval()

        self.adapter_model = None
        if bundle.get("adapter_state_dict") is not None:
            self.adapter_model = StageBSarcasmAdapter(
                model_name=self.student_model,
                stagea_checkpoint=None,
                stagea_state_dict=bundle["base_state_dict"],
                num_labels=self.num_labels,
                local_files_only=local_files_only,
            ).to(self.device)
            self.adapter_model.load_state_dict(bundle["adapter_state_dict"], strict=True)
            self.adapter_model.eval()

        return self

    @torch.no_grad()
    def predict(self, rows: list[dict[str, str]]) -> pd.DataFrame:
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "title": str(row.get("title", row.get("news_title", "")) or ""),
                    "comment": str(row.get("comment", "") or ""),
                    **_copy_metadata(row),
                }
            )

        results = []
        top_k = min(self.top_k, self.num_labels)
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            titles = [row["title"] for row in batch]
            comments = [row["comment"] for row in batch]
            title_present = torch.tensor([1.0 if title.strip() else 0.0 for title in titles], dtype=torch.float32, device=self.device)

            comment_enc = self.tokenizer(
                comments,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            context_texts = [build_context_text(title, comment) for title, comment in zip(titles, comments)]
            context_enc = self.tokenizer(
                context_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            comment_enc = {k: v.to(self.device) for k, v in comment_enc.items()}
            context_enc = {k: v.to(self.device) for k, v in context_enc.items()}

            base_logits = self.base_model(comment_enc["input_ids"], comment_enc["attention_mask"])
            final_logits = base_logits.clone()
            gate_prob = torch.zeros(len(batch), dtype=torch.float32, device=self.device)

            if self.adapter_model is not None:
                outputs = self.adapter_model(
                    comment_input_ids=comment_enc["input_ids"],
                    comment_attention_mask=comment_enc["attention_mask"],
                    context_input_ids=context_enc["input_ids"],
                    context_attention_mask=context_enc["attention_mask"],
                )
                gate_prob = outputs["gate_prob"] * title_present
                final_logits = outputs["base_logits"] + gate_prob.unsqueeze(-1) * outputs["delta_logits"]

            base_prob = torch.softmax(base_logits, dim=-1)
            final_prob = torch.softmax(final_logits, dim=-1)
            base_top_prob, base_top_idx = base_prob.max(dim=-1)
            final_top_prob, final_top_idx = final_prob.max(dim=-1)
            topk_prob, topk_idx = final_prob.topk(top_k, dim=-1)

            for idx in range(len(batch)):
                base_label = self.id2label[int(base_top_idx[idx].item())]
                final_label = self.id2label[int(final_top_idx[idx].item())]
                top5 = []
                for label_idx, score in zip(topk_idx[idx].tolist(), topk_prob[idx].tolist()):
                    label = self.id2label[int(label_idx)]
                    top5.append(
                        {
                            "label": label,
                            "major": self.small_to_major.get(label, ""),
                            "score": float(score),
                        }
                    )
                result = {
                    **_copy_metadata(batch[idx]),
                    "title": titles[idx],
                    "comment": comments[idx],
                    "title_provided": bool(titles[idx].strip()),
                    "gate_score": float(gate_prob[idx].item()),
                    "base_emotion_label_fine": base_label,
                    "base_emotion_label_major": self.small_to_major.get(base_label, ""),
                    "base_emotion_score": float(base_top_prob[idx].item()),
                    "final_emotion_label_fine": final_label,
                    "final_emotion_label_major": self.small_to_major.get(final_label, ""),
                    "final_emotion_score": float(final_top_prob[idx].item()),
                    "primary_emotion_label": final_label,
                    "primary_emotion_major": self.small_to_major.get(final_label, ""),
                    "primary_emotion_score": float(final_top_prob[idx].item()),
                    "top_5_json": json.dumps(top5, ensure_ascii=False),
                }
                results.append(result)

        return pd.DataFrame(results)
