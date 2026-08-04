from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from .labels import KOTE_LABELS, get_default_label_map_path


def build_context_text(title: str | None, comment: str) -> str:
    title = (title or "").strip()
    comment = (comment or "").strip()
    if not title:
        return comment
    return f"제목: {title}\n댓글: {comment}"


def load_label_map(path: str | Path | None = None) -> dict[int, str]:
    resolved_path = Path(path) if path is not None else get_default_label_map_path()
    if resolved_path.exists():
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}
    return {idx: label for idx, label in enumerate(KOTE_LABELS)}


class StageAStudent(nn.Module):
    def __init__(self, model_name: str, num_labels: int, local_files_only: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


class FrozenStageAStudent(StageAStudent):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        checkpoint_path: str | Path | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        local_files_only: bool = False,
    ):
        super().__init__(model_name=model_name, num_labels=num_labels, local_files_only=local_files_only)
        if checkpoint_path is None and state_dict is None:
            raise ValueError("Either checkpoint_path or state_dict must be provided")
        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)
        else:
            self._load_checkpoint(checkpoint_path)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _load_checkpoint(self, checkpoint_path: str | Path | None) -> None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must not be None")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return super().forward(input_ids=input_ids, attention_mask=attention_mask)


class StageBSarcasmAdapter(nn.Module):
    def __init__(
        self,
        model_name: str,
        stagea_checkpoint: str | Path | None,
        num_labels: int,
        local_files_only: bool = False,
        stagea_state_dict: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        if stagea_checkpoint is None and stagea_state_dict is None:
            raise ValueError("Either stagea_checkpoint or stagea_state_dict must be provided")

        self.base_model = FrozenStageAStudent(
            model_name=model_name,
            num_labels=num_labels,
            checkpoint_path=stagea_checkpoint,
            state_dict=stagea_state_dict,
            local_files_only=local_files_only,
        )
        self.context_encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = self.context_encoder.config.hidden_size
        self.context_dropout = nn.Dropout(getattr(self.context_encoder.config, "hidden_dropout_prob", 0.1))
        self.context_norm = nn.LayerNorm(hidden)
        self.gate_head = nn.Linear(hidden, 1)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )
        self._load_context_init(stagea_checkpoint=stagea_checkpoint, stagea_state_dict=stagea_state_dict)

    def _load_context_init(
        self,
        stagea_checkpoint: str | Path | None = None,
        stagea_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if stagea_state_dict is None:
            if stagea_checkpoint is None:
                raise ValueError("Either stagea_checkpoint or stagea_state_dict must be provided")
            stagea_state_dict = torch.load(stagea_checkpoint, map_location="cpu")
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in stagea_state_dict.items()
            if key.startswith("encoder.")
        }
        self.context_encoder.load_state_dict(encoder_state, strict=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        comment_input_ids: torch.Tensor,
        comment_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.base_model.eval()
        base_logits = self.base_model(comment_input_ids, comment_attention_mask)
        context_out = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            return_dict=True,
        )
        cls = context_out.last_hidden_state[:, 0, :]
        cls = self.context_dropout(self.context_norm(cls))
        gate_logits = self.gate_head(cls).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logits)
        delta_logits = self.delta_head(cls)
        corrected_logits = base_logits + gate_prob.unsqueeze(-1) * delta_logits
        return {
            "base_logits": base_logits,
            "gate_logits": gate_logits,
            "gate_prob": gate_prob,
            "delta_logits": delta_logits,
            "corrected_logits": corrected_logits,
        }
