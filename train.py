"""
SFT + QLoRA on Mistral-7B-v0.1
Dataset: orca-math-200k + gsm8k (train) + python_code_instructions_18k
"""

import os
import torch
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL    = "/home/sohyun44/nlp_project/mistral-orpo-alpha"
OUTPUT_DIR    = "/home/sohyun44/nlp_project/outputs/sft_lora_mistral7b"
MAX_LENGTH    = 2048
EPOCHS        = 1
BATCH_SIZE    = 8
GRAD_ACCUM    = 4                   # effective batch = 32
LEARNING_RATE = 5e-5
WARMUP_STEPS  = 100
SEED          = 42
GPU           = "0,1,2"


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_orca_math():
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    return ds.map(
        lambda x: {
            "prompt":   f"### Question:\n{x['question']}\n\n### Answer:\n",
            "response": x["answer"],
        },
        remove_columns=ds.column_names,
    )


def load_gsm8k_train():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return ds.map(
        lambda x: {
            "prompt":   f"### Question:\n{x['question']}\n\n### Answer:\n",
            "response": x["answer"],
        },
        remove_columns=ds.column_names,
    )


def load_python_code():
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    def fmt(x):
        inp = x["input"].strip()
        prompt = f"### Instruction:\n{x['instruction']}"
        if inp:
            prompt += f"\n\n### Input:\n{inp}"
        prompt += "\n\n### Output:\n"
        return {"prompt": prompt, "response": x["output"]}
    return ds.map(fmt, remove_columns=ds.column_names)


def build_dataset():
    print("Loading orca-math-200k...")
    orca = load_orca_math()
    print(f"  {len(orca):,} samples")

    print("Loading gsm8k train...")
    gsm = load_gsm8k_train()
    print(f"  {len(gsm):,} samples")

    print("Loading python_code_instructions_18k...")
    code = load_python_code()
    print(f"  {len(code):,} samples")

    combined = concatenate_datasets([orca, gsm, code]).shuffle(seed=SEED)
    print(f"  Total: {len(combined):,} samples")
    return combined


# ── Tokenization ──────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt_ids = self.tokenizer(
            ex["prompt"], add_special_tokens=False
        )["input_ids"]
        prompt_len = len(prompt_ids)

        full = ex["prompt"] + ex["response"]
        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, features):
        def pad(seqs, val):
            max_len = max(len(s) for s in seqs)
            return [s + [val] * (max_len - len(s)) for s in seqs]

        return {
            "input_ids":      torch.tensor(pad([f["input_ids"] for f in features], self.pad_token_id)),
            "attention_mask": torch.tensor(pad([f["attention_mask"] for f in features], 0)),
            "labels":         torch.tensor(pad([f["labels"] for f in features], -100)),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    print(f"Loading {BASE_MODEL} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.pad_token_id

    raw = build_dataset()
    dataset = SFTDataset(raw, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
