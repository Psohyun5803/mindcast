"""
GSM8K Evaluation
Base model vs SFT LoRA fine-tuned model

Usage:
    # base만 평가
    python eval_gsm8k.py --model_id base --model_path mistralai/Mistral-7B-v0.1

    # lora 평가
    python eval_gsm8k.py --model_id lora --model_path mistralai/Mistral-7B-v0.1 \
        --lora_path /home/sohyun44/nlp_project/outputs/sft_lora_mistral7b
"""

import argparse
import json
import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",   type=str, required=True, help="결과 저장용 식별자 (e.g. base, lora)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path",  type=str, default=None)
    parser.add_argument("--gpu",        type=str, default="2")
    parser.add_argument("--num_samples", type=int, default=None, help="테스트 샘플 수 (기본: 전체 1319개)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./gsm8k_results")
    return parser.parse_args()


def load_model(model_path, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if lora_path:
        print(f"Loading LoRA from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("Merged.")

    model.eval()
    return model, tokenizer


def extract_gt_answer(answer_str):
    """GSM8K ground truth: 마지막 '#### {숫자}' 파싱"""
    match = re.search(r"####\s*([\-\d,\.]+)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def extract_pred_answer(generated_str):
    """모델 생성 텍스트에서 숫자 추출 — '####' 우선, 없으면 마지막 숫자"""
    match = re.search(r"####\s*([\-\d,\.]+)", generated_str)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", generated_str)
    return numbers[-1].replace(",", "") if numbers else None


def build_prompt(question):
    return f"### Question:\n{question}\n\n### Answer:\n"


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.lora_path)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_samples:
        ds = ds.select(range(args.num_samples))
    print(f"Evaluating on {len(ds)} GSM8K test samples...")

    correct = 0
    results = []

    for ex in tqdm(ds, desc=f"[{args.model_id}]"):
        prompt = build_prompt(ex["question"])
        gt = extract_gt_answer(ex["answer"])

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        pred = extract_pred_answer(generated)

        is_correct = (pred == gt)
        if is_correct:
            correct += 1

        results.append({
            "question": ex["question"],
            "gt":       gt,
            "pred":     pred,
            "correct":  is_correct,
            "generated": generated,
        })

    accuracy = correct / len(ds) * 100
    print(f"\n[{args.model_id}] GSM8K Accuracy: {correct}/{len(ds)} = {accuracy:.2f}%")

    output_file = os.path.join(args.output_dir, f"{args.model_id}.json")
    with open(output_file, "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": len(ds), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
