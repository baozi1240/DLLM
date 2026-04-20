import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Dream-v0-Base-7B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_shot", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_layer", type=int, default=3)
    parser.add_argument("--focus_topk", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="gsm8k_results.jsonl")
    parser.add_argument("--stats_path", type=str, default="gsm8k_stats.json")
    return parser.parse_args()


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def normalize_number(text):
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace("$", "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value == value.to_integral():
        return str(int(value))
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


def extract_last_number(text):
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace("$", ""))
    if not matches:
        return None
    return normalize_number(matches[-1])


def extract_prediction(text):
    boxed_match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if boxed_match:
        return normalize_number(boxed_match.group(1))
    return extract_last_number(text)


def extract_reference(answer):
    boxed_match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if boxed_match:
        return normalize_number(boxed_match.group(1))
    return extract_last_number(answer)


def build_few_shot_prefix(examples):
    if not examples:
        return ""

    parts = ["Here are some examples:\n"]
    for idx, example in enumerate(examples, start=1):
        parts.append(f"Example {idx}:\n")
        parts.append(f"Question: {example['question'].strip()}\n")
        parts.append(f"Answer: {example['answer'].strip()}\n\n")
    return "".join(parts)


def build_messages(question, few_shot_prefix=""):
    prompt = few_shot_prefix
    if few_shot_prefix:
        prompt += "Now solve the next problem.\n\n"
    prompt += (
        f"Question: {question.strip()}\n\n"
        + "Please solve the math word problem step by step. "
        + "End your response with '#### <answer>'."
    )
    return [{"role": "user", "content": prompt}]


@torch.no_grad()
def generate_batch(model, tokenizer, batch_questions, few_shot_prefix, device, args):
    messages = [build_messages(question, few_shot_prefix) for question in batch_questions]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        padding=True,
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    outputs = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        block_length=args.block_length,
        use_cache=args.use_cache or args.dual_cache,
        dual_cache=args.dual_cache,
        focus_decode=args.focus_decode,
        focus_layer=args.focus_layer,
        focus_topk=args.focus_topk,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )

    responses = []
    for seq, prompt_len in zip(outputs.sequences, prompt_lengths):
        text = tokenizer.decode(seq[prompt_len:].tolist(), skip_special_tokens=False)
        if tokenizer.eos_token:
            text = text.split(tokenizer.eos_token)[0]
        responses.append(text.strip())
    return responses


def main():
    args = parse_args()
    if args.focus_decode:
        if not args.dual_cache:
            raise ValueError("focus_decode evaluation requires --dual_cache.")
        if args.batch_size != 1:
            raise ValueError("focus_decode evaluation currently requires --batch_size 1.")
        if args.focus_layer <= 0:
            raise ValueError(f"focus_layer must be positive, got {args.focus_layer}")
        if args.focus_topk <= 0:
            raise ValueError(f"focus_topk must be positive, got {args.focus_topk}")

    device = select_device()
    dtype_by_device = {
        "cuda": torch.bfloat16,
        "mps": torch.float16,
        "cpu": torch.float32,
    }
    dtype = dtype_by_device[device]

    print(f"Using device: {device} (dtype={dtype})", flush=True)
    print(f"Loading model from: {args.model_path}", flush=True)
    print(
        "Eval config: "
        f"use_cache={args.use_cache or args.dual_cache}, "
        f"dual_cache={args.dual_cache}, "
        f"focus_decode={args.focus_decode}, "
        f"focus_layer={args.focus_layer}, "
        f"focus_topk={args.focus_topk}",
        flush=True,
    )

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device).eval()

    dataset = load_dataset("gsm8k", "main", split=args.split)
    few_shot_examples = []
    if args.n_shot > 0:
        few_shot_dataset = load_dataset("gsm8k", "main", split="train")
        shot_count = min(args.n_shot, len(few_shot_dataset))
        few_shot_examples = [
            few_shot_dataset[i]
            for i in range(shot_count)
        ]
    few_shot_prefix = build_few_shot_prefix(few_shot_examples)

    total_len = len(dataset)
    start = max(args.start, 0)
    end = total_len if args.end is None else min(args.end, total_len)
    dataset = dataset.select(range(start, end))
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Evaluating {len(dataset)} samples from gsm8k[{args.split}]", flush=True)
    print(f"n_shot: {len(few_shot_examples)}", flush=True)

    output_path = Path(args.output_path)
    stats_path = Path(args.stats_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.parent != Path("."):
        stats_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    total_generation_time = 0.0
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    with output_path.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, len(dataset), args.batch_size):
            batch = dataset[batch_start : batch_start + args.batch_size]
            questions = batch["question"]
            references = batch["answer"]
            batch_id = batch_start // args.batch_size + 1

            print(
                f"Starting batch {batch_id}/{total_batches} "
                f"(samples {batch_start}..{batch_start + len(questions) - 1})",
                flush=True,
            )

            if device == "cuda":
                torch.cuda.synchronize()
            batch_t0 = time.perf_counter()
            predictions = generate_batch(model, tokenizer, questions, few_shot_prefix, device, args)
            if device == "cuda":
                torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - batch_t0
            total_generation_time += batch_elapsed

            for question, reference, prediction in zip(questions, references, predictions):
                gold = extract_reference(reference)
                pred = extract_prediction(prediction)
                is_correct = pred is not None and pred == gold

                total += 1
                correct += int(is_correct)

                record = {
                    "index": start + total - 1,
                    "question": question,
                    "reference_answer": reference,
                    "gold": gold,
                    "prediction": prediction,
                    "pred": pred,
                    "correct": is_correct,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                print("=" * 80, flush=True)
                print(f"[{total}] correct={is_correct} acc={correct / total:.4f}", flush=True)
                print("question:", question, flush=True)
                print("gold:", gold, flush=True)
                print("pred:", pred, flush=True)
                print("response:", prediction, flush=True)

    stats = {
        "model_path": args.model_path,
        "split": args.split,
        "n_shot": len(few_shot_examples),
        "block_length": args.block_length,
        "use_cache": bool(args.use_cache or args.dual_cache),
        "dual_cache": bool(args.dual_cache),
        "focus_decode": bool(args.focus_decode),
        "focus_layer": int(args.focus_layer),
        "focus_topk": int(args.focus_topk),
        "start": start,
        "end": start + total,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_generation_time_sec": total_generation_time,
        "avg_generation_time_sec": total_generation_time / total if total else 0.0,
        "output_path": str(output_path),
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 80, flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
