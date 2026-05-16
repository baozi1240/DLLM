import argparse
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from datasets import DownloadConfig, load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_GSM8K_DATASET_PATH = "/home/xuefeng/.cache/opencompass/data/gsm8k"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Dream-v0-Base-7B")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_GSM8K_DATASET_PATH,
        help="Optional local GSM8K dataset path. Supports load_from_disk directories or json/jsonl files.",
    )
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
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_layer", type=int, default=3)
    parser.add_argument("--focus_topk", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="gsm8k_results.jsonl")
    parser.add_argument("--stats_path", type=str, default="gsm8k_stats.json")
    return parser.parse_args()


def resolve_mode_name(args):
    if args.focus_decode:
        return "focus_dual_cache"
    if args.dual_cache:
        return "fast_dllm_dual_cache"
    if args.use_cache:
        return "fast_dllm_prefix_cache"
    return "baseline"


def load_gsm8k_split(args, split_name):
    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local dataset path not found: {dataset_path}")

        if dataset_path.is_dir():
            split_jsonl = dataset_path / f"{split_name}.jsonl"
            split_json = dataset_path / f"{split_name}.json"
            if split_jsonl.exists() or split_json.exists():
                split_file = split_jsonl if split_jsonl.exists() else split_json
                return load_dataset(
                    "json",
                    data_files={split_name: str(split_file)},
                    split=split_name,
                )

            dataset_obj = load_from_disk(str(dataset_path))
            if hasattr(dataset_obj, "keys"):
                if split_name not in dataset_obj:
                    raise ValueError(
                        f"Split '{split_name}' not found in local dataset directory: {dataset_path}"
                    )
                return dataset_obj[split_name]
            return dataset_obj

        suffixes = "".join(dataset_path.suffixes).lower()
        if suffixes.endswith(".jsonl") or suffixes.endswith(".json"):
            return load_dataset(
                "json",
                data_files={split_name: str(dataset_path)},
                split=split_name,
            )
        raise ValueError(
            "Unsupported --dataset_path format. Use a load_from_disk directory or a .json/.jsonl file."
        )

    download_config = DownloadConfig(local_files_only=True)
    return load_dataset("gsm8k", "main", split=split_name, download_config=download_config)


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
        threshold=args.threshold,
        gamma=args.gamma,
    )

    responses = []
    for seq, prompt_len in zip(outputs.sequences, prompt_lengths):
        text = tokenizer.decode(seq[prompt_len:].tolist(), skip_special_tokens=False)
        if tokenizer.eos_token:
            text = text.split(tokenizer.eos_token)[0]
        responses.append(text.strip())
    return responses


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_stats(
    *,
    args,
    dataset_path,
    device,
    few_shot_examples,
    start,
    total,
    correct,
    total_generation_time,
    wall_time,
    output_path,
):
    effective_use_cache = bool(args.use_cache or args.dual_cache)
    mode_name = resolve_mode_name(args)
    return {
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "mode": mode_name,
        "model_path": args.model_path,
        "device": device,
        "split": args.split,
        "n_shot": len(few_shot_examples),
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "threshold": args.threshold if args.alg == "confidence_threshold" else None,
        "gamma": args.gamma,
        "use_cache": effective_use_cache,
        "dual_cache": bool(args.dual_cache),
        "focus_decode": bool(args.focus_decode),
        "focus_layer": int(args.focus_layer) if args.focus_decode else 0,
        "focus_topk": int(args.focus_topk) if args.focus_decode else 0,
        "start": start,
        "end": start + total,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_generation_time_sec": total_generation_time,
        "avg_generation_time_sec": total_generation_time / total if total else 0.0,
        "wall_time_s": wall_time,
        "output_path": str(output_path),
    }


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

    dataset_source = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else None
    dataset = load_gsm8k_split(args, args.split)
    few_shot_examples = []
    if args.n_shot > 0:
        few_shot_dataset = load_gsm8k_split(args, "train")
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
    output_path.write_text("", encoding="utf-8")

    total = 0
    correct = 0
    total_generation_time = 0.0
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    wall_start = time.perf_counter()

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
                fout.flush()

                running_stats = build_stats(
                    args=args,
                    dataset_path=dataset_source,
                    device=device,
                    few_shot_examples=few_shot_examples,
                    start=start,
                    total=total,
                    correct=correct,
                    total_generation_time=total_generation_time,
                    wall_time=time.perf_counter() - wall_start,
                    output_path=output_path,
                )
                save_json(stats_path, running_stats)

                print("=" * 80, flush=True)
                print(
                    f"[{total}/{len(dataset)}] correct={is_correct} "
                    f"acc={correct / total:.4f} "
                    f"batch_gen={batch_elapsed / len(predictions):.4f}s "
                    f"running_avg_gen={total_generation_time / total:.4f}s",
                    flush=True,
                )
                print("question:", question, flush=True)
                print("gold:", gold, flush=True)
                print("pred:", pred, flush=True)
                print("response:", prediction, flush=True)

    wall_time = time.perf_counter() - wall_start
    stats = build_stats(
        args=args,
        dataset_path=dataset_source,
        device=device,
        few_shot_examples=few_shot_examples,
        start=start,
        total=total,
        correct=correct,
        total_generation_time=total_generation_time,
        wall_time=wall_time,
        output_path=output_path,
    )
    save_json(stats_path, stats)

    print("=" * 80, flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
