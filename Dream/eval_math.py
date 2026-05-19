import argparse
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Dream-v0-Base-7B")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/math",
        help=(
            "Optional local MATH dataset path. Supports load_from_disk directories "
            "or json/jsonl files. If omitted, the script tries common Hugging Face "
            "MATH dataset identifiers."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fewshot_split", type=str, default="train")
    parser.add_argument("--n_shot", type=int, default=4)
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
    parser.add_argument("--output_path", type=str, default="math_results.jsonl")
    parser.add_argument("--stats_path", type=str, default="math_stats.json")
    return parser.parse_args()


def resolve_mode_name(args):
    if args.focus_decode:
        return "focus_decode"
    if args.dual_cache:
        return "fast_dllm_dual_cache"
    if args.use_cache:
        return "fast_dllm_prefix_cache"
    return "baseline"


def load_json_split(dataset_path: Path, split_name: str):
    if dataset_path.is_dir():
        split_jsonl = dataset_path / f"{split_name}.jsonl"
        split_json = dataset_path / f"{split_name}.json"
        if split_jsonl.exists() or split_json.exists():
            split_file = split_jsonl if split_jsonl.exists() else split_json
            return load_dataset("json", data_files={split_name: str(split_file)}, split=split_name)

        dataset_obj = load_from_disk(str(dataset_path))
        if hasattr(dataset_obj, "keys"):
            if split_name not in dataset_obj:
                raise ValueError(f"Split '{split_name}' not found in dataset directory: {dataset_path}")
            return dataset_obj[split_name]
        return dataset_obj

    suffixes = "".join(dataset_path.suffixes).lower()
    if suffixes.endswith(".jsonl") or suffixes.endswith(".json"):
        return load_dataset("json", data_files={split_name: str(dataset_path)}, split=split_name)
    raise ValueError(
        "Unsupported --dataset_path format. Use a load_from_disk directory or a .json/.jsonl file."
    )


def load_math_split(args, split_name: str):
    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local dataset path not found: {dataset_path}")
        return load_json_split(dataset_path, split_name)

    candidates = [
        ("competition_math", None),
        ("hendrycks/competition_math", None),
        ("lighteval/MATH", None),
    ]
    errors = []
    for name, subset in candidates:
        try:
            kwargs = {"path": name, "split": split_name}
            if subset is not None:
                kwargs["name"] = subset
            return load_dataset(**kwargs)
        except Exception as exc:  # pragma: no cover - best effort fallback chain
            errors.append(f"{name}[{split_name}]: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(
        "Unable to load MATH split. Pass --dataset_path to a local dataset or ensure one of the "
        f"default Hugging Face datasets is available.\n{joined}"
    )


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def resolve_mask_token_id(model, tokenizer):
    candidates = [
        getattr(getattr(model, "generation_config", None), "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError("Unable to resolve mask_token_id.")


def normalize_example(example: Dict[str, Any], fallback_id: int) -> Dict[str, Any]:
    problem = example.get("problem") or example.get("question") or example.get("prompt") or ""
    solution = example.get("solution") or example.get("answer") or example.get("response") or ""
    task_id = example.get("task_id") or example.get("id") or example.get("unique_id") or fallback_id
    return {
        "task_id": str(task_id),
        "problem": str(problem).strip(),
        "solution": str(solution).strip(),
        "raw": example,
    }


def extract_last_boxed(text: str) -> Optional[str]:
    start = text.rfind("\\boxed{")
    token = "\\boxed{"
    if start == -1:
        start = text.rfind("\\fbox{")
        token = "\\fbox{"
    if start == -1:
        return None

    depth = 1
    current = start + len(token)
    pieces: List[str] = []
    while current < len(text):
        char = text[current]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(pieces).strip()
        pieces.append(char)
        current += 1
    return None


def strip_outer_braces(text: str) -> str:
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        balanced = True
        for idx, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and idx != len(text) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        text = text[1:-1].strip()
    return text


def normalize_number(text: str) -> Optional[str]:
    candidate = text.strip().replace(",", "")
    try:
        if "/" in candidate and re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", candidate):
            frac = Fraction(candidate.replace(" ", ""))
            return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        value = Decimal(candidate)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None
    if value == value.to_integral():
        return str(int(value))
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


def normalize_math_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None
    cleaned = cleaned.strip("$")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\!", "").replace("\\,", "")
    cleaned = cleaned.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.rstrip(".")
    cleaned = strip_outer_braces(cleaned)
    cleaned = cleaned.replace("{,}", ",")

    numeric = normalize_number(cleaned)
    if numeric is not None:
        return numeric
    return cleaned or None


def extract_prediction(text: str) -> Optional[str]:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        normalized = normalize_math_answer(boxed)
        if normalized is not None:
            return normalized

    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        normalized = normalize_math_answer(hash_match.group(1))
        if normalized is not None:
            return normalized

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        normalized = normalize_math_answer(lines[-1])
        if normalized is not None:
            return normalized

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    if matches:
        return normalize_math_answer(matches[-1])
    return None


def extract_reference(example: Dict[str, Any]) -> Optional[str]:
    explicit = (
        example.get("answer")
        or example.get("final_answer")
        or example.get("target")
        or example.get("label")
    )
    if explicit:
        normalized = normalize_math_answer(str(explicit))
        if normalized is not None:
            return normalized

    solution = example.get("solution") or example.get("answer") or ""
    return extract_prediction(str(solution))


def build_few_shot_prefix(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return ""

    parts = [
        "Solve the following competition math problems carefully. "
        "Show the reasoning and put the final answer inside \\boxed{}.\n\n"
    ]
    for idx, example in enumerate(examples, start=1):
        parts.append(f"Example {idx}:\n")
        parts.append(f"Problem: {example['problem']}\n\n")
        parts.append(f"Solution: {example['solution']}\n\n")
    return "".join(parts)


def build_messages(problem: str, few_shot_prefix: str = ""):
    prompt = few_shot_prefix
    if few_shot_prefix:
        prompt += "Now solve the next problem.\n\n"
    prompt += (
        f"Problem: {problem.strip()}\n\n"
        "Solution: Please reason step by step and end with the final answer in \\boxed{}."
    )
    return [{"role": "user", "content": prompt}]


@torch.no_grad()
def generate_batch(model, tokenizer, batch_examples, few_shot_prefix, device, args):
    messages = [build_messages(example["problem"], few_shot_prefix) for example in batch_examples]
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

    mask_token_id = resolve_mask_token_id(model, tokenizer)
    responses = []
    generated_tokens = []
    for seq, prompt_len in zip(outputs.sequences, prompt_lengths):
        gen_ids = seq[prompt_len:].tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        if tokenizer.eos_token:
            text = text.split(tokenizer.eos_token)[0]
        responses.append(text.strip())
        generated_tokens.append(sum(1 for token_id in gen_ids if int(token_id) != mask_token_id))
    return responses, generated_tokens


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
    total_generated_tokens,
    wall_time,
    output_path,
):
    effective_use_cache = bool(args.use_cache or args.dual_cache)
    return {
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "task": "math",
        "mode": resolve_mode_name(args),
        "model_path": args.model_path,
        "device": device,
        "split": args.split,
        "fewshot_split": args.fewshot_split,
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
        "total_generated_tokens": total_generated_tokens,
        "token_per_second": (
            total_generated_tokens / total_generation_time if total_generation_time > 0 else 0.0
        ),
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
    dataset = load_math_split(args, args.split)
    normalized_dataset = [normalize_example(dataset[i], i) for i in range(len(dataset))]

    few_shot_examples = []
    if args.n_shot > 0:
        few_shot_dataset = load_math_split(args, args.fewshot_split)
        shot_count = min(args.n_shot, len(few_shot_dataset))
        few_shot_examples = [normalize_example(few_shot_dataset[i], i) for i in range(shot_count)]
    few_shot_prefix = build_few_shot_prefix(few_shot_examples)

    total_len = len(normalized_dataset)
    start = max(args.start, 0)
    end = total_len if args.end is None else min(args.end, total_len)
    eval_examples = normalized_dataset[start:end]
    if args.max_samples is not None:
        eval_examples = eval_examples[: args.max_samples]

    print(f"Evaluating {len(eval_examples)} samples from math[{args.split}]", flush=True)
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
    total_generated_tokens = 0
    total_batches = (len(eval_examples) + args.batch_size - 1) // args.batch_size
    wall_start = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, len(eval_examples), args.batch_size):
            batch_examples = eval_examples[batch_start : batch_start + args.batch_size]
            batch_id = batch_start // args.batch_size + 1

            print(
                f"Starting batch {batch_id}/{total_batches} "
                f"(samples {batch_start}..{batch_start + len(batch_examples) - 1})",
                flush=True,
            )

            if device == "cuda":
                torch.cuda.synchronize()
            batch_t0 = time.perf_counter()
            predictions, generated_tokens = generate_batch(
                model,
                tokenizer,
                batch_examples,
                few_shot_prefix,
                device,
                args,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - batch_t0
            total_generation_time += batch_elapsed
            total_generated_tokens += sum(generated_tokens)

            for example, prediction, sample_generated_tokens in zip(
                batch_examples, predictions, generated_tokens
            ):
                gold = extract_reference(example["raw"])
                pred = extract_prediction(prediction)
                is_correct = pred is not None and pred == gold

                total += 1
                correct += int(is_correct)

                record = {
                    "index": start + total - 1,
                    "task_id": example["task_id"],
                    "problem": example["problem"],
                    "reference_solution": example["solution"],
                    "gold": gold,
                    "prediction": prediction,
                    "pred": pred,
                    "generated_tokens": sample_generated_tokens,
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
                    total_generated_tokens=total_generated_tokens,
                    wall_time=time.perf_counter() - wall_start,
                    output_path=output_path,
                )
                save_json(stats_path, running_stats)

                print("=" * 80, flush=True)
                print(
                    f"[{total}/{len(eval_examples)}] correct={is_correct} "
                    f"acc={correct / total:.4f} "
                    f"batch_gen={batch_elapsed / len(predictions):.4f}s "
                    f"running_avg_gen={total_generation_time / total:.4f}s "
                    f"tok/s={running_stats['token_per_second']:.2f}",
                    flush=True,
                )
                print("task_id:", example["task_id"], flush=True)
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
        total_generated_tokens=total_generated_tokens,
        wall_time=wall_time,
        output_path=output_path,
    )
    save_json(stats_path, stats)

    print("=" * 80, flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
