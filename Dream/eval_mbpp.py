import argparse
import ast
import json
import os
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import evaluate as hf_evaluate
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def refine_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def syntax_check(code: str, verbose: bool = False) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 150:
        lines = lines[:150]

    max_valid_lines = 0
    max_valid_snippet = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i : j + 1])
            if syntax_check(current_snippet):
                valid_line_count = sum(1 for line in lines[i : j + 1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = current_snippet
    return max_valid_snippet


def get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps: Set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    deps.add(child.attr)
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))


def fastdllm_sanitize(text: str, entrypoint: Optional[str] = None) -> str:
    text = refine_text(text)
    code = extract_longest_valid_code(text)
    if not code.strip():
        return ""
    tree = ast.parse(code)

    definitions = {}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = ("class", node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)

    reachable = set(definitions)
    if entrypoint:
        name2deps = get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))
    for name, (_, node) in definitions.items():
        if name in reachable:
            sanitized_output.append(ast.unparse(node))
    return "\n".join(sanitized_output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Dream-v0-Base-7B")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/mbpp",
        help=(
            "Optional local MBPP dataset path. Supports load_from_disk directories "
            "or json/jsonl files. If omitted, the script tries common Hugging Face "
            "MBPP dataset identifiers."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fewshot_split", type=str, default="prompt")
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_layer", type=int, default=3)
    parser.add_argument("--focus_topk", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="mbpp_results.jsonl")
    parser.add_argument("--stats_path", type=str, default="mbpp_stats.json")
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action="store_true",
        help="Required because MBPP evaluation executes model-generated Python.",
    )
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--escape_until", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_mode_name(args):
    if args.focus_decode:
        return "focus_decode"
    if args.dual_cache:
        return "fast_dllm_dual_cache"
    if args.use_cache:
        return "fast_dllm_prefix_cache"
    return "baseline"


def effective_focus_params(args):
    if args.focus_decode:
        return args.focus_layer, args.focus_topk
    return 0, 0


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


def load_mbpp_split(args, split_name: str):
    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local dataset path not found: {dataset_path}")
        return load_json_split(dataset_path, split_name)

    candidates = ["mbpp", "google-research-datasets/mbpp"]
    errors = []
    for name in candidates:
        try:
            return load_dataset(name, split=split_name)
        except Exception as exc:  # pragma: no cover - best effort fallback chain
            errors.append(f"{name}[{split_name}]: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(
        "Unable to load MBPP split. Pass --dataset_path to a local dataset or ensure one of the "
        f"default Hugging Face datasets is available.\n{joined}"
    )


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def synchronize_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def maybe_parse_test_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [line.strip() for line in stripped.splitlines() if line.strip()]
    return [str(value).strip()]


def normalize_example(example: Dict[str, Any], fallback_id: int) -> Dict[str, Any]:
    text = example.get("text") or example.get("prompt") or example.get("question") or ""
    code = (
        example.get("code")
        or example.get("canonical_solution")
        or example.get("reference_code")
        or ""
    )
    test_setup_code = example.get("test_setup_code") or ""
    test_list = maybe_parse_test_list(example.get("test_list") or example.get("tests") or example.get("test"))
    challenge_test_list = maybe_parse_test_list(example.get("challenge_test_list"))
    task_id = example.get("task_id") or example.get("id") or fallback_id
    return {
        "task_id": str(task_id),
        "text": str(text).strip(),
        "code": str(code).strip(),
        "test_setup_code": str(test_setup_code).strip(),
        "test_list": test_list,
        "challenge_test_list": challenge_test_list,
        "raw": example,
    }


def build_few_shot_prefix(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return ""

    parts = ["Write Python code that solves each task. Return only valid Python code.\n\n"]
    for idx, example in enumerate(examples, start=1):
        parts.append(f"Example {idx}:\n")
        parts.append(f"Task: {example['text']}\n")
        if example["test_list"]:
            parts.append("Tests:\n")
            for test in example["test_list"]:
                parts.append(f"{test}\n")
        parts.append("\nAnswer:\n```python\n")
        parts.append(example["code"].rstrip() + "\n")
        parts.append("```\n\n")
    return "".join(parts)


def build_prompt(example: Dict[str, Any], few_shot_prefix: str) -> str:
    prompt = few_shot_prefix
    if few_shot_prefix:
        prompt += "Now solve the next task.\n\n"
    prompt += f"Task: {example['text']}\n"
    if example["test_list"]:
        prompt += "Tests:\n"
        for test in example["test_list"]:
            prompt += f"{test}\n"
    prompt += "\nReturn only valid Python code."
    return prompt


def build_inputs(tokenizer, prompt, device, add_bos_token):
    if add_bos_token and tokenizer.bos_token:
        prompt = tokenizer.bos_token + prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def trim_completion_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    stop_patterns = [
        r"\n```",
        r"\nclass\s+",
        r"\ndef\s+",
        r"\nif __name__",
        r"\nprint\(",
        r"\n\s*Explanation\s*:",
        r"\n\s*This code\b",
    ]
    end = len(text)
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match is not None:
            end = min(end, match.start())
    return text[:end].rstrip()


def extract_fastdllm_code_block(text: str) -> str:
    text = text.replace("\r\n", "\n")
    return text.split("```python\n", 1)[-1].split("```", 1)[0]


def postprocess_completion(raw_generation: str, args) -> Tuple[str, str, str]:
    processed_generation = raw_generation if args.escape_until else trim_completion_text(raw_generation)
    extracted_code = extract_fastdllm_code_block(processed_generation)
    candidate = extracted_code if extracted_code.strip() else processed_generation
    completion = fastdllm_sanitize(candidate)
    if not completion.strip():
        completion = trim_completion_text(candidate)
    return completion, extracted_code, processed_generation


def generate_completion(model, tokenizer, example, few_shot_prefix, args, device):
    prompt = build_prompt(example, few_shot_prefix)
    input_ids, attention_mask = build_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        add_bos_token=args.add_bos_token,
    )

    synchronize_device(device)
    start_time = time.perf_counter()
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        block_length=args.block_length,
        use_cache=args.use_cache,
        dual_cache=args.dual_cache,
        alg=args.alg,
        alg_temp=args.alg_temp,
        threshold=args.threshold,
        gamma=args.gamma,
        focus_decode=args.focus_decode,
        focus_layer=args.focus_layer,
        focus_topk=args.focus_topk,
    )
    synchronize_device(device)
    elapsed = time.perf_counter() - start_time

    generated_token_ids = output.sequences[0, input_ids.shape[1] :].tolist()
    mask_token_id = resolve_mask_token_id(model, tokenizer)
    generated_tokens = sum(1 for token_id in generated_token_ids if int(token_id) != mask_token_id)
    raw_generation = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    if tokenizer.eos_token:
        raw_generation = raw_generation.split(tokenizer.eos_token)[0]

    completion, extracted_code, processed_generation = postprocess_completion(raw_generation, args)
    return completion, raw_generation, processed_generation, extracted_code, elapsed, generated_tokens, prompt


def build_reference(example: Dict[str, Any]) -> str:
    parts = []
    if example["test_setup_code"]:
        parts.append(example["test_setup_code"].rstrip())
    all_tests = example["test_list"][:]
    for test in example["challenge_test_list"]:
        if test not in all_tests:
            all_tests.append(test)
    parts.extend(test.rstrip() for test in all_tests if test.strip())
    return "\n".join(parts).rstrip() + "\n"


def extract_metric_detail(results):
    task_results = results.get(0, [])
    if not task_results:
        return {"passed": False, "result": "no_result"}
    task_results = sorted(task_results, key=lambda item: item[0])
    return task_results[0][1]


def evaluate_problem(metric, completion, reference, args):
    pass_at_k, results = metric.compute(
        references=[reference],
        predictions=[[completion]],
        k=[1],
        num_workers=args.num_workers,
        timeout=args.timeout,
    )
    detail = extract_metric_detail(results)
    pass_at_1 = float(pass_at_k.get("pass@1", 0.0))
    return pass_at_1, bool(detail.get("passed", False)), detail


def require_unsafe_code_confirmation(args):
    if not args.confirm_run_unsafe_code:
        raise ValueError(
            "MBPP evaluation executes model-generated Python. "
            "Re-run with --confirm_run_unsafe_code after confirming the environment is sandboxed."
        )
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_stats(
    *,
    args,
    dataset_path,
    device,
    dtype,
    examples,
    few_shot_examples,
    start,
    passed,
    total_generate_time,
    total_generated_tokens,
    total_eval_time,
    wall_time,
    output_path,
    completed,
):
    pass_at_1 = passed / completed if completed else 0.0
    layer, topk = effective_focus_params(args)
    return {
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "task": "mbpp",
        "model_path": args.model_path,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "split": args.split,
        "fewshot_split": args.fewshot_split,
        "n_shot": len(few_shot_examples),
        "mode": resolve_mode_name(args),
        "num_problems": len(examples),
        "completed": completed,
        "remaining": len(examples) - completed,
        "passed": passed,
        "pass_at_1": pass_at_1,
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "threshold": args.threshold if args.alg == "confidence_threshold" else None,
        "gamma": args.gamma,
        "use_cache": bool(args.use_cache),
        "dual_cache": bool(args.dual_cache),
        "focus_decode": bool(args.focus_decode),
        "focus_layer": layer,
        "focus_topk": topk,
        "start": start,
        "end": start + completed,
        "total_generated_tokens": total_generated_tokens,
        "tokens_per_second": (
            total_generated_tokens / total_generate_time if total_generate_time > 0 else 0.0
        ),
        "avg_generated_tokens": total_generated_tokens / completed if completed else 0.0,
        "total_generate_time_s": total_generate_time,
        "avg_generate_time_s": total_generate_time / completed if completed else 0.0,
        "total_eval_time_s": total_eval_time,
        "avg_eval_time_s": total_eval_time / completed if completed else 0.0,
        "wall_time_s": wall_time,
        "timeout": args.timeout,
        "num_workers": args.num_workers,
        "output_path": str(output_path),
    }


def main():
    args = parse_args()
    require_unsafe_code_confirmation(args)
    set_seed(args.seed)

    if args.focus_decode:
        if not args.dual_cache or not args.use_cache:
            raise ValueError("focus_decode evaluation requires both --use_cache and --dual_cache.")
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
        f"use_cache={args.use_cache}, "
        f"dual_cache={args.dual_cache}, "
        f"focus_decode={args.focus_decode}, "
        f"focus_layer={args.focus_layer}, "
        f"focus_topk={args.focus_topk}",
        flush=True,
    )

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(device).eval()

    dataset_source = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else None
    dataset = load_mbpp_split(args, args.split)
    examples = [normalize_example(dataset[i], i) for i in range(len(dataset))]

    few_shot_examples = []
    if args.n_shot > 0:
        few_shot_dataset = load_mbpp_split(args, args.fewshot_split)
        shot_count = min(args.n_shot, len(few_shot_dataset))
        few_shot_examples = [normalize_example(few_shot_dataset[i], i) for i in range(shot_count)]
    few_shot_prefix = build_few_shot_prefix(few_shot_examples)

    total_len = len(examples)
    start = max(args.start, 0)
    end = total_len if args.end is None else min(args.end, total_len)
    eval_examples = examples[start:end]
    if args.max_samples is not None:
        eval_examples = eval_examples[: args.max_samples]

    print(f"Evaluating {len(eval_examples)} samples from mbpp[{args.split}]", flush=True)
    print(f"n_shot: {len(few_shot_examples)}", flush=True)

    output_path = Path(args.output_path)
    stats_path = Path(args.stats_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.parent != Path("."):
        stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    metric = hf_evaluate.load("code_eval")
    total_generate_time = 0.0
    total_generated_tokens = 0
    total_eval_time = 0.0
    passed = 0
    wall_start = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, example in enumerate(eval_examples):
            (
                completion,
                raw_generation,
                processed_generation,
                extracted_code,
                gen_time,
                generated_tokens,
                prompt,
            ) = generate_completion(
                model=model,
                tokenizer=tokenizer,
                example=example,
                few_shot_prefix=few_shot_prefix,
                args=args,
                device=device,
            )

            eval_start = time.perf_counter()
            pass_at_1, ok, detail = evaluate_problem(
                metric=metric,
                completion=completion,
                reference=build_reference(example),
                args=args,
            )
            eval_time = time.perf_counter() - eval_start

            total_generate_time += gen_time
            total_generated_tokens += generated_tokens
            total_eval_time += eval_time
            passed += int(ok)

            row = {
                "index": start + idx,
                "task_id": example["task_id"],
                "passed": ok,
                "pass_at_1": pass_at_1,
                "generate_time_s": gen_time,
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / gen_time if gen_time > 0 else 0.0,
                "eval_time_s": eval_time,
                "result": detail.get("result"),
                "text": example["text"],
                "test_list": example["test_list"],
                "challenge_test_list": example["challenge_test_list"],
                "reference_code": example["code"],
                "completion": completion,
                "extracted_code": extracted_code,
                "processed_generation": processed_generation,
                "raw_generation": raw_generation,
                "prompt": prompt,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            running_stats = build_stats(
                args=args,
                dataset_path=dataset_source,
                device=device,
                dtype=dtype,
                examples=eval_examples,
                few_shot_examples=few_shot_examples,
                start=start,
                passed=passed,
                total_generate_time=total_generate_time,
                total_generated_tokens=total_generated_tokens,
                total_eval_time=total_eval_time,
                wall_time=time.perf_counter() - wall_start,
                output_path=output_path,
                completed=idx + 1,
            )
            save_json(stats_path, running_stats)

            print(
                f"[{idx + 1}/{len(eval_examples)}] {example['task_id']} "
                f"passed={ok} pass@1={pass_at_1:.4f} "
                f"gen={gen_time:.4f}s eval={eval_time:.4f}s "
                f"generated_tokens={generated_tokens} "
                f"tok/s={running_stats['tokens_per_second']:.2f} "
                f"running_pass@1={passed / (idx + 1):.4f}",
                flush=True,
            )

    wall_time = time.perf_counter() - wall_start
    stats = build_stats(
        args=args,
        dataset_path=dataset_source,
        device=device,
        dtype=dtype,
        examples=eval_examples,
        few_shot_examples=few_shot_examples,
        start=start,
        passed=passed,
        total_generate_time=total_generate_time,
        total_generated_tokens=total_generated_tokens,
        total_eval_time=total_eval_time,
        wall_time=wall_time,
        output_path=output_path,
        completed=len(eval_examples),
    )
    save_json(stats_path, stats)

    print("=" * 80, flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
