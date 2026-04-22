import argparse
import ast
import gzip
import json
import os
import random
import re
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import evaluate as hf_evaluate
import torch
from transformers import AutoModel, AutoTokenizer


HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


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
    if len(lines) > 100:
        lines = lines[:100]

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
    # Inlined from Fast-dLLM's sanitize.py so this script is self-contained.
    text = refine_text(text)
    code = extract_longest_valid_code(text)
    tree = ast.parse(code)

    definitions = {}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = ("class", node)
        elif isinstance(node, ast.FunctionDef):
            if has_return_statement(node):
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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dream on HumanEval with a unified protocol so different "
            "generation/cache configurations remain directly comparable."
        )
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="data/HumanEval.jsonl.gz")
    parser.add_argument("--download_if_missing", action="store_true")
    parser.add_argument("--output_dir", type=str, default="humaneval_results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)

    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_layer", type=int, default=3)
    parser.add_argument("--focus_topk", type=int, default=8)

    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action="store_true",
        help=(
            "Required for code_eval. This mirrors Fast-dLLM / lm-eval's "
            "confirm_run_unsafe_code safeguard for HumanEval."
        ),
    )

    # Fast-dLLM compatibility: humaneval runs there use add_bos_token=true.
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    # Fast-dLLM compatibility: humaneval runs there use escape_until=true so the
    # postprocessor sees the full raw generation instead of a stop-sequence-truncated one.
    parser.add_argument("--escape_until", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def default_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Dream-v0-Base-7B")


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


def find_local_humaneval():
    candidates = [
        "data/HumanEval.jsonl.gz",
        "data/HumanEval.jsonl",
        str(Path.home() / ".cache" / "HumanEval.jsonl.gz"),
        str(Path.home() / ".cache" / "HumanEval.jsonl"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def ensure_dataset(dataset_path, download_if_missing):
    abs_path = os.path.abspath(dataset_path)
    if os.path.exists(abs_path):
        return abs_path

    local_found = find_local_humaneval()
    if local_found is not None:
        return local_found

    if not download_if_missing:
        raise FileNotFoundError(
            f"HumanEval dataset not found at {abs_path}. "
            "Pass --download_if_missing to fetch it."
        )

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    urllib.request.urlretrieve(HUMANEVAL_URL, abs_path)
    return abs_path


def load_humaneval(dataset_path):
    open_fn = gzip.open if dataset_path.endswith(".gz") else open
    problems = []
    with open_fn(dataset_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def resolve_mode_name(args):
    if args.focus_decode:
        if not args.use_cache or not args.dual_cache:
            raise ValueError("focus_decode=True requires both use_cache=True and dual_cache=True")
        return "focus_dual_cache"
    if args.dual_cache:
        if not args.use_cache:
            raise ValueError("dual_cache=True requires use_cache=True for a meaningful comparison")
        return "fast_dllm_dual_cache"
    if args.use_cache:
        return "fast_dllm_prefix_cache"
    return "baseline"


def effective_steps(args):
    return args.steps if args.steps is not None else args.max_new_tokens


def effective_focus_params(args):
    if args.focus_decode:
        return args.focus_layer, args.focus_topk
    return 0, 0


def format_float_tag(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def build_run_tag(args):
    mode_name = resolve_mode_name(args)
    layer, topk = effective_focus_params(args)
    parts = [
        f"mode{mode_name}",
        f"alg{args.alg}",
        f"len{args.max_new_tokens}",
        f"steps{effective_steps(args)}",
        f"blk{args.block_length}",
        f"temp{format_float_tag(args.temperature)}",
        f"topp{format_float_tag(args.top_p)}",
        f"bos{int(bool(args.add_bos_token))}",
        f"esc{int(bool(args.escape_until))}",
        f"seed{args.seed}",
        f"layer{layer}",
        f"topk{topk}",
    ]
    if args.top_k is not None:
        parts.append(f"topkgen{args.top_k}")
    if args.alg_temp != 0.0:
        parts.append(f"algtemp{format_float_tag(args.alg_temp)}")
    if args.alg == "confidence_threshold":
        parts.append(f"th{format_float_tag(args.threshold)}")
    return "_".join(parts)


def load_model_and_tokenizer(model_path, device):
    dtype_by_device = {
        "cuda": torch.bfloat16,
        "mps": torch.float16,
        "cpu": torch.float32,
    }
    dtype = dtype_by_device[device]
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()
    return model, tokenizer, dtype


def build_inputs(tokenizer, prompt, device, add_bos_token):
    # Fast-dLLM compatibility: prepend BOS in the same way as eval.py / eval_humaneval.sh.
    if add_bos_token and tokenizer.bos_token:
        prompt = tokenizer.bos_token + prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def trim_completion_text(text):
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    stop_patterns = [
        r"\n```",
        r"\nclass\s+",
        r"\ndef\s+",
        r"\nif __name__",
        r"\nprint\(",
        r"\n\s*This function\b",
        r"\n\s*This code\b",
        r"\n\s*Explanation\s*:",
        r"\n\s*The function\b",
    ]
    end = len(text)
    for pattern in stop_patterns:
        match = re.search(pattern, text)
        if match is not None:
            end = min(end, match.start())
    return text[:end].rstrip()


def extract_fastdllm_code_block(text):
    # Match Fast-dLLM postprocess_code.py exactly:
    # sample['resps'][0][0].split('```python\n', 1)[-1].split('```')[0]
    #
    # This intentionally keeps prefix code when the model emits a stray closing
    # fence without an opening ```python block, which happens in some HumanEval
    # generations and affects pass@1.
    text = text.replace("\r\n", "\n")
    return text.split("```python\n", 1)[-1].split("```", 1)[0]


def postprocess_completion(raw_generation, problem, args):
    processed_generation = raw_generation if args.escape_until else trim_completion_text(raw_generation)
    extracted_code = extract_fastdllm_code_block(processed_generation)

    # Fast-dLLM compatibility: reuse sanitize(prompt + generation, entry_point)
    # from dream/postprocess_code.py so every configuration is judged with the
    # exact same cleanup rule before HumanEval execution.
    completion = fastdllm_sanitize(
        problem["prompt"] + "\n" + extracted_code,
        problem["entry_point"],
    )
    return completion, extracted_code, processed_generation


def generate_completion(model, tokenizer, problem, args, device):
    prompt = problem["prompt"]
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
        steps=effective_steps(args),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        block_length=args.block_length,
        use_cache=args.use_cache,
        dual_cache=args.dual_cache,
        alg=args.alg,
        alg_temp=args.alg_temp,
        threshold=args.threshold,
        focus_decode=args.focus_decode,
        focus_layer=args.focus_layer,
        focus_topk=args.focus_topk,
    )
    synchronize_device(device)
    elapsed = time.perf_counter() - start_time

    raw_generation = tokenizer.decode(
        output.sequences[0, input_ids.shape[1] :].tolist(),
        skip_special_tokens=False,
    )
    if tokenizer.eos_token:
        raw_generation = raw_generation.split(tokenizer.eos_token)[0]

    completion, extracted_code, processed_generation = postprocess_completion(
        raw_generation=raw_generation,
        problem=problem,
        args=args,
    )
    return completion, raw_generation, processed_generation, extracted_code, elapsed


def build_reference(problem):
    return problem["test"].rstrip() + "\n" + f"check({problem['entry_point']})\n"


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
            "HumanEval needs to execute model-generated Python. "
            "Re-run with --confirm_run_unsafe_code after confirming the environment is sandboxed."
        )

    # Fast-dLLM compatibility: postprocess_code.py uses Hugging Face code_eval and
    # enables it via HF_ALLOW_CODE_EVAL=1 after the caller explicitly opts in.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(
    *,
    dataset_path,
    model_path,
    device,
    dtype,
    problems,
    passed,
    total_generate_time,
    total_eval_time,
    wall_time,
    args,
    completed,
):
    pass_at_1 = passed / completed if completed else 0.0
    mode_name = resolve_mode_name(args)
    steps = effective_steps(args)
    layer, topk = effective_focus_params(args)
    return {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "mode": mode_name,
        "num_problems": len(problems),
        "completed": completed,
        "remaining": len(problems) - completed,
        "passed": passed,
        "pass_at_1": pass_at_1,
        "total_generate_time_s": total_generate_time,
        "total_eval_time_s": total_eval_time,
        "wall_time_s": wall_time,
        "avg_generate_time_s": total_generate_time / completed if completed else 0.0,
        "avg_eval_time_s": total_eval_time / completed if completed else 0.0,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "steps": steps,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "alg": args.alg,
            "alg_temp": args.alg_temp,
            "threshold": args.threshold,
            "use_cache": bool(args.use_cache),
            "dual_cache": bool(args.dual_cache),
            "focus_decode": bool(args.focus_decode),
            "focus_layer": layer,
            "focus_topk": topk,
            "timeout": args.timeout,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "add_bos_token": bool(args.add_bos_token),
            "escape_until": bool(args.escape_until),
            "confirm_run_unsafe_code": bool(args.confirm_run_unsafe_code),
        },
        "evaluation_protocol": {
            "metric": "evaluate/code_eval",
            "per_task_samples": 1,
            "aggregate_pass_at_1": "mean(single-sample pass@1 over tasks)",
            "fastdllm_compat": {
                "add_bos_token": True,
                "escape_until": True,
                "sanitize": "Fast-dLLM dream/sanitize.py",
                "confirm_run_unsafe_code": True,
            },
        },
    }


def main():
    args = parse_args()
    require_unsafe_code_confirmation(args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    run_tag = build_run_tag(args)
    dataset_path = ensure_dataset(args.dataset_path, args.download_if_missing)
    problems = load_humaneval(dataset_path)
    if args.max_samples is not None:
        problems = problems[: args.max_samples]

    model_path = os.path.abspath(args.model_path) if args.model_path else default_model_path()
    device = select_device()
    model, tokenizer, dtype = load_model_and_tokenizer(model_path, device)
    metric = hf_evaluate.load("code_eval")

    summary_path = os.path.join(args.output_dir, f"humaneval_{run_tag}_summary.json")
    results_path = os.path.join(args.output_dir, f"humaneval_{run_tag}_results.jsonl")
    save_jsonl(results_path, [])

    print(f"Using device: {device} (dtype={dtype})")
    print(f"Model path (local): {model_path}")
    print(f"HumanEval dataset: {dataset_path}")
    print(f"Num problems: {len(problems)}")
    print(f"Run tag: {run_tag}")

    total_generate_time = 0.0
    total_eval_time = 0.0
    passed = 0
    wall_start = time.perf_counter()

    for idx, problem in enumerate(problems):
        completion, raw_generation, processed_generation, extracted_code, gen_time = generate_completion(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            args=args,
            device=device,
        )

        eval_start = time.perf_counter()
        pass_at_1, ok, detail = evaluate_problem(
            metric=metric,
            completion=completion,
            reference=build_reference(problem),
            args=args,
        )
        eval_time = time.perf_counter() - eval_start

        total_generate_time += gen_time
        total_eval_time += eval_time
        passed += int(ok)

        row = {
            "task_id": problem["task_id"],
            "passed": ok,
            "pass_at_1": pass_at_1,
            "generate_time_s": gen_time,
            "eval_time_s": eval_time,
            "result": detail.get("result"),
            "completion": completion,
            "extracted_code": extracted_code,
            "processed_generation": processed_generation,
            "raw_generation": raw_generation,
        }
        append_jsonl(results_path, row)

        save_json(
            summary_path,
            build_summary(
                dataset_path=dataset_path,
                model_path=model_path,
                device=device,
                dtype=dtype,
                problems=problems,
                passed=passed,
                total_generate_time=total_generate_time,
                total_eval_time=total_eval_time,
                wall_time=time.perf_counter() - wall_start,
                args=args,
                completed=idx + 1,
            ),
        )

        running_pass_at_1 = passed / (idx + 1)
        print(
            f"[{idx + 1}/{len(problems)}] {problem['task_id']} "
            f"passed={ok} pass@1={pass_at_1:.4f} "
            f"gen={gen_time:.4f}s eval={eval_time:.4f}s "
            f"running_pass@1={running_pass_at_1:.4f}"
        )

    wall_time = time.perf_counter() - wall_start
    summary = build_summary(
        dataset_path=dataset_path,
        model_path=model_path,
        device=device,
        dtype=dtype,
        problems=problems,
        passed=passed,
        total_generate_time=total_generate_time,
        total_eval_time=total_eval_time,
        wall_time=wall_time,
        args=args,
        completed=len(problems),
    )
    save_json(summary_path, summary)

    final_pass_at_1 = summary["pass_at_1"]
    print(f"pass@1: {final_pass_at_1:.4f} ({passed}/{len(problems)})")
    print(f"Total generation time: {total_generate_time:.4f}s")
    print(f"Total eval time: {total_eval_time:.4f}s")
    print(f"Wall time: {wall_time:.4f}s")
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
