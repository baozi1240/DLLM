import argparse
import csv
import gzip
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
DEFAULT_HUMANEVAL_DATASET_PATH = "data/HumanEval.jsonl.gz"
DEFAULT_GSM8K_DATASET_PATH = "/home/xuefeng/.cache/opencompass/data/gsm8k"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Dream diffusion generation, count newly decoded tokens at each "
            "history step, and plot a line chart of ordinary decode throughput."
        )
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "gsm8k"])
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Local dataset path. For humaneval, supports .jsonl/.jsonl.gz/.json. "
            "For gsm8k, supports a directory containing <split>.jsonl/.json, "
            "a load_from_disk directory, or a .json/.jsonl file."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_index", type=int, default=0)
    parser.add_argument("--download_if_missing", action="store_true")
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--alg",
        type=str,
        default="confidence_threshold",
        choices=["origin", "entropy", "maskgit_plus", "topk_margin", "confidence_threshold"],
    )
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_layer", type=int, default=3)
    parser.add_argument("--focus_topk", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="stats/decode_step_stats")
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--show_time", action="store_true")
    return parser.parse_args()


def default_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Dream-v0-Base-7B")


def sync_local_transformers_module_cache(model_path):
    if not os.path.isdir(model_path):
        return
    try:
        from transformers.dynamic_module_utils import get_cached_module_file

        get_cached_module_file(model_path, "generation_utils.py", local_files_only=True)
    except Exception as exc:
        print(f"Warning: unable to refresh transformers dynamic module cache: {exc}")


def find_local_humaneval():
    candidates = [
        DEFAULT_HUMANEVAL_DATASET_PATH,
        "data/HumanEval.jsonl",
        str(Path.home() / ".cache" / "HumanEval.jsonl.gz"),
        str(Path.home() / ".cache" / "HumanEval.jsonl"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def ensure_humaneval_dataset(dataset_path, download_if_missing):
    dataset_path = dataset_path or DEFAULT_HUMANEVAL_DATASET_PATH
    abs_path = os.path.abspath(dataset_path)
    if os.path.exists(abs_path):
        return abs_path

    local_found = find_local_humaneval()
    if local_found is not None:
        return local_found

    if not download_if_missing:
        raise FileNotFoundError(
            f"HumanEval dataset not found at {abs_path}. Pass --prompt or --download_if_missing."
        )

    import urllib.request

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    urllib.request.urlretrieve(HUMANEVAL_URL, abs_path)
    return abs_path


def load_json_records(dataset_path):
    open_fn = gzip.open if dataset_path.endswith(".gz") else open
    with open_fn(dataset_path, "rt", encoding="utf-8") as f:
        if dataset_path.endswith(".json") and not dataset_path.endswith(".jsonl"):
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("data", "test", "train", "validation"):
                    if isinstance(data.get(key), list):
                        return data[key]
            raise ValueError(f"Unsupported JSON dataset structure in {dataset_path}")

        records = []
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def load_humaneval(dataset_path):
    return load_json_records(dataset_path)


def resolve_gsm8k_dataset_path(dataset_path):
    if dataset_path is not None:
        return Path(dataset_path).expanduser().resolve()
    return Path(DEFAULT_GSM8K_DATASET_PATH).expanduser().resolve()


def load_gsm8k(dataset_path, split):
    path = resolve_gsm8k_dataset_path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"GSM8K dataset not found at {path}. Pass --dataset_path with a local GSM8K path."
        )

    if path.is_dir():
        split_jsonl = path / f"{split}.jsonl"
        split_json = path / f"{split}.json"
        if split_jsonl.exists() or split_json.exists():
            return load_json_records(str(split_jsonl if split_jsonl.exists() else split_json))

        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "Loading a GSM8K load_from_disk directory requires the `datasets` package."
            ) from exc

        dataset_obj = load_from_disk(str(path))
        if hasattr(dataset_obj, "keys"):
            if split not in dataset_obj:
                raise ValueError(f"Split '{split}' not found in local GSM8K dataset: {path}")
            return list(dataset_obj[split])
        return list(dataset_obj)

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl") or suffixes.endswith(".json") or suffixes.endswith(".jsonl.gz"):
        records = load_json_records(str(path))
        if isinstance(records, dict):
            records = records[split]
        return records

    raise ValueError(
        "Unsupported GSM8K --dataset_path format. Use a directory, .json, .jsonl, or .jsonl.gz file."
    )


def build_gsm8k_prompt(example):
    question = example.get("question")
    if question is None:
        raise KeyError("GSM8K example must contain a `question` field.")
    return (
        f"Question: {question.strip()}\n\n"
        "Please solve the math word problem step by step. "
        "End your response with '#### <answer>'."
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


def to_cpu_2d_tokens(tokens):
    tokens = tokens.detach().to("cpu")
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    return tokens


def normalize_history(history):
    if history is None:
        return []
    if torch.is_tensor(history):
        return [history]
    return list(history)


def token_text(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def build_token_observation(tokenizer, token_id, generated_position, input_len):
    token_id = int(token_id)
    generated_position = int(generated_position)
    return {
        "generated_position": generated_position,
        "absolute_position": int(input_len + generated_position),
        "token_id": token_id,
        "token_text": token_text(tokenizer, token_id),
    }


def build_step_stats(history, input_ids, sequences, mask_token_id, tokenizer):
    history = normalize_history(history)
    input_ids_cpu = to_cpu_2d_tokens(input_ids)
    sequences_cpu = to_cpu_2d_tokens(sequences)
    input_len = input_ids_cpu.shape[1]
    generated_length = sequences_cpu.shape[1] - input_len
    if generated_length < 0:
        raise ValueError("sequences is shorter than input_ids.")

    prev_generated = torch.full(
        (sequences_cpu.shape[0], generated_length),
        int(mask_token_id),
        dtype=sequences_cpu.dtype,
    )
    records = []

    for step_idx, state in enumerate(history):
        state_cpu = to_cpu_2d_tokens(state)
        current_generated = state_cpu[:, input_len : input_len + generated_length]
        if current_generated.shape != prev_generated.shape:
            raise ValueError(
                "history state shape does not match final sequence shape: "
                f"history generated span={tuple(current_generated.shape)}, "
                f"expected={tuple(prev_generated.shape)}"
            )

        prev_mask = prev_generated == mask_token_id
        current_mask = current_generated == mask_token_id
        newly_decoded = prev_mask & ~current_mask
        changed_after_decode = (~prev_mask) & ~current_mask & (prev_generated != current_generated)

        for batch_idx in range(current_generated.shape[0]):
            positions = newly_decoded[batch_idx].nonzero(as_tuple=True)[0].tolist()
            changed_positions = changed_after_decode[batch_idx].nonzero(as_tuple=True)[0].tolist()
            decoded_tokens = [
                build_token_observation(
                    tokenizer,
                    current_generated[batch_idx, pos].item(),
                    pos,
                    input_len,
                )
                for pos in positions
            ]

            records.append(
                {
                    "step": int(step_idx),
                    "batch": int(batch_idx),
                    "new_tokens": int(len(positions)),
                    "cumulative_tokens": int((~current_mask[batch_idx]).sum().item()),
                    "remaining_masks": int(current_mask[batch_idx].sum().item()),
                    "positions": [int(pos) for pos in positions],
                    "decoded_tokens": decoded_tokens,
                    "changed_after_decode_positions": [int(pos) for pos in changed_positions],
                }
            )

        prev_generated = current_generated.clone()

    total_new_tokens = int(sum(record["new_tokens"] for record in records))
    active_steps = int(sum(1 for record in records if record["new_tokens"] > 0))
    num_history_steps = int(len(history))

    summary = {
        "num_history_steps": num_history_steps,
        "batch_size": int(sequences_cpu.shape[0]),
        "prompt_length": int(input_len),
        "generated_length": int(generated_length),
        "total_new_tokens": total_new_tokens,
        "average_tokens_per_history_step": (
            float(total_new_tokens / num_history_steps) if num_history_steps > 0 else 0.0
        ),
        "average_tokens_per_active_step": (
            float(total_new_tokens / active_steps) if active_steps > 0 else 0.0
        ),
        "active_decode_steps": active_steps,
        "final_remaining_masks": [
            int(((sequences_cpu[b, input_len:] == mask_token_id).sum()).item())
            for b in range(sequences_cpu.shape[0])
        ],
    }
    return records, summary


def make_output_stem(args, model_path):
    if args.output_prefix:
        return args.output_prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.normpath(model_path))
    focus = f"fd-{int(args.focus_decode)}_fl-{args.focus_layer}_ftk-{args.focus_topk}"
    cache = f"uc-{int(args.use_cache or args.dual_cache)}_dc-{int(args.dual_cache)}"
    block = args.block_length if args.block_length is not None else "none"
    steps = args.steps if args.steps is not None else args.max_new_tokens
    return (
        f"decode_step_stats_model-{model_name}_alg-{args.alg}_blk-{block}_"
        f"{cache}_{focus}_max-{args.max_new_tokens}_steps-{steps}_{timestamp}"
    )


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path, records):
    fieldnames = [
        "step",
        "batch",
        "new_tokens",
        "cumulative_tokens",
        "remaining_masks",
        "positions",
        "decoded_tokens",
        "changed_after_decode_positions",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in (
                "positions",
                "decoded_tokens",
                "changed_after_decode_positions",
            ):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def prepare_matplotlib_cache(output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    cache_dir = os.path.join(output_dir, ".matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)


def plot_decode_counts(records, output_path):
    prepare_matplotlib_cache(output_path)
    import matplotlib.pyplot as plt

    by_batch = defaultdict(list)
    for record in records:
        by_batch[int(record["batch"])].append(record)

    fig, ax = plt.subplots(figsize=(11, 5))
    for batch_idx in sorted(by_batch):
        batch_records = sorted(by_batch[batch_idx], key=lambda item: item["step"])
        ax.plot(
            [record["step"] for record in batch_records],
            [record["new_tokens"] for record in batch_records],
            marker="o",
            linewidth=1.6,
            markersize=3,
            label=f"batch {batch_idx}",
        )

    ax.set_title("Newly decoded tokens per diffusion step")
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Newly decoded tokens")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_decoded_positions(records, output_path):
    prepare_matplotlib_cache(output_path)
    import matplotlib.pyplot as plt

    points_by_batch = defaultdict(lambda: {"steps": [], "positions": []})

    for record in records:
        step = int(record["step"])
        batch_idx = int(record["batch"])
        for token in record.get("decoded_tokens", []):
            position = int(token["generated_position"])
            points_by_batch[batch_idx]["steps"].append(step)
            points_by_batch[batch_idx]["positions"].append(position)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plotted_any = False
    for batch_idx in sorted(points_by_batch):
        batch_points = points_by_batch[batch_idx]
        if not batch_points["steps"]:
            continue
        plotted_any = True
        ax.scatter(
            batch_points["steps"],
            batch_points["positions"],
            s=18,
            alpha=0.7,
            label=f"batch {batch_idx}",
        )

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No decoded positions recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_title("Decoded position ids per diffusion step")
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Decoded generated position id")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_config(args, device, dtype, model_path, prompt_source, steps):
    return {
        "model_path": model_path,
        "device": device,
        "dtype": str(dtype),
        "prompt_source": prompt_source,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "task_index": int(args.task_index),
        "max_new_tokens": int(args.max_new_tokens),
        "steps": int(steps),
        "block_length": args.block_length,
        "temperature": float(args.temperature),
        "top_p": args.top_p,
        "top_k": args.top_k,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "threshold": float(args.threshold),
        "gamma": float(args.gamma),
        "use_cache": bool(args.use_cache or args.dual_cache),
        "dual_cache": bool(args.dual_cache),
        "focus_decode": bool(args.focus_decode),
        "focus_layer": int(args.focus_layer),
        "focus_topk": int(args.focus_topk),
        "use_chat_template": bool(args.use_chat_template),
        "add_bos_token": bool(args.add_bos_token),
    }


def main():
    args = parse_args()
    use_cache = args.use_cache or args.dual_cache
    model_path = os.path.abspath(args.model_path) if args.model_path else default_model_path()
    device = select_device()
    dtype_by_device = {
        "cuda": torch.bfloat16,
        "mps": torch.float16,
        "cpu": torch.float32,
    }
    dtype = dtype_by_device[device]
    steps = args.steps if args.steps is not None else args.max_new_tokens

    print(f"Using device: {device} (dtype={dtype})")
    print(f"Model path: {model_path}")
    sync_local_transformers_module_cache(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()

    if args.prompt is not None:
        prompt = args.prompt
        prompt_source = {"kind": "cli_prompt"}
    else:
        if args.dataset == "humaneval":
            dataset_path = ensure_humaneval_dataset(args.dataset_path, args.download_if_missing)
            problems = load_humaneval(dataset_path)
            if not problems:
                raise ValueError(f"No samples found in HumanEval dataset: {dataset_path}")
            if args.task_index < 0 or args.task_index >= len(problems):
                raise ValueError(f"--task_index must be in [0, {len(problems) - 1}], got {args.task_index}")
            problem = problems[args.task_index]
            prompt = problem["prompt"]
            prompt_source = {
                "kind": "humaneval",
                "dataset_path": dataset_path,
                "task_index": int(args.task_index),
                "task_id": problem.get("task_id"),
            }
        elif args.dataset == "gsm8k":
            dataset_path = resolve_gsm8k_dataset_path(args.dataset_path)
            examples = load_gsm8k(args.dataset_path, args.split)
            if not examples:
                raise ValueError(f"No samples found in GSM8K dataset: {dataset_path}")
            if args.task_index < 0 or args.task_index >= len(examples):
                raise ValueError(f"--task_index must be in [0, {len(examples) - 1}], got {args.task_index}")
            example = examples[args.task_index]
            prompt = build_gsm8k_prompt(example)
            prompt_source = {
                "kind": "gsm8k",
                "dataset_path": str(dataset_path),
                "split": args.split,
                "task_index": int(args.task_index),
                "question": example.get("question"),
                "reference_answer": example.get("answer"),
            }
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
    else:
        if args.add_bos_token and tokenizer.bos_token:
            prompt = tokenizer.bos_token + prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

    synchronize_device(device)
    start_time = time.perf_counter()
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=True,
        return_dict_in_generate=True,
        steps=steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        block_length=args.block_length,
        use_cache=use_cache,
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
    elapsed_time = time.perf_counter() - start_time

    generations = [
        tokenizer.decode(g[len(p) :].tolist())
        for p, g in zip(input_ids, output.sequences)
    ]
    generation_text = generations[0]
    if tokenizer.eos_token:
        generation_text = generation_text.split(tokenizer.eos_token)[0]

    mask_token_id = resolve_mask_token_id(model, tokenizer)
    records, summary = build_step_stats(
        getattr(output, "history", None),
        input_ids,
        output.sequences,
        mask_token_id,
        tokenizer,
    )
    summary["elapsed_time_s"] = float(elapsed_time)
    summary["tokens_per_second"] = (
        float(summary["total_new_tokens"] / elapsed_time) if elapsed_time > 0 else None
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stem = make_output_stem(args, model_path)
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    png_path = os.path.join(args.output_dir, f"{stem}.png")
    positions_png_path = os.path.join(args.output_dir, f"{stem}_positions.png")

    payload = {
        "schema": "decode_step_stats_v2",
        "config": build_config(args, device, dtype, model_path, prompt_source, steps),
        "generation": generation_text,
        "decode_stats": {
            "summary": summary,
            "steps": records,
        },
    }
    write_json(json_path, payload)
    write_csv(csv_path, records)
    if not args.no_plot:
        plot_decode_counts(records, png_path)
        plot_decoded_positions(records, positions_png_path)
    print(
        "Decode throughput: "
        f"{summary['tokens_per_second']:.4f} tokens/s "
        f"({summary['total_new_tokens']} generated tokens)"
    )
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")
    if not args.no_plot:
        print(f"Plot saved to: {png_path}")
        print(f"Position plot saved to: {positions_png_path}")
    print(generation_text)


if __name__ == "__main__":
    main()
