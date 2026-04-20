import argparse
import csv
import json
import os
import time
from statistics import mean

import torch
from transformers import AutoModel, AutoTokenizer


PROMPT = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
    "and bakes muffins for her friends every day with four. She sells the remainder "
    "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
    "does she make every day at the farmers' market?"
)


MODES = [
    {
        "name": "baseline",
        "use_cache": False,
        "dual_cache": False,
        "focus_decode": False,
    },
    {
        "name": "fast_dllm_prefix_cache",
        "use_cache": True,
        "dual_cache": False,
        "focus_decode": False,
    },
    {
        "name": "fast_dllm_dual_cache",
        "use_cache": True,
        "dual_cache": True,
        "focus_decode": False,
    },
    {
        "name": "focus_dual_cache",
        "use_cache": True,
        "dual_cache": True,
        "focus_decode": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型目录；默认使用本仓库 models/Dream-v0-Base-7B",
    )
    parser.add_argument(
        "--gen_lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="扫描的 gen_length/max_new_tokens",
    )
    parser.add_argument(
        "--block_lengths",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="扫描的 block_length",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每组配置重复次数，默认 1",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=0,
        help="每组配置正式计时前的 warmup 次数",
    )
    parser.add_argument(
        "--focus_topk",
        type=int,
        default=8,
        help="focus 模式固定 focus_topk",
    )
    parser.add_argument(
        "--focus_layer",
        type=int,
        default=3,
        help="focus 模式固定 focus_layer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="结果输出目录",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=PROMPT,
        help="测试 prompt；默认与 demo_completion.py 一致",
    )
    parser.add_argument("--show_text", action="store_true", help="打印每组生成文本")
    return parser.parse_args()


def default_model_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "Dream-v0-Base-7B",
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def build_inputs(tokenizer, prompt_text, device):
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    return inputs.input_ids.to(device), inputs.attention_mask.to(device)


def run_generation(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    mode,
    gen_length,
    block_length,
    focus_layer,
    focus_topk,
    device,
):
    synchronize_device(device)
    start_time = time.perf_counter()
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=True,
        steps=gen_length,
        temperature=0.0,
        top_p=0.95,
        block_length=block_length,
        use_cache=mode["use_cache"],
        dual_cache=mode["dual_cache"],
        alg="entropy",
        alg_temp=0.0,
        focus_decode=mode["focus_decode"],
        focus_layer=focus_layer,
        focus_topk=focus_topk,
    )
    synchronize_device(device)
    elapsed = time.perf_counter() - start_time

    generated_text = tokenizer.decode(output.sequences[0, input_ids.shape[1]:].tolist())
    if tokenizer.eos_token:
        generated_text = generated_text.split(tokenizer.eos_token)[0]
    return elapsed, generated_text


def make_row(
    *,
    mode_name,
    use_cache,
    dual_cache,
    focus_decode,
    gen_length,
    block_length,
    repeat_idx,
    elapsed_s,
    tokens_per_s,
    generated_text,
):
    return {
        "mode": mode_name,
        "use_cache": int(use_cache),
        "dual_cache": int(dual_cache),
        "focus_decode": int(focus_decode),
        "gen_length": int(gen_length),
        "steps": int(gen_length),
        "block_length": int(block_length),
        "repeat_idx": int(repeat_idx),
        "elapsed_s": float(elapsed_s),
        "tokens_per_s": float(tokens_per_s),
        "generated_chars": len(generated_text),
        "generated_preview": generated_text[:160],
    }


def save_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["mode"], row["gen_length"], row["block_length"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (mode, gen_length, block_length), items in sorted(grouped.items()):
        elapsed_values = [item["elapsed_s"] for item in items]
        tps_values = [item["tokens_per_s"] for item in items]
        summary.append(
            {
                "mode": mode,
                "gen_length": gen_length,
                "steps": gen_length,
                "block_length": block_length,
                "runs": len(items),
                "elapsed_s_mean": mean(elapsed_values),
                "elapsed_s_min": min(elapsed_values),
                "elapsed_s_max": max(elapsed_values),
                "tokens_per_s_mean": mean(tps_values),
                "tokens_per_s_min": min(tps_values),
                "tokens_per_s_max": max(tps_values),
            }
        )
    return summary


def plot_results(summary_rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skip plotting: matplotlib not available ({exc})")
        return []

    generated_paths = []

    for metric, ylabel, filename in [
        ("elapsed_s_mean", "Latency (s)", "latency_by_mode.png"),
        ("tokens_per_s_mean", "Tokens / s", "throughput_by_mode.png"),
    ]:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 5),
            sharex=False,
            sharey=False,
        )

        for ax, block_length in zip(axes, sorted({row["block_length"] for row in summary_rows})):
            block_rows = [row for row in summary_rows if row["block_length"] == block_length]
            gen_lengths = sorted({row["gen_length"] for row in block_rows})
            for mode in [m["name"] for m in MODES]:
                mode_rows = [row for row in block_rows if row["mode"] == mode]
                if not mode_rows:
                    continue
                value_by_length = {row["gen_length"]: row[metric] for row in mode_rows}
                y = [value_by_length.get(length) for length in gen_lengths]
                ax.plot(gen_lengths, y, marker="o", label=mode)

            ax.set_title(f"block_length={block_length}")
            ax.set_xlabel("gen_length")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated_paths.append(out_path)

    return generated_paths


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    model_path = os.path.abspath(args.model_path) if args.model_path else default_model_path()
    device = select_device()
    model, tokenizer, dtype = load_model_and_tokenizer(model_path, device)
    input_ids, attention_mask = build_inputs(tokenizer, args.prompt_text, device)

    print(f"Using device: {device} (dtype={dtype})")
    print(f"Model path (local): {model_path}")
    print(f"gen_lengths={args.gen_lengths}, block_lengths={args.block_lengths}, repeats={args.repeats}")

    raw_rows = []
    generation_records = []

    for gen_length in args.gen_lengths:
        for block_length in args.block_lengths:
            for mode in MODES:
                print(
                    f"[run] mode={mode['name']} gen_length={gen_length} "
                    f"block_length={block_length} use_cache={mode['use_cache']} "
                    f"dual_cache={mode['dual_cache']} focus_decode={mode['focus_decode']}"
                )

                for _ in range(args.warmup_runs):
                    _elapsed, _text = run_generation(
                        model,
                        tokenizer,
                        input_ids,
                        attention_mask,
                        mode=mode,
                        gen_length=gen_length,
                        block_length=block_length,
                        focus_layer=args.focus_layer,
                        focus_topk=args.focus_topk,
                        device=device,
                    )
                    del _elapsed, _text
                    if device == "cuda":
                        torch.cuda.empty_cache()

                for repeat_idx in range(args.repeats):
                    elapsed_s, generated_text = run_generation(
                        model,
                        tokenizer,
                        input_ids,
                        attention_mask,
                        mode=mode,
                        gen_length=gen_length,
                        block_length=block_length,
                        focus_layer=args.focus_layer,
                        focus_topk=args.focus_topk,
                        device=device,
                    )
                    tokens_per_s = gen_length / elapsed_s if elapsed_s > 0 else 0.0
                    row = make_row(
                        mode_name=mode["name"],
                        use_cache=mode["use_cache"],
                        dual_cache=mode["dual_cache"],
                        focus_decode=mode["focus_decode"],
                        gen_length=gen_length,
                        block_length=block_length,
                        repeat_idx=repeat_idx,
                        elapsed_s=elapsed_s,
                        tokens_per_s=tokens_per_s,
                        generated_text=generated_text,
                    )
                    raw_rows.append(row)
                    generation_records.append(
                        {
                            "mode": mode["name"],
                            "gen_length": gen_length,
                            "block_length": block_length,
                            "repeat_idx": repeat_idx,
                            "generated_text": generated_text,
                        }
                    )
                    print(
                        f"  repeat={repeat_idx} elapsed={elapsed_s:.4f}s "
                        f"tokens_per_s={tokens_per_s:.4f}"
                    )
                    if args.show_text:
                        print(generated_text)
                    if device == "cuda":
                        torch.cuda.empty_cache()

    summary_rows = aggregate_rows(raw_rows)

    raw_csv_path = os.path.join(args.output_dir, "benchmark_raw.csv")
    summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    meta_json_path = os.path.join(args.output_dir, "benchmark_meta.json")
    text_json_path = os.path.join(args.output_dir, "benchmark_generations.json")

    save_csv(raw_rows, raw_csv_path)
    save_csv(summary_rows, summary_csv_path)
    save_json(
        {
            "model_path": model_path,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "prompt_text": args.prompt_text,
            "gen_lengths": args.gen_lengths,
            "block_lengths": args.block_lengths,
            "repeats": args.repeats,
            "warmup_runs": args.warmup_runs,
            "focus_layer": args.focus_layer,
            "focus_topk": args.focus_topk,
            "modes": MODES,
        },
        meta_json_path,
    )
    save_json(generation_records, text_json_path)

    plot_paths = plot_results(summary_rows, args.output_dir)

    print(f"Raw results saved to: {raw_csv_path}")
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Meta saved to: {meta_json_path}")
    print(f"Generations saved to: {text_json_path}")
    for path in plot_paths:
        print(f"Plot saved to: {path}")


if __name__ == "__main__":
    main()
