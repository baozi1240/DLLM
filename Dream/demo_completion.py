import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--show_time", action="store_true")
    parser.add_argument("--profile_ops", action="store_true")
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="profiling",
        help="算子 profiling 输出目录（默认 profiling/）",
    )
    parser.add_argument(
        "--profile_stem",
        type=str,
        default=None,
        help="输出文件名前缀；默认根据 block_length/cache/focus/max_new_tokens 等超参自动生成",
    )
    parser.add_argument(
        "--profile_jsonl",
        type=str,
        default=None,
        help="覆盖逐步明细 jsonl 文件名（置于 --profile_dir 下，除非为绝对路径）",
    )
    parser.add_argument("--profile_top_shapes", type=int, default=10)
    parser.add_argument(
        "--trace_decode",
        action="store_true",
        help="将生成结果和每个 diffusion history step 新解码 token 数量写入 trace/*.json。",
    )
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_layer", type=int, default=3)
    parser.add_argument("--focus_topk", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--alg",
        type=str,
        default="entropy",
        choices=["origin", "entropy", "maskgit_plus", "topk_margin", "confidence_threshold"],
        help="diffusion 解码采样算法；confidence_threshold 启用阈值策略，每步可解锁多个位置",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="alg=confidence_threshold 时使用的置信度阈值",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="focus_decode + confidence_threshold 时使用的置信度补偿衰减系数",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="diffusion 步数；默认与 --max_new_tokens 相同",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型目录；默认为本脚本同级的 models/Dream-v0-Base-7B（仓库内本地权重与自定义代码）",
    )
    parser.add_argument("--dataset_path", type=str, default="data/HumanEval.jsonl.gz")
    parser.add_argument("--download_if_missing", action="store_true")
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="使用 chat template 构造输入（user + generation prompt）进行补全。",
    )
    return parser.parse_args()


def default_model_path():
    """始终指向与本文件同仓库的 Dream-v0-Base-7B，不依赖进程 cwd。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Dream-v0-Base-7B")


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


def locate_decoder_layers(model):
    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", None)
    if layers is None and hasattr(decoder, "model"):
        layers = getattr(decoder.model, "layers", None)
    if layers is None:
        raise ValueError("Unable to locate decoder layers for operator profiling.")
    return layers


def _tensor_shape(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    return None


def _shape_tuple(values):
    return tuple(_tensor_shape(value) for value in values)


OP_GROUPS = {
    "q_proj": "q/k/v proj",
    "k_proj": "q/k/v proj",
    "v_proj": "q/k/v proj",
    "sdpa": "sdpa",
    "o_proj": "o proj",
    "gate_proj": "up/gate proj",
    "up_proj": "up/gate proj",
    "down_proj": "down proj",
}

GROUP_ORDER = [
    "q/k/v proj",
    "sdpa",
    "o proj",
    "up/gate proj",
    "down proj",
]

MEMBER_ORDER = [
    "q_proj",
    "k_proj",
    "v_proj",
    "sdpa",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# 汇总里 attention = Q/K/V 线性 + SDPA；mlp = gate/up/down（不含 o_proj）
SUMMARY_ATTENTION_MEMBERS = ("q_proj", "k_proj", "v_proj", "sdpa")
SUMMARY_MLP_MEMBERS = ("gate_proj", "up_proj", "down_proj")


def _format_linear_shape(input_shape, output_shape):
    return {
        "kind": "linear",
        "input_shape": list(input_shape) if input_shape is not None else None,
        "output_shape": list(output_shape) if output_shape is not None else None,
    }


def _format_sdpa_shape(query_shape, key_shape, value_shape, output_shape):
    return {
        "kind": "sdpa",
        "query_shape": list(query_shape) if query_shape is not None else None,
        "key_shape": list(key_shape) if key_shape is not None else None,
        "value_shape": list(value_shape) if value_shape is not None else None,
        "output_shape": list(output_shape) if output_shape is not None else None,
    }


def _append_unique_shape(shape_list, shape_text):
    normalized = json.dumps(shape_text, sort_keys=True)
    if normalized not in {json.dumps(existing, sort_keys=True) for existing in shape_list}:
        shape_list.append(shape_text)


def _record_step_profile(step_record, op_name, shape_text, elapsed_ms):
    group_name = OP_GROUPS[op_name]
    group_record = step_record["groups"].setdefault(
        group_name,
        {
            "total_ms": 0.0,
            "members": {},
        },
    )
    group_record["total_ms"] += elapsed_ms

    member_record = group_record["members"].setdefault(
        op_name,
        {
            "calls": 0,
            "total_ms": 0.0,
            "shapes": [],
        },
    )
    member_record["calls"] += 1
    member_record["total_ms"] += elapsed_ms
    _append_unique_shape(member_record["shapes"], shape_text)


def install_operator_profilers(model, device):
    layers = locate_decoder_layers(model)
    step_records = []
    active_calls = defaultdict(list)
    hook_handles = []
    restore_callbacks = []
    active_steps = []

    def start_timer():
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return ("cuda", start_event, end_event)
        synchronize_device(device)
        return ("host", time.perf_counter())

    def stop_timer(timer_state):
        if timer_state[0] == "cuda":
            _, start_event, end_event = timer_state
            end_event.record()
            end_event.synchronize()
            return start_event.elapsed_time(end_event)
        synchronize_device(device)
        return (time.perf_counter() - timer_state[1]) * 1000.0

    def make_model_pre_hook():
        def pre_hook(module, args, kwargs):
            del module, args, kwargs
            step_record = {
                "step": len(step_records),
                "groups": {},
            }
            step_records.append(step_record)
            active_steps.append(step_record)
        return pre_hook

    def make_model_post_hook():
        def post_hook(module, args, kwargs, output):
            del module, args, kwargs, output
            active_steps.pop()
        return post_hook

    def make_module_pre_hook(op_name):
        def pre_hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states") if kwargs.get("hidden_states") is not None else (args[0] if args else None)
            active_calls[id(module)].append(
                {
                    "timer": start_timer(),
                    "op_name": op_name,
                    "input_shape": _tensor_shape(hidden_states),
                }
            )
        return pre_hook

    def make_module_post_hook(op_name):
        def post_hook(module, args, kwargs, output):
            del args, kwargs
            call_state = active_calls[id(module)].pop()
            elapsed_ms = stop_timer(call_state["timer"])
            if not active_steps:
                return
            _record_step_profile(
                active_steps[-1],
                op_name,
                _format_linear_shape(call_state["input_shape"], _tensor_shape(output)),
                elapsed_ms,
            )
        return post_hook

    def make_attention_context_pre_hook():
        def pre_hook(module, args, kwargs):
            del module, args, kwargs
        return pre_hook

    def make_attention_context_post_hook():
        def post_hook(module, args, kwargs, output):
            del module, args, kwargs, output
        return post_hook

    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def profiled_sdpa(*args, **kwargs):
        query_states = kwargs.get("query")
        key_states = kwargs.get("key")
        value_states = kwargs.get("value")
        if query_states is None and len(args) >= 3:
            query_states, key_states, value_states = args[:3]

        timer_state = start_timer()
        output = original_sdpa(*args, **kwargs)
        elapsed_ms = stop_timer(timer_state)

        if active_steps:
            _record_step_profile(
                active_steps[-1],
                "sdpa",
                _format_sdpa_shape(
                    _tensor_shape(query_states),
                    _tensor_shape(key_states),
                    _tensor_shape(value_states),
                    _tensor_shape(output),
                ),
                elapsed_ms,
            )
        return output

    torch.nn.functional.scaled_dot_product_attention = profiled_sdpa
    restore_callbacks.append(
        lambda: setattr(torch.nn.functional, "scaled_dot_product_attention", original_sdpa)
    )

    hook_handles.append(model.register_forward_pre_hook(make_model_pre_hook(), with_kwargs=True))
    hook_handles.append(model.register_forward_hook(make_model_post_hook(), with_kwargs=True))

    for layer in layers:
        hook_handles.append(
            layer.self_attn.register_forward_pre_hook(
                make_attention_context_pre_hook(),
                with_kwargs=True,
            )
        )
        hook_handles.append(
            layer.self_attn.register_forward_hook(
                make_attention_context_post_hook(),
                with_kwargs=True,
            )
        )

        modules = [
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
            ("o_proj", layer.self_attn.o_proj),
            ("gate_proj", layer.mlp.gate_proj),
            ("up_proj", layer.mlp.up_proj),
            ("down_proj", layer.mlp.down_proj),
        ]
        for op_name, module in modules:
            hook_handles.append(
                module.register_forward_pre_hook(
                    make_module_pre_hook(op_name),
                    with_kwargs=True,
                )
            )
            hook_handles.append(
                module.register_forward_hook(
                    make_module_post_hook(op_name),
                    with_kwargs=True,
                )
            )

    return hook_handles, restore_callbacks, step_records


def remove_hooks(hook_handles, restore_callbacks):
    for handle in hook_handles:
        handle.remove()
    for callback in reversed(restore_callbacks):
        callback()


def _step_record_to_jsonable(step_record, top_shapes=None):
    groups = {}
    for group_name in GROUP_ORDER:
        group_record = step_record["groups"].get(group_name)
        if group_record is None:
            continue
        members = {}
        for member_name in MEMBER_ORDER:
            member_record = group_record["members"].get(member_name)
            if member_record is None:
                continue
            shapes = member_record["shapes"]
            if top_shapes is not None:
                shapes = shapes[:top_shapes]
            members[member_name] = {
                "calls": member_record["calls"],
                "total_ms": round(member_record["total_ms"], 6),
                "avg_ms": round(
                    member_record["total_ms"] / member_record["calls"], 6
                ) if member_record["calls"] else 0.0,
                "shapes": shapes,
            }
        groups[group_name] = {
            "total_ms": round(group_record["total_ms"], 6),
            "members": members,
        }

    return {
        "step": step_record["step"],
        "total_ms": round(
            sum(group_record["total_ms"] for group_record in step_record["groups"].values()),
            6,
        ),
        "groups": groups,
    }


def build_profile_stem(args):
    """由可配置超参生成默认文件名前缀（不含目录与后缀）。"""
    steps = args.steps if args.steps is not None else args.max_new_tokens
    blk = args.block_length if args.block_length is not None else "none"
    parts = [
        "dream",
        f"blk{blk}",
        f"uc{int(args.use_cache)}",
        f"dc{int(args.dual_cache)}",
        f"fd{int(args.focus_decode)}",
        f"fl{args.focus_layer}",
        f"ftk{args.focus_topk}",
        f"max{args.max_new_tokens}",
        f"s{steps}",
    ]
    return "_".join(str(p) for p in parts)


def resolve_profile_output_paths(args):
    """逐步明细 jsonl 与总览 summary json 的完整路径。"""
    profile_dir = args.profile_dir
    os.makedirs(profile_dir, exist_ok=True)
    if args.profile_jsonl:
        detail = args.profile_jsonl
        if not os.path.isabs(detail):
            detail = os.path.join(profile_dir, os.path.basename(detail))
        summary = os.path.splitext(detail)[0] + "_summary.json"
    else:
        stem = args.profile_stem or build_profile_stem(args)
        detail = os.path.join(profile_dir, f"{stem}_per_step.jsonl")
        summary = os.path.join(profile_dir, f"{stem}_summary.json")
    return detail, summary


def _safe_filename_part(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def build_trace_stem(args, model_path):
    """由 demo_completion.py 中影响解码的参数生成可识别 trace 文件名。"""
    steps = args.steps if args.steps is not None else args.max_new_tokens
    block_length = args.block_length if args.block_length is not None else "none"
    model_name = os.path.basename(os.path.normpath(model_path))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [
        "demo_completion",
        f"model-{model_name}",
        f"alg-{args.alg}",
        f"thr-{args.threshold:g}",
        f"gamma-{args.gamma:g}",
        f"blk-{block_length}",
        f"uc-{int(args.use_cache)}",
        f"euc-{int(args.use_cache or args.dual_cache)}",
        f"dc-{int(args.dual_cache)}",
        f"fd-{int(args.focus_decode)}",
        f"fl-{args.focus_layer}",
        f"ftk-{args.focus_topk}",
        f"max-{args.max_new_tokens}",
        f"steps-{steps}",
        timestamp,
    ]
    return "_".join(_safe_filename_part(part) for part in parts)


def resolve_trace_output_path(args, model_path):
    trace_dir = "trace"
    os.makedirs(trace_dir, exist_ok=True)
    return os.path.join(trace_dir, f"{build_trace_stem(args, model_path)}.json")


def build_profile_config_dict(args, device, dtype, model_path, steps_used):
    return {
        "model_path": model_path,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "block_length": args.block_length,
        "use_cache": args.use_cache,
        "dual_cache": args.dual_cache,
        "focus_decode": args.focus_decode,
        "focus_layer": args.focus_layer,
        "focus_topk": args.focus_topk,
        "max_new_tokens": args.max_new_tokens,
        "steps": steps_used,
        "temperature": 0.0,
        "top_p": 0.95,
        "alg": args.alg,
        "alg_temp": 0.0,
        "threshold": args.threshold,
        "gamma": args.gamma,
        "profile_top_shapes": args.profile_top_shapes,
    }


def build_trace_config_dict(args, device, dtype, model_path, steps_used):
    return {
        "script": "demo_completion.py",
        "model_path": model_path,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "block_length": args.block_length,
        "use_cache": args.use_cache,
        "effective_use_cache": bool(args.use_cache or args.dual_cache),
        "dual_cache": args.dual_cache,
        "focus_decode": args.focus_decode,
        "focus_layer": args.focus_layer,
        "focus_topk": args.focus_topk,
        "max_new_tokens": args.max_new_tokens,
        "steps": steps_used,
        "temperature": 0.0,
        "top_p": 0.95,
        "alg": args.alg,
        "alg_temp": 0.0,
        "threshold": args.threshold,
        "gamma": args.gamma,
    }


def _aggregate_member_totals(step_records):
    member_ms = defaultdict(float)
    member_calls = defaultdict(int)
    for step in step_records:
        for group in step["groups"].values():
            for name, rec in group["members"].items():
                member_ms[name] += rec["total_ms"]
                member_calls[name] += rec["calls"]
    return member_ms, member_calls


def build_profile_summary_dict(step_records, wall_time_s, config):
    member_ms, member_calls = _aggregate_member_totals(step_records)
    all_ops_total = sum(member_ms.values())

    att_by = {n: round(member_ms.get(n, 0.0), 6) for n in SUMMARY_ATTENTION_MEMBERS}
    att_total = sum(member_ms.get(n, 0.0) for n in SUMMARY_ATTENTION_MEMBERS)
    att_calls = {n: int(member_calls.get(n, 0)) for n in SUMMARY_ATTENTION_MEMBERS}

    mlp_by = {n: round(member_ms.get(n, 0.0), 6) for n in SUMMARY_MLP_MEMBERS}
    mlp_total = sum(member_ms.get(n, 0.0) for n in SUMMARY_MLP_MEMBERS)
    mlp_calls = {n: int(member_calls.get(n, 0)) for n in SUMMARY_MLP_MEMBERS}

    o_ms = round(member_ms.get("o_proj", 0.0), 6)
    o_calls = int(member_calls.get("o_proj", 0))

    return {
        "schema": "dream_operator_profile_summary_v1",
        "wall_time_s": round(wall_time_s, 6),
        "num_decode_steps_profiled": len(step_records),
        "config": config,
        "aggregates_ms": {
            "all_ops_total": round(all_ops_total, 6),
            "attention": {
                "description": "q_proj + k_proj + v_proj + sdpa",
                "total_ms": round(att_total, 6),
                "by_member": att_by,
                "calls_by_member": att_calls,
            },
            "mlp": {
                "description": "gate_proj + up_proj + down_proj",
                "total_ms": round(mlp_total, 6),
                "by_member": mlp_by,
                "calls_by_member": mlp_calls,
            },
            "o_proj": {
                "description": "output projection（单独统计，不在 attention 与 mlp 汇总中）",
                "total_ms": o_ms,
                "calls": o_calls,
            },
        },
    }


def write_operator_profile_per_step_jsonl(step_records, top_shapes, output_path, log_stream=None):
    """逐步明细：每行一步，含 GROUP_ORDER 中全部算子类型。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for step_record in step_records:
            line = json.dumps(
                _step_record_to_jsonable(step_record, top_shapes), ensure_ascii=False
            )
            f.write(line + "\n")
    if not step_records:
        print(
            "Operator profile: warning — no step records collected (empty per-step jsonl).",
            file=log_stream or sys.stdout,
        )


def write_operator_profile_summary_json(summary_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_operator_profiles(
    step_records, top_shapes, detail_path, summary_path, wall_time_s, config, log_stream=None
):
    """写入 profiling/ 下两个文件：逐步明细 jsonl + 总览 json。"""
    write_operator_profile_per_step_jsonl(step_records, top_shapes, detail_path, log_stream)
    summary = build_profile_summary_dict(step_records, wall_time_s, config)
    write_operator_profile_summary_json(summary, summary_path)
    print(f"Operator profile (per-step) saved to: {detail_path}", file=log_stream or sys.stdout)
    print(f"Operator profile (summary) saved to: {summary_path}", file=log_stream or sys.stdout)


def resolve_mask_token_id(model, tokenizer):
    candidates = [
        getattr(getattr(model, "generation_config", None), "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError("Unable to resolve mask_token_id for decode tracing.")


def _to_cpu_2d_tokens(tokens):
    tokens = tokens.detach().to("cpu")
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    return tokens


def build_decode_trace_records(history, input_ids, sequences, mask_token_id, block_length):
    if history is None:
        history = []
    elif torch.is_tensor(history):
        history = [history]
    else:
        history = list(history)

    input_ids_cpu = _to_cpu_2d_tokens(input_ids)
    sequences_cpu = _to_cpu_2d_tokens(sequences)
    input_len = input_ids_cpu.shape[1]
    generated_length = sequences_cpu.shape[1] - input_len
    if generated_length < 0:
        raise ValueError("sequences is shorter than input_ids; cannot trace generated span.")
    effective_block_length = int(block_length) if block_length else int(generated_length)

    if not history:
        return [], {
            "num_history_steps": 0,
            "batch_size": int(sequences_cpu.shape[0]),
            "generated_length": int(generated_length),
            "block_length": int(effective_block_length),
            "total_new_tokens": 0,
            "final_remaining_masks": [
                int(((sequences_cpu[b, input_len:] == mask_token_id).sum()).item())
                for b in range(sequences_cpu.shape[0])
            ],
        }

    prev_generated = torch.full(
        (sequences_cpu.shape[0], generated_length),
        int(mask_token_id),
        dtype=sequences_cpu.dtype,
    )

    records = []
    for step_idx, state in enumerate(history):
        state_cpu = _to_cpu_2d_tokens(state)
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
            if effective_block_length > 0:
                blocks = sorted({int(pos // effective_block_length) for pos in positions})
                block_start_tokens = sum(1 for pos in positions if pos % effective_block_length == 0)
            else:
                blocks = []
                block_start_tokens = 0

            records.append(
                {
                    "step": int(step_idx),
                    "batch": int(batch_idx),
                    "new_tokens": int(len(positions)),
                    "cumulative_tokens": int((~current_mask[batch_idx]).sum().item()),
                    "remaining_masks": int(current_mask[batch_idx].sum().item()),
                    "blocks": blocks,
                    "block_start_tokens": int(block_start_tokens),
                    "positions": [int(pos) for pos in positions],
                    "changed_after_decode_positions": [
                        int(pos) for pos in changed_positions
                    ],
                }
            )

        prev_generated = current_generated.clone()

    summary = {
        "num_history_steps": int(len(history)),
        "batch_size": int(sequences_cpu.shape[0]),
        "generated_length": int(generated_length),
        "block_length": int(effective_block_length),
        "total_new_tokens": int(sum(record["new_tokens"] for record in records)),
        "final_remaining_masks": [
            int(((sequences_cpu[b, input_len:] == mask_token_id).sum()).item())
            for b in range(sequences_cpu.shape[0])
        ],
    }
    return records, summary


def build_decode_trace_json(generation, records, summary, config):
    return {
        "schema": "demo_completion_decode_trace_v1",
        "config": config,
        "generation": generation,
        "decode_trace": {
            "summary": summary,
            "steps": records,
        },
    }


def write_decode_trace_json(trace_payload, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

args = parse_args()
use_cache = args.use_cache or args.dual_cache
log_stream = sys.stderr if args.trace_decode else sys.stdout

# --- Model Loading ---
model_path = os.path.abspath(args.model_path) if args.model_path else default_model_path()
device = select_device()
dtype_by_device = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": torch.float32,
}
dtype = dtype_by_device[device]
print(f"Using device: {device} (dtype={dtype})", file=log_stream)
print(f"Model path (local): {model_path}", file=log_stream)
_steps_preview = args.steps if args.steps is not None else args.max_new_tokens
print(
    f"use_cache={use_cache}, dual_cache={args.dual_cache}, "
    f"focus_decode={args.focus_decode}, focus_layer=-{args.focus_layer}, "
    f"focus_topk={args.focus_topk}, profile_ops={args.profile_ops}, "
    f"trace_decode={args.trace_decode}, "
    f"max_new_tokens={args.max_new_tokens}, steps={_steps_preview}, "
    f"gamma={args.gamma}",
    file=log_stream,
)

model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device).eval()
profile_handles = []
restore_callbacks = []
profile_stats = None
if args.profile_ops:
    profile_handles, restore_callbacks, profile_stats = install_operator_profilers(model, device)


dataset_path = ensure_dataset(args.dataset_path, args.download_if_missing)
problems = load_humaneval(dataset_path)
if not problems:
    raise ValueError(f"No samples found in HumanEval dataset: {dataset_path}")
first_problem = problems[0]
prompt = first_problem["prompt"]
if args.use_chat_template:
    messages = [{"role": "user", "content": prompt}]
    chat_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    input_ids = chat_inputs.input_ids.to(device)
    attention_mask = chat_inputs.attention_mask.to(device)
else:
    if args.add_bos_token and tokenizer.bos_token:
        prompt = tokenizer.bos_token + prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
print(f"HumanEval first task: {first_problem['task_id']}", file=log_stream)
if args.use_chat_template:
    print("Prompt build mode: chat_template(user + generation_prompt)", file=log_stream)
else:
    print(
        f"Prompt build mode: raw_prompt + bos({int(bool(args.add_bos_token))})",
        file=log_stream,
    )

if args.show_time:
    synchronize_device(device)
start_time = time.perf_counter()

try:
    steps = args.steps if args.steps is not None else args.max_new_tokens
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=True,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        top_p=0.95,
        block_length=args.block_length,
        use_cache=use_cache,
        dual_cache=args.dual_cache,
        alg=args.alg,
        alg_temp=0.,
        threshold=args.threshold,
        gamma=args.gamma,
        focus_decode=args.focus_decode,
        focus_layer=args.focus_layer,
        focus_topk=args.focus_topk,
    )
finally:
    if profile_handles:
        remove_hooks(profile_handles, restore_callbacks)

if args.show_time:
    synchronize_device(device)
elapsed_time = time.perf_counter() - start_time

generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]
generation_text = generations[0].split(tokenizer.eos_token)[0]

if args.trace_decode:
    _mask_token_id = resolve_mask_token_id(model, tokenizer)
    _trace_records, _trace_summary = build_decode_trace_records(
        getattr(output, "history", None),
        input_ids,
        output.sequences,
        _mask_token_id,
        args.block_length,
    )
    _steps_used = args.steps if args.steps is not None else args.max_new_tokens
    _trace_path = resolve_trace_output_path(args, model_path)
    _trace_payload = build_decode_trace_json(
        generation_text,
        _trace_records,
        _trace_summary,
        build_trace_config_dict(args, device, dtype, model_path, _steps_used),
    )
    write_decode_trace_json(_trace_payload, _trace_path)
    print(f"Decode trace saved to: {_trace_path}", file=log_stream)
else:
    print(generation_text)
if args.show_time:
    print(f"Inference time: {elapsed_time:.4f}s", file=log_stream)
if args.profile_ops and profile_stats is not None:
    _detail_path, _summary_path = resolve_profile_output_paths(args)
    _steps_used = args.steps if args.steps is not None else args.max_new_tokens
    _prof_config = build_profile_config_dict(
        args, device, dtype, model_path, _steps_used
    )
    save_operator_profiles(
        profile_stats,
        args.profile_top_shapes,
        _detail_path,
        _summary_path,
        elapsed_time,
        _prof_config,
        log_stream,
    )
