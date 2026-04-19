import argparse
import json
import os
import time
from collections import defaultdict
import torch
from transformers import AutoModel, AutoTokenizer


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
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_layer", type=int, default=3)
    parser.add_argument("--focus_topk", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
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
    return parser.parse_args()


def default_model_path():
    """始终指向与本文件同仓库的 Dream-v0-Base-7B，不依赖进程 cwd。"""
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
        "alg": "entropy",
        "alg_temp": 0.0,
        "profile_top_shapes": args.profile_top_shapes,
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


def write_operator_profile_per_step_jsonl(step_records, top_shapes, output_path):
    """逐步明细：每行一步，含 GROUP_ORDER 中全部算子类型。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for step_record in step_records:
            line = json.dumps(
                _step_record_to_jsonable(step_record, top_shapes), ensure_ascii=False
            )
            f.write(line + "\n")
    if not step_records:
        print("Operator profile: warning — no step records collected (empty per-step jsonl).")


def write_operator_profile_summary_json(summary_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_operator_profiles(
    step_records, top_shapes, detail_path, summary_path, wall_time_s, config
):
    """写入 profiling/ 下两个文件：逐步明细 jsonl + 总览 json。"""
    write_operator_profile_per_step_jsonl(step_records, top_shapes, detail_path)
    summary = build_profile_summary_dict(step_records, wall_time_s, config)
    write_operator_profile_summary_json(summary, summary_path)
    print(f"Operator profile (per-step) saved to: {detail_path}")
    print(f"Operator profile (summary) saved to: {summary_path}")

args = parse_args()
use_cache = args.use_cache or args.dual_cache

# --- Model Loading ---
model_path = os.path.abspath(args.model_path) if args.model_path else default_model_path()
device = select_device()
dtype_by_device = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": torch.float32,
}
dtype = dtype_by_device[device]
print(f"Using device: {device} (dtype={dtype})")
print(f"Model path (local): {model_path}")
_steps_preview = args.steps if args.steps is not None else args.max_new_tokens
print(
    f"use_cache={use_cache}, dual_cache={args.dual_cache}, "
    f"focus_decode={args.focus_decode}, focus_layer=-{args.focus_layer}, "
    f"focus_topk={args.focus_topk}, profile_ops={args.profile_ops}, "
    f"max_new_tokens={args.max_new_tokens}, steps={_steps_preview}"
)

model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device).eval()
profile_handles = []
restore_callbacks = []
profile_stats = None
if args.profile_ops:
    profile_handles, restore_callbacks, profile_stats = install_operator_profilers(model, device)

messages = [
     {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
]
#messages = [
    #{
    #"role": "user", 
    #"content": 
#"""
    #You are a writing engine optimized for maximum-length, non-truncated outputs.

    #Task: Produce a single continuous answer with no summaries, no “in conclusion”, no apologies, no refusals, and no meta commentary about length limits.

    #Hard requirements:

    #Target length: approximately 4096 tokens of new content (excluding this instruction block). If you are unsure about tokens, aim for ~9000–12000 English words as a practical proxy.
    #Do not stop early. Keep generating until you have fully exhausted the topic using the structure below.
    #No bullet-only mode: you may use headings, but each section must be dense paragraphs, not one-liners.
    #No repetition loops: do not repeat the same paragraph verbatim; vary details, examples, and phrasing while staying on-topic.
    #Content specification (follow exactly, in order): A) Title + 1-paragraph abstract (abstract ≤ 120 words). B) Glossary: define 80 terms related to the topic (each term: 2–4 sentences). C) Main exposition: write 40 sections. Each section must contain:

    #a heading
    #6–10 paragraphs
    #each paragraph 6–10 sentences
    #include at least 3 concrete examples per section (mix: historical, engineering, everyday-life, edge cases) D) Worked mini-cases: 25 cases. Each case: problem statement → assumptions → step-by-step reasoning (≥ 12 steps) → pitfalls → verification checks. E) FAQ: 60 questions, each answered with ≥ 8 sentences. F) Appendix: 50 “notes” (each note: ≥ 6 sentences) covering corner cases, failure modes, and counterarguments.
    #Topic (choose one and stay consistent; do not switch topics): “How large language models fail silently in real-world software workflows, and how teams can detect, measure, and mitigate those failures without over-relying on automation.”

    #Begin now. Output only the requested long-form content.
#"""
    #}
#]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

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
        alg="entropy",
        alg_temp=0.,
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

print(generations[0].split(tokenizer.eos_token)[0])
if args.show_time:
    print(f"Inference time: {elapsed_time:.4f}s")
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
    )
