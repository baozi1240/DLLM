import argparse
import json
import logging
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

eval_logger = logging.getLogger(__name__)

try:
    import sympy
    from sympy.parsing.latex import parse_latex

    HAS_LATEX_EQUIV = True
except Exception:
    sympy = None
    parse_latex = None
    HAS_LATEX_EQUIV = False


OFFICIAL_MINERVA_FEWSHOT = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


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
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=("official", "legacy"),
        default="official",
        help="Use the official Minerva-style 4-shot prompt by default.",
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap prompts with the tokenizer chat template. Disabled by default to match lm-eval.",
    )
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
        except Exception as exc:  # pragma: no cover
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


def doc_to_text(problem: str) -> str:
    return "Problem:\n" + problem.strip() + "\n\nSolution:"


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left) :]
        return s

    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def get_unnormalized_answer(text: str) -> Optional[str]:
    match = re.search(
        r"Final Answer: The final answer is(.*?)(?:\. I hope it is correct\.|$)",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    boxed = remove_boxed(last_boxed_only_string(text))
    if boxed is not None:
        return boxed.strip()
    return None


def normalize_final_answer(final_answer: Optional[str]) -> Optional[str]:
    if final_answer is None:
        return None

    final_answer = final_answer.strip()
    if not final_answer:
        return None

    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    final_answer = final_answer.strip()
    return final_answer or None


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a_int = int(a)
        b_int = int(b)
        if string != f"{a_int}/{b_int}":
            return string
        return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
    except Exception:
        return string


def remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: Optional[str]) -> Optional[str]:
    if string is None:
        return None

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)
    return string


class timeout:
    def __init__(self, seconds=5, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


def is_equiv(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None

    pred_norm = normalize_final_answer(pred)
    gold_norm = normalize_final_answer(gold)
    if pred_norm is None or gold_norm is None:
        return pred_norm == gold_norm

    pred_stripped = strip_string(pred_norm)
    gold_stripped = strip_string(gold_norm)
    if pred_stripped == gold_stripped:
        return True

    if not HAS_LATEX_EQUIV:
        return False

    try:
        with timeout(seconds=5):
            parsed_pred = parse_latex(pred_norm)
            parsed_gold = parse_latex(gold_norm)
            diff = parsed_pred - parsed_gold
            return bool(sympy.simplify(diff) == 0)
    except Exception as exc:
        eval_logger.debug("latex equivalence failed for %s vs %s: %s", pred_norm, gold_norm, exc)
        return False


def extract_prediction(text: str, prompt_style: str) -> Optional[str]:
    text = truncate_generation(text)

    if prompt_style == "official":
        return normalize_final_answer(get_unnormalized_answer(text))

    boxed = remove_boxed(last_boxed_only_string(text))
    if boxed is not None:
        return normalize_final_answer(boxed)
    return normalize_final_answer(text.splitlines()[-1] if text.splitlines() else text)


def extract_reference(example: Dict[str, Any], prompt_style: str) -> Optional[str]:
    explicit = (
        example.get("answer")
        or example.get("final_answer")
        or example.get("target")
        or example.get("label")
    )
    if explicit:
        return normalize_final_answer(str(explicit))

    solution = str(example.get("solution") or example.get("answer") or "")
    if prompt_style == "official":
        return normalize_final_answer(remove_boxed(last_boxed_only_string(solution)))

    boxed = remove_boxed(last_boxed_only_string(solution))
    if boxed is not None:
        return normalize_final_answer(boxed)
    return normalize_final_answer(solution)


def select_few_shot_examples(args, normalized_train_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if args.n_shot <= 0:
        return []

    if args.prompt_style == "official":
        shot_count = min(args.n_shot, len(OFFICIAL_MINERVA_FEWSHOT))
        return [
            {
                "task_id": f"official_{idx}",
                "problem": shot["problem"].strip(),
                "solution": shot["solution"].strip(),
                "raw": shot,
            }
            for idx, shot in enumerate(OFFICIAL_MINERVA_FEWSHOT[:shot_count], start=1)
        ]

    shot_count = min(args.n_shot, len(normalized_train_examples))
    return normalized_train_examples[:shot_count]


def build_few_shot_prefix(examples: List[Dict[str, Any]], prompt_style: str) -> str:
    if not examples:
        return ""

    if prompt_style == "official":
        return "\n\n".join(
            doc_to_text(example["problem"]) + example["solution"] for example in examples
        )

    parts = [
        "Solve the following competition math problems carefully. "
        "Show the reasoning and put the final answer inside \\boxed{}.\n\n"
    ]
    for idx, example in enumerate(examples, start=1):
        parts.append(f"Example {idx}:\n")
        parts.append(f"Problem: {example['problem']}\n\n")
        parts.append(f"Solution: {example['solution']}\n\n")
    return "".join(parts).rstrip()


def build_prompt(problem: str, few_shot_prefix: str, prompt_style: str) -> str:
    if prompt_style == "official":
        prompt = doc_to_text(problem)
        if few_shot_prefix:
            return few_shot_prefix + "\n\n" + prompt
        return prompt

    prompt = few_shot_prefix
    if few_shot_prefix:
        prompt += "\n\nNow solve the next problem.\n\n"
    prompt += (
        f"Problem: {problem.strip()}\n\n"
        "Solution: Please reason step by step and end with the final answer in \\boxed{}."
    )
    return prompt


def tokenize_prompts(tokenizer, prompts: List[str], use_chat_template: bool):
    if use_chat_template:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        return tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
        )
    return tokenizer(
        prompts,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
    )


def truncate_generation(text: str) -> str:
    stop_markers = [
        "\nProblem:",
        "\n\nProblem:",
        "<|im_end|>",
    ]
    cutoffs = [text.find(marker) for marker in stop_markers if text.find(marker) != -1]
    if cutoffs:
        text = text[: min(cutoffs)]
    return text.strip()


@torch.no_grad()
def generate_batch(model, tokenizer, batch_examples, few_shot_prefix, device, args):
    prompts = [
        build_prompt(example["problem"], few_shot_prefix, args.prompt_style)
        for example in batch_examples
    ]
    inputs = tokenize_prompts(tokenizer, prompts, args.use_chat_template)
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
        responses.append(truncate_generation(text))
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
        "prompt_style": args.prompt_style,
        "use_chat_template": bool(args.use_chat_template),
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
        f"focus_topk={args.focus_topk}, "
        f"prompt_style={args.prompt_style}, "
        f"use_chat_template={args.use_chat_template}",
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

    normalized_fewshot_dataset: List[Dict[str, Any]] = []
    if args.n_shot > 0 and args.prompt_style != "official":
        few_shot_dataset = load_math_split(args, args.fewshot_split)
        normalized_fewshot_dataset = [
            normalize_example(few_shot_dataset[i], i) for i in range(len(few_shot_dataset))
        ]
    few_shot_examples = select_few_shot_examples(args, normalized_fewshot_dataset)
    few_shot_prefix = build_few_shot_prefix(few_shot_examples, args.prompt_style)

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
                gold = extract_reference(example["raw"], args.prompt_style)
                pred = extract_prediction(prediction, args.prompt_style)
                is_correct = is_equiv(pred, gold)

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
