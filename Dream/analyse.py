import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens_local(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0

sample_tokens = sample_tokens_local


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def resolve_mask_id(model, tokenizer=None, explicit_mask_id=None):
    if explicit_mask_id is not None:
        return int(explicit_mask_id)

    candidates = []

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        candidates.append(getattr(generation_config, "mask_token_id", None))

    model_config = getattr(model, "config", None)
    if model_config is not None:
        candidates.append(getattr(model_config, "mask_token_id", None))

    if tokenizer is not None:
        candidates.append(getattr(tokenizer, "mask_token_id", None))
        mask_token = getattr(tokenizer, "mask_token", None)
        if mask_token is not None:
            try:
                candidates.append(tokenizer.convert_tokens_to_ids(mask_token))
            except Exception:
                pass

    for candidate in candidates:
        if candidate is None:
            continue
        candidate = int(candidate)
        if candidate >= 0:
            return candidate

    raise ValueError(
        "Cannot resolve Dream mask id from --mask_id, generation_config, model.config, or tokenizer."
    )


def get_all_transformer_blocks(model):
    return list(model.model.layers)



def get_target_block(model, analyze_layer_from_end=1):
    blocks = get_all_transformer_blocks(model)
    num_blocks = len(blocks)

    if analyze_layer_from_end < 1 or analyze_layer_from_end > num_blocks:
        raise ValueError(
            f"analyze_layer_from_end must be in [1, {num_blocks}], got {analyze_layer_from_end}"
        )

    target_index = num_blocks - analyze_layer_from_end
    return blocks[target_index], target_index, num_blocks


def patch_attention_for_qk(target_block):
    capture = {}
    original_forward = target_block.self_attn.forward
    attention_module = target_block.self_attn
    globals_dict = original_forward.__func__.__globals__
    apply_rotary_pos_emb = globals_dict["apply_rotary_pos_emb"]
    repeat_kv = globals_dict["repeat_kv"]

    def wrapped_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        raw_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if isinstance(attention_mask, torch.Tensor):
            attn_bias = attention_mask[:, :, :, : key_states.shape[-2]]
            if attn_bias.dtype == torch.bool:
                attn_bias = torch.where(
                    attn_bias,
                    torch.zeros((), device=attn_bias.device, dtype=raw_scores.dtype),
                    torch.full((), torch.finfo(raw_scores.dtype).min, device=attn_bias.device, dtype=raw_scores.dtype),
                )
            else:
                attn_bias = attn_bias.to(dtype=raw_scores.dtype)
            raw_scores = raw_scores + attn_bias

        capture["raw_scores"] = raw_scores.detach().float().cpu()

        attn_weights = nn_functional_softmax(raw_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    nn_functional_softmax = torch.nn.functional.softmax
    target_block.self_attn.forward = types.MethodType(wrapped_forward, attention_module)

    def restore():
        target_block.self_attn.forward = original_forward

    return restore, capture


@torch.no_grad()
def generate_and_analyze_prev_focus(
    model,
    prompt,
    attention_mask=None,
    steps=512,
    gen_length=512,
    block_length=None,
    temperature=0.0,
    mask_id=None,
    top_p=0.95,
    sample_top_k=None,
    alg="entropy",
    alg_temp=0.0,
    topk=8,
    analyze_layer_from_end=1,
):
    assert prompt.size(0) == 1, "当前代码按 batch_size=1 写。"
    if mask_id is None:
        raise ValueError("mask_id must be provided for Dream analyse.py.")

    device = model.device
    if block_length is None:
        block_length = gen_length
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length {gen_length} must be divisible by block_length {block_length}"
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps {steps} must be divisible by number of blocks {num_blocks}"
        )
    inner_steps = steps // num_blocks

    x = F.pad(prompt, (0, gen_length), value=mask_id)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1.0)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        model_attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        position_ids = None
        model_attention_mask = "full"

    timesteps = torch.linspace(1, model.generation_config.eps, inner_steps + 1, device=device)

    target_block, target_block_index, total_layers = get_target_block(
        model, analyze_layer_from_end=analyze_layer_from_end
    )
    restore_fn, capture = patch_attention_for_qk(target_block)

    step_records = []
    decoded_positions = []
    prev_step_cache = None

    try:
        for block_id in range(num_blocks):
            block_start = prompt.shape[1] + block_id * block_length
            block_end = block_start + block_length
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, inner_steps)

            for i in range(inner_steps):
                mask_index = (x == mask_id)
                if not mask_index.any():
                    break

                logits = model(x, model_attention_mask, position_ids).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                current_step_raw_scores = capture["raw_scores"]

                t = timesteps[i]
                s = timesteps[i + 1]
                current_transfer_positions = []
                current_transfer_confidences = []

                block_mask_index = mask_index.clone()
                block_mask_index[:, :block_start] = False
                block_mask_index[:, block_end:] = False
                mask_logits = logits[block_mask_index]
                if mask_logits.numel() == 0:
                    continue

                if alg == "origin":
                    p_transfer = 1 - s / t if i < inner_steps - 1 else 1.0
                    block_slice = x[:, block_start:block_end].clone()
                    local_mask_index = (block_slice == mask_id)
                    x0 = torch.full_like(block_slice[local_mask_index], mask_id)
                    transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
                    if transfer_index_t_s.any():
                        _, sampled = sample_tokens(
                            logits[:, block_start:block_end][local_mask_index][transfer_index_t_s],
                            temperature=temperature,
                            top_p=top_p,
                            top_k=sample_top_k,
                        )
                        x0[transfer_index_t_s] = sampled

                    current_transfer_positions = (
                        torch.where(local_mask_index[0])[0][transfer_index_t_s].detach().cpu() + block_start
                    ).tolist()
                    current_transfer_confidences = [float("nan")] * len(current_transfer_positions)

                    block_slice[local_mask_index] = x0.clone()
                    x[:, block_start:block_end] = block_slice
                else:
                    if alg == "maskgit_plus":
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=sample_top_k,
                        )
                    elif alg == "topk_margin":
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=sample_top_k,
                            margin_confidence=True,
                        )
                    elif alg == "entropy":
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=sample_top_k,
                            neg_entropy=True,
                        )
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")

                    full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype, device=device)
                    full_confidence[block_mask_index] = confidence
                    x_candidate = torch.full_like(x, mask_id)
                    x_candidate[block_mask_index] = x0.clone()

                    for batch_idx in range(x.size(0)):
                        k_transfer = int(num_transfer_tokens[batch_idx, i].item())
                        if k_transfer <= 0:
                            continue

                        if alg_temp is None or alg_temp == 0:
                            transfer_index = torch.topk(full_confidence[batch_idx], k_transfer).indices
                        else:
                            probs = F.softmax(full_confidence[batch_idx] / alg_temp, dim=-1)
                            transfer_index = torch.multinomial(probs, num_samples=k_transfer)

                        x[batch_idx, transfer_index] = x_candidate[batch_idx, transfer_index]
                        if batch_idx == 0:
                            current_transfer_positions = transfer_index.detach().cpu().tolist()
                            current_transfer_confidences = (
                                full_confidence[batch_idx, transfer_index].detach().float().cpu().tolist()
                            )

                paired = list(zip(current_transfer_positions, current_transfer_confidences))
                paired.sort(key=lambda z: (float("-inf") if math.isnan(z[1]) else z[1]), reverse=True)

                current_block_candidate_positions = (
                    torch.where(x[0, block_start:block_end] == mask_id)[0] + block_start
                ).detach().cpu()
                if current_block_candidate_positions.numel() > 0:
                    next_step_candidate_positions = current_block_candidate_positions
                else:
                    if block_id + 1 < num_blocks:
                        next_block_start = prompt.shape[1] + (block_id + 1) * block_length
                        next_block_end = next_block_start + block_length
                        next_step_candidate_positions = (
                            torch.where(x[0, next_block_start:next_block_end] == mask_id)[0] + next_block_start
                        ).detach().cpu()
                    else:
                        next_step_candidate_positions = torch.empty(0, dtype=torch.long)

                for curr_pos, curr_conf in paired:
                    record = {
                        "global_step": len(step_records),
                        "block_id": block_id,
                        "inner_step": i,
                        "curr_pos": int(curr_pos),
                        "curr_confidence": None if math.isnan(curr_conf) else float(curr_conf),
                        "hit": None,
                        "prev_focus_info": None,
                        "analyzed_layer_index": int(target_block_index),
                        "analyzed_layer_from_end": int(analyze_layer_from_end),
                        "total_layers": int(total_layers),
                    }

                    if prev_step_cache is not None:
                        prev_raw_scores = prev_step_cache["raw_scores"]
                        prev_decoded_positions = prev_step_cache["decoded_positions"]
                        prev_candidate_positions = prev_step_cache["candidate_positions"]
                        prev_focus_info = []
                        hit = None

                        if prev_decoded_positions.numel() > 0 and prev_candidate_positions.numel() > 0:
                            agg_scores = prev_raw_scores[0].sum(dim=0)
                            any_hit = 0

                            for prev_pos in prev_decoded_positions.tolist():
                                if not (0 <= prev_pos < agg_scores.shape[0]):
                                    continue

                                candidate_scores = agg_scores[prev_pos, prev_candidate_positions]
                                if candidate_scores.numel() == 0:
                                    continue

                                k_local = min(topk, candidate_scores.numel())
                                top_idx = torch.topk(candidate_scores, k=k_local).indices
                                focus_topk = prev_candidate_positions[top_idx].cpu().tolist()
                                one_hit = int(curr_pos in focus_topk)

                                prev_focus_info.append(
                                    {
                                        "prev_pos": int(prev_pos),
                                        "topk_positions": focus_topk,
                                        "hit": one_hit,
                                    }
                                )
                                if one_hit:
                                    any_hit = 1

                            hit = any_hit if prev_focus_info else None

                        record["prev_focus_info"] = prev_focus_info if hit is not None else None
                        record["hit"] = hit

                    step_records.append(record)

                for curr_pos, _ in paired:
                    decoded_positions.append(int(curr_pos))

                prev_step_cache = {
                    "raw_scores": current_step_raw_scores,
                    "decoded_positions": torch.tensor(
                        [int(pos) for pos, _ in paired], dtype=torch.long
                    ),
                    "candidate_positions": next_step_candidate_positions,
                }

        total = sum(1 for r in step_records if r["hit"] is not None)
        hit_count = sum(int(r["hit"]) for r in step_records if r["hit"] is not None)
        hit_ratio = hit_count / total if total > 0 else 0.0

        return {
            "final_ids": x,
            "decoded_positions": decoded_positions,
            "records": step_records,
            "summary": {
                "total_valid_steps": total,
                "hit_count": hit_count,
                "hit_ratio": hit_ratio,
                "topk": topk,
                "gen_length": gen_length,
                "block_length": block_length,
                "steps": steps,
                "alg": alg,
                "alg_temp": alg_temp,
                "analyzed_layer_index": int(target_block_index),
                "analyzed_layer_from_end": int(analyze_layer_from_end),
                "total_layers": int(total_layers),
            },
        }
    finally:
        restore_fn()



def parse_args():
    parser = argparse.ArgumentParser()
    default_model_path = str(Path(__file__).resolve().parent / "models" / "Dream-v0-Base-7B")

    parser.add_argument("--model_path", type=str, default=default_model_path)
    parser.add_argument("--prompt_text", type=str, default="Explain why the sky appears blue.")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--analyze_layer_from_end", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--sample_top_k", type=int, default=None)
    parser.add_argument("--mask_token_id", dest="mask_id", type=int)
    parser.add_argument(
        "--alg",
        type=str,
        default="entropy",
        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
    )
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--output_prefix", type=str, default="dream_prev_focus")
    return parser.parse_args()



def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_dir = Path(args.model_path).resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    parent_dir = str(model_dir.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    mask_id = resolve_mask_id(
        model=model,
        tokenizer=tokenizer,
        explicit_mask_id=args.mask_id,
    )
    model.generation_config.mask_token_id = mask_id
    print(f"Using mask_id: {mask_id}")

    encoded = tokenizer(
        [args.prompt_text],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    result = generate_and_analyze_prev_focus(
        model=model,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        mask_id=mask_id,
        top_p=args.top_p,
        sample_top_k=args.sample_top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        topk=args.topk,
        analyze_layer_from_end=args.analyze_layer_from_end,
    )

    final_ids = result["final_ids"]
    output_text = tokenizer.batch_decode(
        final_ids[:, input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    print("=" * 80)
    print("Generated text:")
    print(output_text)
    print("=" * 80)
    print("Summary:")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("=" * 80)
    print("First 10 records:")
    for record in result["records"][:10]:
        print(json.dumps(record, ensure_ascii=False))

    summary_path = f"{args.output_prefix}_summary.json"
    records_path = f"{args.output_prefix}_stats.jsonl"

    with open(records_path, "w", encoding="utf-8") as f:
        for record in result["records"]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved records to: {records_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
