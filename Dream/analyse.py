import argparse
import json
import math
import sys
import types
from collections import deque
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



def get_target_block(model, focus_layer=1):
    blocks = get_all_transformer_blocks(model)
    num_blocks = len(blocks)

    if focus_layer < 1 or focus_layer > num_blocks:
        raise ValueError(
            f"focus_layer must be in [1, {num_blocks}], got {focus_layer}"
        )

    target_index = num_blocks - focus_layer
    return blocks[target_index], target_index, num_blocks


def tensor_to_int_list(value):
    if value is None:
        return None
    if not torch.is_tensor(value):
        return list(value)
    return [int(v) for v in value.detach().cpu().tolist()]


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
        dual_cache=False,
        replace_position=None,
    ):
        del cache_position
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if past_key_value is not None:
            if dual_cache:
                if replace_position is None:
                    raise ValueError("`replace_position` must be provided when `dual_cache=True`.")
                past_key, past_value = past_key_value
                if replace_position.shape[0] != bsz:
                    raise ValueError("batch size mismatch between hidden states and replace_position")
                for batch_idx in range(bsz):
                    batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
                    if batch_replace_indices.numel() > 0:
                        if batch_replace_indices.numel() != key_states.shape[1]:
                            raise ValueError(
                                "In dual-cache mode, the number of `replace_position` entries must match "
                                "the number of recomputed key/value states."
                            )
                        past_key[batch_idx, batch_replace_indices, :] = key_states[batch_idx]
                        past_value[batch_idx, batch_replace_indices, :] = value_states[batch_idx]
                key_states = past_key
                value_states = past_value
            else:
                past_key, past_value = past_key_value
                key_states = torch.cat([past_key, key_states], dim=-2)
                value_states = torch.cat([past_value, value_states], dim=-2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        if dual_cache:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                replace_position=replace_position,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        if replace_position is not None:
            capture["query_position_ids"] = torch.stack(
                [batch_replace_position.nonzero(as_tuple=True)[0] for batch_replace_position in replace_position],
                dim=0,
            ).detach().clone()
        elif position_ids is not None and position_ids.shape[-1] >= q_len:
            capture["query_position_ids"] = position_ids[:, -q_len:].detach().clone()
        else:
            capture["query_position_ids"] = torch.arange(
                key_states.shape[-2] - q_len,
                key_states.shape[-2],
                device=hidden_states.device,
            ).unsqueeze(0)
        capture["score_sums"] = raw_scores.detach().float().sum(dim=1)

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
    focus_layer=1,
    use_cache=False,
    dual_cache=False,
    focus_decode=False,
    focus_topk=None,
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
        model_attention_mask = None

    if dual_cache and not use_cache:
        raise ValueError("dual_cache=True requires use_cache=True")
    if focus_layer is None:
        raise ValueError("focus_layer must be provided")
    if focus_decode:
        if not use_cache:
            raise ValueError("focus_decode requires use_cache=True")
        if focus_topk is None:
            raise ValueError("focus_decode requires focus_topk")
        if int(focus_topk) <= 0:
            raise ValueError(f"focus_topk must be positive, got {focus_topk}")

    timesteps = torch.linspace(1, model.generation_config.eps, inner_steps + 1, device=device)

    target_block, target_block_index, total_layers = get_target_block(
        model, focus_layer=int(focus_layer)
    )
    restore_callbacks = []
    restore_fn, capture = patch_attention_for_qk(target_block)
    restore_callbacks.append(restore_fn)
    focus_capture = capture
    if focus_decode:
        focus_capture = capture

    step_records = []
    decoded_positions = []
    prev_step_cache = None

    try:
        cache_mode = "dual_cache" if dual_cache else ("use_cache" if use_cache else "no_cache")

        def finalize_step(
            block_id,
            inner_step,
            block_start,
            block_end,
            paired,
            current_step_raw_scores,
            query_position_offset=0,
            query_positions=None,
            focus_step_info=None,
        ):
            nonlocal prev_step_cache

            paired = list(paired)
            if not paired:
                return
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
                    "inner_step": inner_step,
                    "curr_pos": int(curr_pos),
                    "curr_confidence": None if math.isnan(curr_conf) else float(curr_conf),
                    "hit": None,
                    "prev_focus_info": None,
                    "analyzed_layer_index": int(target_block_index),
                    "focus_layer": int(focus_layer),
                    "total_layers": int(total_layers),
                    "cache_mode": cache_mode,
                    "use_cache": bool(use_cache),
                    "dual_cache": bool(dual_cache),
                    "focus_decode": bool(focus_decode),
                    "focus_step_info": focus_step_info,
                }

                if prev_step_cache is not None:
                    prev_raw_scores = prev_step_cache["raw_scores"]
                    prev_decoded_positions = prev_step_cache["decoded_positions"]
                    prev_candidate_positions = prev_step_cache["candidate_positions"]
                    prev_query_position_offset = prev_step_cache.get("query_position_offset", 0)
                    prev_query_positions = prev_step_cache.get("query_positions")
                    prev_focus_info = []
                    hit = None

                    if prev_decoded_positions.numel() > 0 and prev_candidate_positions.numel() > 0:
                        agg_scores = prev_raw_scores[0].sum(dim=0)
                        any_hit = 0

                        for prev_pos in prev_decoded_positions.tolist():
                            prev_query_idx = None
                            if prev_query_positions is not None:
                                matches = (prev_query_positions == prev_pos).nonzero(as_tuple=True)[0]
                                if matches.numel() == 0:
                                    continue
                                prev_query_idx = int(matches[0].item())
                            else:
                                prev_query_idx = prev_pos - prev_query_position_offset
                                if not (0 <= prev_query_idx < agg_scores.shape[0]):
                                    continue

                            valid_candidate_positions = prev_candidate_positions[
                                (prev_candidate_positions >= 0) & (prev_candidate_positions < agg_scores.shape[1])
                            ]
                            if valid_candidate_positions.numel() == 0:
                                continue

                            candidate_scores = agg_scores[prev_query_idx, valid_candidate_positions]
                            if candidate_scores.numel() == 0:
                                continue

                            k_local = min(topk, candidate_scores.numel())
                            top_idx = torch.topk(candidate_scores, k=k_local).indices
                            focus_topk = valid_candidate_positions[top_idx].cpu().tolist()
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
                "query_position_offset": int(query_position_offset),
                "query_positions": (
                    torch.tensor(query_positions, dtype=torch.long)
                    if query_positions is not None
                    else None
                ),
            }

        if use_cache:
            def _trim_past_key_values(cache, end_idx):
                if cache is None or end_idx <= 0:
                    return None
                trimmed = []
                for layer_cache in cache:
                    trimmed.append(tuple(cache_tensor[:, :end_idx, :] for cache_tensor in layer_cache))
                return trimmed

            def _gather_attention_rows(mask, positions):
                if mask is None or mask == "full":
                    return None
                return mask.index_select(2, positions)

            def _select_focus_candidates(prev_score_sums, prev_query_positions, prev_sample_pos, masked_positions):
                if masked_positions.numel() == 0:
                    return masked_positions
                k = min(int(focus_topk), masked_positions.numel())
                if (
                    prev_score_sums is None
                    or prev_query_positions is None
                    or prev_sample_pos is None
                ):
                    return masked_positions[:k]

                matches = (prev_query_positions[0] == int(prev_sample_pos)).nonzero(as_tuple=True)[0]
                if matches.numel() == 0:
                    return masked_positions[:k]

                attention_scores = prev_score_sums[0, matches[0], masked_positions]
                top_indices = torch.topk(attention_scores, k=k).indices
                return masked_positions[top_indices]

            def _sample_focus_logits(candidate_logits):
                if alg == "origin":
                    _, sampled_tokens = sample_tokens(
                        candidate_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=sample_top_k,
                    )
                    confidence = torch.zeros(
                        candidate_logits.shape[0],
                        device=candidate_logits.device,
                        dtype=candidate_logits.dtype,
                    )
                    return confidence, sampled_tokens
                if alg == "maskgit_plus":
                    return sample_tokens(
                        candidate_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=sample_top_k,
                    )
                if alg == "topk_margin":
                    return sample_tokens(
                        candidate_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=sample_top_k,
                        margin_confidence=True,
                    )
                if alg == "entropy":
                    return sample_tokens(
                        candidate_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=sample_top_k,
                        neg_entropy=True,
                    )
                raise RuntimeError(f"Unknown alg: {alg}")

            for block_id in range(num_blocks):
                block_start = prompt.shape[1] + block_id * block_length
                block_end = block_start + block_length

                model_output = model(x, model_attention_mask, position_ids, use_cache=True)
                past_key_values = model_output.past_key_values
                logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                current_step_raw_scores = capture["raw_scores"]
                confidence, x0 = sample_tokens(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=sample_top_k,
                )
                x[:, block_start] = x0[:, block_start]
                prev_focus_scores = None
                prev_focus_query_positions = None
                last_sample_pos = None
                focus_update_pos = deque(maxlen=int(focus_topk)) if focus_decode else None
                focus_replace_position = torch.zeros_like(x, dtype=torch.bool) if focus_decode else None
                focus_row_indices = (
                    torch.arange(x.size(0), device=device).unsqueeze(1) if focus_decode else None
                )
                focus_full_confidence = None
                focus_x_candidate = None
                warmup_focus_info = None
                if focus_decode:
                    prev_focus_scores = focus_capture.get("score_sums")
                    prev_focus_query_positions = focus_capture.get("query_position_ids")
                    last_sample_pos = int(block_start)
                    focus_update_pos.append(int(block_start))
                    warmup_focus_info = {
                        "mode": "warmup",
                        "selected_pos": int(block_start),
                        "sample_positions": None,
                        "compute_positions": None,
                        "focus_update_pos_before": [],
                        "focus_update_pos_after": list(focus_update_pos),
                    }
                finalize_step(
                    block_id,
                    0,
                    block_start,
                    block_end,
                    [(block_start, float(confidence[0, block_start].detach().float().cpu().item()))],
                    current_step_raw_scores,
                    query_position_offset=0,
                    query_positions=list(range(x.shape[1])),
                    focus_step_info=warmup_focus_info,
                )

                if not dual_cache:
                    past_key_values = _trim_past_key_values(past_key_values, block_start)
                    replace_position = None
                else:
                    replace_position = torch.zeros_like(x, dtype=torch.bool)
                    replace_position[:, block_start:block_end] = True

                for i in range(1, inner_steps):
                    current_focus_info = None
                    current_query_positions = None
                    current_step_raw_scores = None
                    if focus_decode:
                        block_mask_index = (x[:, block_start:block_end] == mask_id)
                        if not block_mask_index.any():
                            break

                        t = timesteps[i]
                        s = timesteps[i + 1]
                        masked_positions = torch.where(block_mask_index[0])[0] + block_start
                        sample_positions = _select_focus_candidates(
                            prev_focus_scores,
                            prev_focus_query_positions,
                            last_sample_pos,
                            masked_positions,
                        )
                        if sample_positions.numel() == 0:
                            raise RuntimeError("focus_decode selected no sample_positions")

                        update_positions = torch.tensor(
                            list(focus_update_pos),
                            device=device,
                            dtype=torch.long,
                        )
                        if dual_cache:
                            sample_positions = torch.sort(sample_positions).values
                            compute_positions = torch.cat([update_positions, sample_positions], dim=0)
                            compute_positions = torch.sort(compute_positions).values
                            sample_mask = torch.isin(compute_positions, sample_positions)

                            current_x = x.index_select(1, compute_positions)
                            current_attention_mask = _gather_attention_rows(model_attention_mask, compute_positions)
                            current_position_ids = position_ids if position_ids is not None else None
                            replace_position = focus_replace_position
                            replace_position.zero_()
                            replace_position[:, compute_positions] = True

                            model_output = model(
                                current_x,
                                current_attention_mask,
                                current_position_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                dual_cache=True,
                                replace_position=replace_position,
                            )
                            past_key_values = model_output.past_key_values
                            current_step_raw_scores = capture["raw_scores"]
                            logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                            sample_logits = logits[:, sample_mask]
                            current_query_positions = tensor_to_int_list(compute_positions)
                        else:
                            current_x = x[:, block_start:]
                            current_position_ids = (
                                position_ids[:, block_start:] if position_ids is not None else None
                            )
                            current_attention_mask = _gather_attention_rows(
                                model_attention_mask,
                                torch.arange(block_start, x.shape[1], device=device, dtype=torch.long),
                            )

                            model_output = model(
                                current_x,
                                current_attention_mask,
                                current_position_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            past_key_values = model_output.past_key_values
                            current_step_raw_scores = capture["raw_scores"]
                            logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                            block_logits = logits[:, :block_length]
                            sample_logits = block_logits[:, sample_positions - block_start]
                            current_query_positions = list(range(block_start, x.shape[1]))

                        candidate_logits = sample_logits[0]
                        confidence, sampled_tokens = _sample_focus_logits(candidate_logits)

                        if (
                            focus_full_confidence is None
                            or focus_full_confidence.shape != (x.size(0), block_length)
                            or focus_full_confidence.dtype != sample_logits.dtype
                        ):
                            focus_full_confidence = torch.empty(
                                (x.size(0), block_length),
                                device=device,
                                dtype=sample_logits.dtype,
                            )
                        if focus_x_candidate is None or focus_x_candidate.shape != (x.size(0), block_length):
                            focus_x_candidate = torch.empty(
                                (x.size(0), block_length),
                                device=device,
                                dtype=torch.long,
                            )
                        full_confidence = focus_full_confidence
                        full_confidence.fill_(-torch.inf)
                        relative_sample_positions = sample_positions - block_start
                        full_confidence[0, relative_sample_positions] = confidence

                        x_candidate = focus_x_candidate
                        x_candidate.fill_(mask_id)
                        x_candidate[0, relative_sample_positions] = sampled_tokens

                        if alg_temp is None or alg_temp == 0 or alg == "origin":
                            _, transfer_index = torch.topk(full_confidence, 1)
                        else:
                            sampled_confidence = full_confidence / alg_temp
                            sampled_confidence = F.softmax(sampled_confidence, dim=-1)
                            transfer_index = torch.multinomial(sampled_confidence, num_samples=1)

                        x[:, block_start:block_end][focus_row_indices, transfer_index] = (
                            x_candidate[focus_row_indices, transfer_index]
                        )
                        selected_pos = int((transfer_index[0, 0] + block_start).item())
                        selected_confidence = float(
                            full_confidence[0, transfer_index[0, 0]].detach().float().cpu().item()
                        )
                        focus_update_before = list(focus_update_pos)
                        focus_update_pos.append(selected_pos)
                        last_sample_pos = selected_pos
                        prev_focus_scores = focus_capture.get("score_sums")
                        prev_focus_query_positions = focus_capture.get("query_position_ids")

                        current_transfer_positions = [selected_pos]
                        current_transfer_confidences = [selected_confidence]
                        current_focus_info = {
                            "mode": "focus_decode",
                            "sample_positions": tensor_to_int_list(sample_positions),
                            "compute_positions": current_query_positions,
                            "update_positions": tensor_to_int_list(update_positions),
                            "selected_pos": selected_pos,
                            "focus_update_pos_before": focus_update_before,
                            "focus_update_pos_after": list(focus_update_pos),
                            "last_sample_pos": last_sample_pos,
                        }
                    elif dual_cache:
                        current_x = x[:, block_start:block_end]
                        current_position_ids = position_ids[:, block_start:block_end] if position_ids is not None else None
                    else:
                        current_x = x[:, block_start:]
                        current_position_ids = position_ids[:, block_start:] if position_ids is not None else None

                    if not (dual_cache and focus_decode):
                        if attention_mask is not None and torch.any(attention_mask == 0.0):
                            if dual_cache:
                                current_attention_mask = model_attention_mask[:, :, block_start:block_end, :]
                            else:
                                current_attention_mask = model_attention_mask[:, :, block_start:, :]
                        else:
                            current_attention_mask = model_attention_mask

                        if dual_cache:
                            model_output = model(
                                current_x,
                                current_attention_mask,
                                current_position_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                dual_cache=True,
                                replace_position=replace_position,
                            )
                        else:
                            model_output = model(
                                current_x,
                                current_attention_mask,
                                current_position_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )

                        logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                        current_step_raw_scores = capture["raw_scores"]
                        block_logits = logits[:, :block_length]
                        block_mask_index = (x[:, block_start:block_end] == mask_id)
                        if not block_mask_index.any():
                            break

                        t = timesteps[i]
                        s = timesteps[i + 1]

                        if alg == "origin":
                            block_slice = x[:, block_start:block_end].clone()
                            local_mask_index = (block_slice == mask_id)
                            x0 = torch.full_like(block_slice[local_mask_index], mask_id)
                            p_transfer = 1 - s / t if i < inner_steps - 1 else 1.0
                            transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
                            if transfer_index_t_s.any():
                                _, sampled = sample_tokens(
                                    block_logits[local_mask_index][transfer_index_t_s],
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=sample_top_k,
                                )
                                x0[transfer_index_t_s] = sampled
                            block_positions = torch.where(local_mask_index[0])[0]
                            current_transfer_positions = (block_positions[transfer_index_t_s].detach().cpu() + block_start).tolist()
                            current_transfer_confidences = [float("nan")] * len(current_transfer_positions)
                            block_slice[local_mask_index] = x0.clone()
                            x[:, block_start:block_end] = block_slice
                        else:
                            mask_logits = block_logits[block_mask_index]
                            if mask_logits.numel() == 0:
                                continue

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

                            full_confidence = torch.full_like(
                                x[:, block_start:block_end], -torch.inf, dtype=block_logits.dtype, device=device
                            )
                            full_confidence[block_mask_index] = confidence
                            x_candidate = torch.full_like(x[:, block_start:block_end], mask_id)
                            x_candidate[block_mask_index] = x0.clone()
                            num_mask_token = (block_mask_index.sum() / block_mask_index.shape[0]).detach().float().cpu().item()
                            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < inner_steps - 1 else int(num_mask_token)
                            current_transfer_positions = []
                            current_transfer_confidences = []

                            if number_transfer_tokens > 0:
                                if alg_temp is None or alg_temp == 0:
                                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                                else:
                                    sampled_confidence = full_confidence / alg_temp
                                    sampled_confidence = F.softmax(sampled_confidence, dim=-1)
                                    transfer_index = torch.multinomial(
                                        sampled_confidence, num_samples=number_transfer_tokens
                                    )
                                row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                                x[:, block_start:block_end][row_indices, transfer_index] = x_candidate[row_indices, transfer_index]
                                current_transfer_positions = (transfer_index[0].detach().cpu() + block_start).tolist()
                                current_transfer_confidences = (
                                    full_confidence[0, transfer_index[0]].detach().float().cpu().tolist()
                                )
                        current_query_positions = (
                            list(range(block_start, block_end))
                            if dual_cache
                            else list(range(block_start, x.shape[1]))
                        )

                    finalize_step(
                        block_id,
                        i,
                        block_start,
                        block_end,
                        zip(current_transfer_positions, current_transfer_confidences),
                        current_step_raw_scores,
                        query_position_offset=block_start,
                        query_positions=current_query_positions,
                        focus_step_info=current_focus_info,
                    )

                    if not (x[:, block_start:block_end] == mask_id).any():
                        break
        else:
            for block_id in range(num_blocks):
                block_start = prompt.shape[1] + block_id * block_length
                block_end = block_start + block_length

                for i in range(inner_steps):
                    mask_index = (x == mask_id)
                    if not mask_index.any():
                        break

                    logits = model(x, model_attention_mask, position_ids).logits
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                    current_step_raw_scores = capture["raw_scores"]

                    t = timesteps[i]
                    s = timesteps[i + 1]
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

                        block_positions = torch.where(local_mask_index[0])[0]
                        current_transfer_positions = (block_positions[transfer_index_t_s].detach().cpu() + block_start).tolist()
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
                        num_mask_token = (block_mask_index.sum() / block_mask_index.shape[0]).detach().float().cpu().item()
                        number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < inner_steps - 1 else int(num_mask_token)
                        current_transfer_positions = []
                        current_transfer_confidences = []

                        if number_transfer_tokens > 0:
                            if alg_temp is None or alg_temp == 0:
                                _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                            else:
                                sampled_confidence = full_confidence / alg_temp
                                sampled_confidence = F.softmax(sampled_confidence, dim=-1)
                                transfer_index = torch.multinomial(
                                    sampled_confidence, num_samples=number_transfer_tokens
                                )
                            row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                            x[row_indices, transfer_index] = x_candidate[row_indices, transfer_index]
                            current_transfer_positions = transfer_index[0].detach().cpu().tolist()
                            current_transfer_confidences = (
                                full_confidence[0, transfer_index[0]].detach().float().cpu().tolist()
                            )

                    finalize_step(
                        block_id,
                        i,
                        block_start,
                        block_end,
                        zip(current_transfer_positions, current_transfer_confidences),
                        current_step_raw_scores,
                        query_position_offset=0,
                    )

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
                "cache_mode": cache_mode,
                "use_cache": bool(use_cache),
                "dual_cache": bool(dual_cache),
                "focus_decode": bool(focus_decode),
                "focus_layer": int(focus_layer),
                "focus_topk": None if focus_topk is None else int(focus_topk),
                "analyzed_layer_index": int(target_block_index),
                "total_layers": int(total_layers),
            },
        }
    finally:
        for callback in reversed(restore_callbacks):
            callback()



def parse_args():
    parser = argparse.ArgumentParser()
    default_model_path = str(Path(__file__).resolve().parent / "models" / "Dream-v0-Base-7B")

    parser.add_argument("--model_path", type=str, default=default_model_path)
    parser.add_argument("--prompt_text", type=str, default="Explain why the sky appears blue.")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--focus_layer", type=int, default=1)
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
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="no_cache",
        choices=["no_cache", "use_cache", "dual_cache"],
    )
    parser.add_argument("--focus_decode", action="store_true")
    parser.add_argument("--focus_topk", type=int, default=8)
    parser.add_argument(
        "--analyze_layer_from_end",
        dest="focus_layer",
        type=int,
        help=argparse.SUPPRESS,
    )
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
        focus_layer=args.focus_layer,
        use_cache=args.cache_mode in {"use_cache", "dual_cache"},
        dual_cache=args.cache_mode == "dual_cache",
        focus_decode=args.focus_decode,
        focus_topk=args.focus_topk,
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
