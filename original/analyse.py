import math
import json
import argparse
import types
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MASK_ID = 126336
EOS_ID = 126081
EOT_ID = 126348


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


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


def get_all_transformer_blocks(model):
    """
    返回 transformer block 列表，兼容常见 LLaDA HF remote code 结构。

    支持：
    - transformer.blocks
    - transformer.block_groups
    """
    core = model.model
    transformer = core.transformer

    if hasattr(transformer, "blocks"):
        return list(transformer.blocks)

    if hasattr(transformer, "block_groups"):
        flat_blocks = []
        for group in transformer.block_groups:
            flat_blocks.extend(list(group))
        return flat_blocks

    raise RuntimeError("Cannot find transformer blocks.")


def get_target_block(model, analyze_layer_from_end=1):
    """
    选择“倒数第几层”进行分析。
    analyze_layer_from_end = 1 表示最后一层
    analyze_layer_from_end = 2 表示倒数第二层
    """
    blocks = get_all_transformer_blocks(model)
    num_blocks = len(blocks)

    if analyze_layer_from_end < 1 or analyze_layer_from_end > num_blocks:
        raise ValueError(
            f"analyze_layer_from_end must be in [1, {num_blocks}], "
            f"but got {analyze_layer_from_end}"
        )

    target_index = num_blocks - analyze_layer_from_end
    target_block = blocks[target_index]
    return target_block, target_index, num_blocks


def patch_attention_for_qk(target_block):
    """
    在指定 block 的 attention() 上打补丁，
    抓取每次 forward 的最终 raw attention score: QK^T / sqrt(d)
    并且与真实实现保持一致：q_norm/k_norm、RoPE、GQA repeat、attention_bias
    """
    capture = {}
    original_attention = target_block.attention

    def wrapped_attention(self, q, k, v, attention_bias=None, layer_past=None, use_cache=False):
        B, T, C = q.size()
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # [B, T, C] -> [B, H, T, Dh]
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None

        if self.config.rope:
            q, k = self.rotary_emb(q, k)

        # GQA / MQA 对齐到 q-head 数
        num_q_heads = q.size(1)
        num_kv_heads = k.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            repeat_factor = num_q_heads // num_kv_heads
            k_for_scores = k.repeat_interleave(repeat_factor, dim=1, output_size=num_q_heads)
            v_for_attn = v.repeat_interleave(repeat_factor, dim=1, output_size=num_q_heads)
        else:
            k_for_scores = k
            v_for_attn = v

        head_dim = q.size(-1)
        raw_scores = torch.matmul(q, k_for_scores.transpose(-1, -2)) / math.sqrt(head_dim)

        if attention_bias is not None:
            query_len, key_len = q.shape[-2], k_for_scores.shape[-2]
            bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len:key_len, :key_len],
                dtype
            )
            raw_scores = raw_scores + bias

        capture["raw_scores"] = raw_scores.detach().float().cpu()

        att = torch.softmax(raw_scores, dim=-1).to(q.dtype)
        att = torch.matmul(att, v_for_attn)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        out = self.attn_out(att)
        return out, present

    target_block.attention = types.MethodType(wrapped_attention, target_block)

    def restore():
        target_block.attention = original_attention

    return restore, capture


@torch.no_grad()
def generate_and_analyze_prev_focus(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=MASK_ID,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    topk=8,
    analyze_layer_from_end=1,
):
    """
    统计目标：
    对于第 t 步解码出的 token 位置 curr_pos，
    使用“第 t-1 步指定分析层中，第 t-1 步解码出的整批 token 位置”
    的注意力，来看它们关注的 top-k 位置里，是否包含 curr_pos。

    特别地：
    - 保持 LLaDA 原本 step 内并行解码方式不变
    - 如果当前 block 已在本步解完，则下一步分析候选位置自动切到下一个 block
    """
    assert prompt.size(0) == 1, "当前代码按 batch_size=1 写。"
    assert gen_length % block_length == 0, "gen_length 必须能被 block_length 整除。"
    assert steps % (gen_length // block_length) == 0, (
        "steps 必须能被 (gen_length // block_length) 整除。"
    )

    device = model.device
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=device
                )
            ],
            dim=-1
        )

    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    inner_steps = steps // num_blocks

    target_block, target_block_index, total_layers = get_target_block(
        model, analyze_layer_from_end=analyze_layer_from_end
    )
    restore_fn, capture = patch_attention_for_qk(target_block)

    step_records = []
    decoded_positions = []

    # 存“上一 step”的信息：
    # - raw_scores: 上一步该层 attention raw scores
    # - decoded_positions: 上一步并行解码出的整批 token 位置
    # - candidate_positions: 当前步应该分析的候选位置集合
    prev_step_cache = None

    try:
        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = (x[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, inner_steps)

            for i in range(inner_steps):
                mask_index = (x == mask_id)

                # ===== 第 t 步 forward：抓“第 t 步”的 attention，供下一步使用 =====
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    else:
                        attention_mask_ = None

                    logits = model(x_, attention_mask=attention_mask_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits

                current_step_raw_scores = capture["raw_scores"]  # [1, H, Q, K], CPU float

                if logits_eos_inf:
                    logits[:, :, EOS_ID] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, EOS_ID] = -torch.inf
                    logits[:, :, EOT_ID] = -torch.inf

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                        -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # 仍然保持“当前 block 内解码”
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

                # ===== 当前步 t 解码哪些位置 =====
                batch_selected_positions = []
                batch_selected_confidences = []

                for j in range(confidence.shape[0]):
                    k_transfer = int(num_transfer_tokens[j, i].item())
                    _, select_index = torch.topk(confidence[j], k=k_transfer)

                    transfer_index[j, select_index] = True
                    batch_selected_positions.append(select_index.detach().cpu().tolist())
                    batch_selected_confidences.append(
                        confidence[j, select_index].detach().cpu().tolist()
                    )

                # 写回：保持原始并行写回
                x[transfer_index] = x0[transfer_index]

                current_positions = batch_selected_positions[0]
                current_confidences = batch_selected_confidences[0]

                paired = list(zip(current_positions, current_confidences))
                paired.sort(key=lambda z: z[1], reverse=True)

                # ===== 用“上一步缓存的 attention”判断“当前步解码位置” =====
                for curr_pos, curr_conf in paired:
                    record = {
                        "global_step": len(step_records),
                        "block_id": num_block,
                        "inner_step": i,
                        "curr_pos": int(curr_pos),
                        "curr_confidence": float(curr_conf),
                        "hit": None,
                        "prev_focus_info": None,
                        "analyzed_layer_index": int(target_block_index),
                        "analyzed_layer_from_end": int(analyze_layer_from_end),
                        "total_layers": int(total_layers),
                    }

                    if prev_step_cache is not None:
                        prev_raw_scores = prev_step_cache["raw_scores"]          # [1, H, Q, K]
                        prev_decoded_positions = prev_step_cache["decoded_positions"]
                        prev_candidate_positions = prev_step_cache["candidate_positions"]

                        prev_focus_info = []
                        hit = None

                        if prev_decoded_positions.numel() > 0 and prev_candidate_positions.numel() > 0:
                            agg_scores = prev_raw_scores[0].sum(dim=0)  # [Q, K]
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

                                prev_focus_info.append({
                                    "prev_pos": int(prev_pos),
                                    "topk_positions": focus_topk,
                                    "hit": one_hit,
                                })

                                if one_hit:
                                    any_hit = 1

                            hit = any_hit if len(prev_focus_info) > 0 else None

                        record["prev_focus_info"] = prev_focus_info if hit is not None else None
                        record["hit"] = hit

                    step_records.append(record)

                for curr_pos, _ in paired:
                    decoded_positions.append(int(curr_pos))

                # 当前步并行解码出的整批 token 位置
                current_step_decoded_positions = torch.tensor(
                    [int(pos) for pos, _ in paired],
                    dtype=torch.long
                )

                # 先看当前 block 内还剩哪些 mask
                current_block_candidate_positions = (
                    torch.where(x[0, block_start:block_end] == mask_id)[0] + block_start
                ).detach().cpu()

                # 如果当前 block 已解完，则下一步候选切到下一个 block
                if current_block_candidate_positions.numel() > 0:
                    next_step_candidate_positions = current_block_candidate_positions
                else:
                    if num_block + 1 < num_blocks:
                        next_block_start = prompt.shape[1] + (num_block + 1) * block_length
                        next_block_end = prompt.shape[1] + (num_block + 2) * block_length
                        next_step_candidate_positions = (
                            torch.where(x[0, next_block_start:next_block_end] == mask_id)[0] + next_block_start
                        ).detach().cpu()
                    else:
                        next_step_candidate_positions = torch.empty(0, dtype=torch.long)

                prev_step_cache = {
                    "raw_scores": current_step_raw_scores,
                    "decoded_positions": current_step_decoded_positions,
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
                "analyzed_layer_index": int(target_block_index),
                "analyzed_layer_from_end": int(analyze_layer_from_end),
                "total_layers": int(total_layers),
            }
        }

    finally:
        restore_fn()


def parse_args():
    parser = argparse.ArgumentParser()
    default_model_path = str(Path(__file__).resolve().parent / "LLaDA-8B-Base")

    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model_path
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="Explain why the sky appears blue."
    )

    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)

    parser.add_argument(
        "--analyze_layer_from_end",
        type=int,
        default=1,
        help="1=last layer, 2=second last layer, ..."
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random"]
    )

    parser.add_argument("--logits_eos_inf", action="store_true")
    parser.add_argument("--confidence_eos_eot_inf", action="store_true")

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="llada_prev_focus"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    assert tokenizer.pad_token_id != MASK_ID

    encoded = tokenizer(
        [args.prompt_text],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
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
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        topk=args.topk,
        analyze_layer_from_end=args.analyze_layer_from_end,
        logits_eos_inf=args.logits_eos_inf,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
    )

    final_ids = result["final_ids"]
    output_text = tokenizer.batch_decode(
        final_ids[:, input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    print("=" * 80)
    print("Generated text:")
    print(output_text)
    print("=" * 80)
    print("Summary:")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("=" * 80)
    print("First 10 records:")
    for r in result["records"][:10]:
        print(json.dumps(r, ensure_ascii=False))

    summary_path = f"{args.output_prefix}_summary.json"
    records_path = f"{args.output_prefix}_stats.jsonl"

    with open(records_path, "w", encoding="utf-8") as f:
        for r in result["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved records to: {records_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
