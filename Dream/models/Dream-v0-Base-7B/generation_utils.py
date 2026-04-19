# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import types
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0

# retrieve the decoder layer we focus
def _get_focus_decoder_layer(model, focus_layer: int):
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Unable to locate decoder layers for focus_decode.")
    if focus_layer < 1 or focus_layer > len(layers):
        raise ValueError(f"focus_layer must be in [1, {len(layers)}], got {focus_layer}")
    return layers[-focus_layer]


def _patch_attention_for_raw_scores(target_block):
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

        raw_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        if isinstance(attention_mask, torch.Tensor):
            attn_bias = attention_mask[:, :, :, : key_states.shape[-2]] # [B, nh, Lq, Lk]
            if attn_bias.dtype == torch.bool:
                attn_bias = torch.where(
                    attn_bias,
                    torch.zeros((), device=attn_bias.device, dtype=raw_scores.dtype),
                    torch.full((), torch.finfo(raw_scores.dtype).min, device=attn_bias.device, dtype=raw_scores.dtype),
                )
            else:
                attn_bias = attn_bias.to(dtype=raw_scores.dtype)
            raw_scores = raw_scores + attn_bias

        capture["score_sums"] = raw_scores.detach().float().sum(dim=1)

        attn_weights = torch.nn.functional.softmax(raw_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    target_block.self_attn.forward = types.MethodType(wrapped_forward, attention_module)

    def restore():
        target_block.self_attn.forward = original_forward

    return restore, capture


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        # generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        # generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        block_length = kwargs.get("block_length", None)
        use_cache = kwargs.get("use_cache", False)
        dual_cache = kwargs.get("dual_cache", False)
        focus_decode = kwargs.get("focus_decode", False)
        focus_layer = kwargs.get("focus_layer", None)
        focus_topk = kwargs.get("focus_topk", None)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            block_length=block_length,
            use_cache=use_cache,
            dual_cache=dual_cache,
            focus_decode=focus_decode,
            focus_layer=focus_layer,
            focus_topk=focus_topk,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        block_length: Optional[int],
        use_cache: bool,
        dual_cache: bool,
        focus_decode: bool,
        focus_layer: Optional[int],
        focus_topk: Optional[int],
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]

        if block_length is None:
            block_length = gen_length
        if block_length <= 0:
            raise ValueError(f"block_length must be positive, got {block_length}")
        if gen_length % block_length != 0:
            raise ValueError(
                f"Generated length {gen_length} must be divisible by block_length {block_length}"
            )

        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            raise ValueError(
                f"steps {steps} must be divisible by number of blocks {num_blocks}"
            )
        inner_steps = steps // num_blocks
        timesteps = torch.linspace(1, eps, inner_steps + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # this allows user-defined token control of the intermediate steps
        # x = generation_tokens_hook_func(None, x, None)

        restore_attention = None
        focus_capture = None
        if focus_decode:
            if not use_cache:
                raise ValueError("focus_decode requires use_cache=True.")
            if x.shape[0] != 1:
                raise ValueError("focus_decode currently only supports batch_size=1.")
            if focus_layer is None or focus_topk is None:
                raise ValueError("focus_decode requires both focus_layer and focus_topk.")
            if int(focus_topk) <= 0:
                raise ValueError(f"focus_topk must be positive, got {focus_topk}")
            restore_attention, focus_capture = _patch_attention_for_raw_scores(
                _get_focus_decoder_layer(self, int(focus_layer))
            )

        if use_cache:
            def _trim_past_key_values(cache, end_idx):
                if cache is None or end_idx <= 0:
                    return None
                trimmed = []
                for layer_cache in cache:
                    trimmed.append(tuple(cache_tensor[:, :end_idx, :] for cache_tensor in layer_cache))
                return trimmed
            def _gather_attention_rows(mask, query_indices):
                if mask == "full":
                    return mask
                return mask.index_select(2, query_indices)

            def _select_focus_indices(prev_score_sums, prev_compute_indices, last_sample_index, masked_indices):
                k = min(int(focus_topk), masked_indices.numel())
                query_match = (prev_compute_indices[0] == int(last_sample_index)).nonzero(as_tuple=True)[0]
                if query_match.numel() == 0:
                    raise RuntimeError("Unable to align the last sampled index with captured focus attention rows.")

                attention_scores = prev_score_sums[0, query_match[0], masked_indices]
                top_indices = torch.topk(attention_scores, k=k).indices
                return masked_indices[top_indices]

            def _sample_focus_logits(candidate_logits):
                if alg == 'origin':
                    _, sampled_tokens = sample_tokens(
                        candidate_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    confidence = torch.zeros(
                        candidate_logits.shape[0],
                        device=candidate_logits.device,
                        dtype=candidate_logits.dtype,
                    )
                    return confidence, sampled_tokens
                if alg == 'maskgit_plus':
                    return sample_tokens(
                        candidate_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                if alg == 'topk_margin':
                    return sample_tokens(
                        candidate_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        margin_confidence=True,
                    )
                if alg == 'entropy':
                    return sample_tokens(
                        candidate_logits,
                        temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                    )
                raise RuntimeError(f"Unknown alg: {alg}")

            try:
                focus_replace_position = torch.zeros_like(x, dtype=torch.bool) if focus_decode else None
                focus_full_confidence = None
                focus_x_candidate = None
                for block_id in range(num_blocks):
                    current_block_start = input_ids.shape[1] + block_id * block_length
                    current_block_end = current_block_start + block_length
                

                    # 1) KV cache warmup & first sampling TODO: remove warmup
                    model_output = self(x, attention_mask, tok_idx, use_cache=True)
                    past_key_values = model_output.past_key_values
                    logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                    _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    x[:, current_block_start] = x0[:, current_block_start]

                    prev_focus_scores = None
                    prev_focus_compute_indices = None
                    last_sample_index = None
                    focus_update_indices = deque(maxlen=int(focus_topk)) if focus_decode else None
                    if focus_decode:
                        prev_focus_scores = focus_capture.get("score_sums")
                        prev_focus_compute_indices = torch.arange(
                            x.shape[1],
                            device=x.device,
                            dtype=torch.long,
                        ).unsqueeze(0)
                        last_sample_index = int(current_block_start)
                        focus_update_indices.append(int(current_block_start))

                    if not dual_cache:
                        past_key_values = _trim_past_key_values(past_key_values, current_block_start)
                        replace_position = None
                    else:
                        replace_position = torch.zeros_like(x, dtype=torch.bool)
                        replace_position[:, current_block_start:current_block_end] = True

                    for i in range(1, inner_steps):
                        t = timesteps[i]
                        s = timesteps[i + 1]
                        block_mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                        if not block_mask_index.any():
                            break

                        if focus_decode:
                            # 挑选出 sample pos 和 update pos
                            masked_indices = torch.where(block_mask_index[0])[0] + current_block_start
                            sample_indices = _select_focus_indices(
                                prev_focus_scores,
                                prev_focus_compute_indices,
                                last_sample_index,
                                masked_indices,
                            )
                            assert sample_indices.numel() != 0

                            update_indices = torch.tensor(
                                list(focus_update_indices),
                                device=x.device,
                                dtype=torch.long,
                            )
                            if dual_cache:
                                # `sample_indices` stay masked and need decoding; `update_indices` are recent
                                # non-masked tokens whose KV/cache rows are refreshed together with the samples.
                                sample_indices = torch.sort(sample_indices).values
                                compute_indices = torch.cat([update_indices, sample_indices], dim=0)
                                # Keep the query order aligned with `replace_position.nonzero()`, which is also
                                # the order used later for RoPE gathering and KV cache write-back.
                                compute_indices = torch.sort(compute_indices).values
                                sample_mask = torch.isin(compute_indices, sample_indices)

                                current_x = x.index_select(1, compute_indices)
                                current_attention_mask = _gather_attention_rows(attention_mask, compute_indices)
                                current_position_ids = tok_idx if tok_idx is not None else None
                                replace_position = focus_replace_position
                                replace_position.zero_()
                                replace_position[:, compute_indices] = True

                                model_output = self(
                                    current_x,
                                    current_attention_mask,
                                    current_position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                    dual_cache=True,
                                    replace_position=replace_position,
                                )
                                past_key_values = model_output.past_key_values
                                logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                                sample_logits = logits[:, sample_mask]
                                current_compute_indices = compute_indices.unsqueeze(0)
                            else:
                                current_x = x[:, current_block_start:]
                                current_position_ids = (
                                    tok_idx[:, current_block_start:] if tok_idx is not None else None
                                )
                                if attention_mask != "full":
                                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
                                else:
                                    current_attention_mask = attention_mask

                                model_output = self(
                                    current_x,
                                    current_attention_mask,
                                    current_position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                )
                                past_key_values = model_output.past_key_values
                                logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                                block_logits = logits[:, :block_length]
                                sample_logits = block_logits[:, sample_indices - current_block_start]
                                current_compute_indices = torch.arange(
                                    current_block_start,
                                    x.shape[1],
                                    device=x.device,
                                    dtype=torch.long,
                                ).unsqueeze(0)

                            candidate_logits = sample_logits[0]
                            confidence, sampled_tokens = _sample_focus_logits(candidate_logits)

                            if (
                                focus_full_confidence is None
                                or focus_full_confidence.shape != (x.size(0), block_length)
                                or focus_full_confidence.dtype != sample_logits.dtype
                            ):
                                focus_full_confidence = torch.empty(
                                    (x.size(0), block_length),
                                    device=self.device,
                                    dtype=sample_logits.dtype,
                                )
                            if focus_x_candidate is None or focus_x_candidate.shape != (x.size(0), block_length):
                                focus_x_candidate = torch.empty(
                                    (x.size(0), block_length),
                                    device=self.device,
                                    dtype=torch.long,
                                )
                            full_confidence = focus_full_confidence
                            full_confidence.fill_(-torch.inf)
                            relative_sample_indices = sample_indices - current_block_start
                            full_confidence[0, relative_sample_indices] = confidence

                            x_candidate = focus_x_candidate
                            x_candidate.fill_(mask_token_id)
                            x_candidate[0, relative_sample_indices] = sampled_tokens

                            if alg_temp is None or alg_temp == 0 or alg == 'origin':
                                _, transfer_index = torch.topk(full_confidence, 1)
                            else:
                                sampled_confidence = full_confidence / alg_temp
                                sampled_confidence = F.softmax(sampled_confidence, dim=-1)
                                transfer_index = torch.multinomial(sampled_confidence, num_samples=1)

                            row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1)
                            x[:, current_block_start:current_block_end][row_indices, transfer_index] = (
                                x_candidate[row_indices, transfer_index]
                            )
                            selected_index = int((transfer_index[0, 0] + current_block_start).item())
                            focus_update_indices.append(selected_index)
                            last_sample_index = selected_index
                            prev_focus_scores = focus_capture.get("score_sums")
                            prev_focus_compute_indices = current_compute_indices
                        else:
                            if dual_cache:
                                current_x = x[:, current_block_start:current_block_end]
                                current_tok_idx = (
                                    tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None
                                )
                            else:
                                current_x = x[:, current_block_start:]
                                current_tok_idx = tok_idx[:, current_block_start:] if tok_idx is not None else None

                            if attention_mask != "full":
                                if dual_cache:
                                    current_attention_mask = attention_mask[:, :, current_block_start:current_block_end, :]
                                else:
                                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
                            else:
                                current_attention_mask = attention_mask

                            if dual_cache:
                                model_output = self(
                                    current_x,
                                    current_attention_mask,
                                    current_tok_idx,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                    dual_cache=True,
                                    replace_position=replace_position,
                                )
                            else:
                                model_output = self(
                                    current_x,
                                    current_attention_mask,
                                    current_tok_idx,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                )

                            logits = torch.cat([model_output.logits[:, :1], model_output.logits[:, :-1]], dim=1)
                            block_logits = logits[:, :block_length]

                            if alg == 'origin':
                                block_slice = x[:, current_block_start:current_block_end].clone()
                                local_mask_index = (block_slice == mask_token_id)
                                x0 = torch.zeros_like(
                                    block_slice[local_mask_index],
                                    device=self.device,
                                    dtype=torch.long,
                                ) + mask_token_id
                                p_transfer = 1 - s / t if i < inner_steps - 1 else 1
                                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                                if transfer_index_t_s.any():
                                    _, x0[transfer_index_t_s] = sample_tokens(
                                        block_logits[local_mask_index][transfer_index_t_s],
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                    )
                                block_slice[local_mask_index] = x0.clone()
                                x[:, current_block_start:current_block_end] = block_slice
                            else:
                                mask_logits = block_logits[block_mask_index]

                                if alg == 'maskgit_plus':
                                    confidence, x0 = sample_tokens(
                                        mask_logits,
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                    )
                                elif alg == 'topk_margin':
                                    confidence, x0 = sample_tokens(
                                        mask_logits,
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                        margin_confidence=True,
                                    )
                                elif alg == 'entropy':
                                    confidence, x0 = sample_tokens(
                                        mask_logits,
                                        temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                        neg_entropy=True,
                                    )
                                else:
                                    raise RuntimeError(f"Unknown alg: {alg}")

                                full_confidence = torch.full_like(
                                    x[:, current_block_start:current_block_end],
                                    -torch.inf,
                                    device=self.device,
                                    dtype=block_logits.dtype,
                                )
                                full_confidence[block_mask_index] = confidence

                                x_ = torch.zeros_like(
                                    x[:, current_block_start:current_block_end],
                                    device=self.device,
                                    dtype=torch.long,
                                ) + mask_token_id
                                x_[block_mask_index] = x0.clone()

                                num_mask_token = block_mask_index.sum() / block_mask_index.shape[0]
                                number_transfer_tokens = (
                                    int(num_mask_token * (1 - s / t)) if i < inner_steps - 1 else int(num_mask_token)
                                )

                                if number_transfer_tokens > 0:
                                    if alg_temp is None or alg_temp == 0:
                                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                                    else:
                                        sampled_confidence = full_confidence / alg_temp
                                        sampled_confidence = F.softmax(sampled_confidence, dim=-1)
                                        transfer_index = torch.multinomial(
                                            sampled_confidence, num_samples=number_transfer_tokens
                                        )

                                    row_indices = (
                                        torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                                    )
                                    x[:, current_block_start:current_block_end][row_indices, transfer_index] = (
                                        x_[row_indices, transfer_index]
                                    )

                        if histories is not None:
                            histories.append(x.clone())

                        if not (x[:, current_block_start:current_block_end] == mask_token_id).any():
                            break
            finally:
                if restore_attention is not None:
                    restore_attention()

            if return_dict_in_generate:
                return DreamModelOutput(
                    sequences=x,
                    history=histories,
                )
            else:
                return x

        for block_id in range(num_blocks):
            block_start = input_ids.shape[1] + block_id * block_length
            block_end = block_start + block_length

            for i in range(inner_steps):
                mask_index = (x == mask_token_id)
                logits = self(x, attention_mask, tok_idx).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # this allows user-defined logits control of the intermediate steps
                # logits = generation_logits_hook_func(global_step, x, logits)

                t = timesteps[i]
                s = timesteps[i + 1]
                block_mask_index = mask_index.clone()
                block_mask_index[:, :block_start] = False
                block_mask_index[:, block_end:] = False

                if alg == 'origin':
                    p_transfer = 1 - s / t if i < inner_steps - 1 else 1
                    block_slice = x[:, block_start:block_end].clone()
                    local_mask_index = (block_slice == mask_token_id)
                    x0 = torch.zeros_like(
                        block_slice[local_mask_index],
                        device=self.device,
                        dtype=torch.long,
                    ) + mask_token_id
                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    if transfer_index_t_s.any():
                        _, x0[transfer_index_t_s] = sample_tokens(
                            logits[:, block_start:block_end][local_mask_index][transfer_index_t_s],
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    block_slice[local_mask_index] = x0.clone()
                    x[:, block_start:block_end] = block_slice
                else:
                    mask_logits = logits[block_mask_index]

                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            margin_confidence=True,
                        )
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature,
                            top_p=top_p,
                            top_k=top_k,
                            neg_entropy=True,
                        )
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")

                    full_confidence = torch.full_like(
                        x,
                        -torch.inf,
                        device=self.device,
                        dtype=logits.dtype,
                    )
                    full_confidence[block_mask_index] = confidence

                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[block_mask_index] = x0.clone()

                    num_mask_token = block_mask_index.sum() / block_mask_index.shape[0]
                    number_transfer_tokens = (
                        int(num_mask_token * (1 - s / t)) if i < inner_steps - 1 else int(num_mask_token)
                    )

                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            sampled_confidence = full_confidence / alg_temp
                            sampled_confidence = F.softmax(sampled_confidence, dim=-1)
                            transfer_index = torch.multinomial(
                                sampled_confidence, num_samples=number_transfer_tokens
                            )

                        row_indices = (
                            torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        )
                        x[row_indices, transfer_index] = x_[row_indices, transfer_index]

                # this allows user-defined token control of the intermediate steps
                # x = generation_tokens_hook_func(global_step, x, logits)

                if histories is not None:
                    histories.append(x.clone())

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x
