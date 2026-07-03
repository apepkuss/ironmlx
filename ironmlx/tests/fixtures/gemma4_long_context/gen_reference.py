"""Generate Gemma4 12B long-context reference fixtures.

The script intentionally uses cache-based chunked prefill for the long prompt,
then records logits and traces for a single decode step after appending the
first greedy reference token. This mirrors the IronMLX failure mode without
materializing a full 20K+ causal attention mask.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm import load
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.mtp import _mtp_cache_offset_max, _mtp_draft_position


OUT_DIR = Path(__file__).parent
DEFAULT_TOKEN_COUNTS = [18000, 19900, 20000, 24000]
PROMPT_LINE = (
    "You are auditing a production agent trace. Keep every constraint in mind, "
    "compare alternatives carefully, and answer with concise engineering notes.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMMA4_LONG_CONTEXT_MODEL"),
        help="Path to mlx-community/gemma-4-12B-it-4bit snapshot",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        action="append",
        default=None,
        help="Prompt token length to generate. May be repeated.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Greedy tokens to generate for each case.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of logits entries to record in JSON.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Chunk size for cache-based prefill.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for generated fixtures.",
    )
    parser.add_argument(
        "--drafter",
        default=os.environ.get("GEMMA4_LONG_CONTEXT_DRAFTER"),
        help="Optional path to a Gemma4 assistant drafter snapshot.",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=2,
        help="Number of assistant draft tokens to record when --drafter is set.",
    )
    return parser.parse_args()


def tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def tokenizer_decode(tokenizer: Any, tokens: list[int]) -> str:
    try:
        return tokenizer.decode(tokens)
    except Exception:
        return ""


def build_prompt_ids(tokenizer: Any, token_count: int) -> list[int]:
    if token_count <= 1:
        raise ValueError(f"token_count must be > 1, got {token_count}")
    ids: list[int] = []
    block_idx = 0
    while len(ids) < token_count:
        text = f"{block_idx:06d}: {PROMPT_LINE}"
        ids.extend(tokenizer_encode(tokenizer, text))
        block_idx += 1
    return ids[:token_count]


def eval_cache(cache: list[Any]) -> None:
    states = [c.state for c in cache if c is not None and getattr(c, "keys", None) is not None]
    if states:
        mx.eval(states)


def last_logits(language_model: Any, hidden: mx.array) -> mx.array:
    logits = language_model.logits_from_hidden(hidden)
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    return logits.astype(mx.float32)


def top_k_records(logits: mx.array, tokenizer: Any, k: int) -> list[dict[str, Any]]:
    flat = np.asarray(logits.reshape(-1), dtype=np.float32)
    indices = np.argsort(-flat)[:k]
    records = []
    for idx in indices:
        token = int(idx)
        records.append(
            {
                "token": token,
                "score": float(flat[token]),
                "text": tokenizer_decode(tokenizer, [token]),
            }
        )
    return records


def argmax_token(logits: mx.array) -> int:
    return int(mx.argmax(logits.reshape(-1)).item())


def token_list(tokens: mx.array) -> list[int]:
    return [int(token) for token in tokens.reshape(-1).tolist()]


def prefill_and_next_logits(
    language_model: Any,
    prompt_ids: list[int],
    prefill_step_size: int,
) -> tuple[list[Any], mx.array]:
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    cache = language_model.make_cache()

    processed = 0
    while len(prompt) - processed > 1:
        remaining = (len(prompt) - processed) - 1
        n_to_process = min(prefill_step_size, remaining)
        input_tokens = prompt[processed : processed + n_to_process][None]
        _ = language_model.model(input_tokens, cache=cache)
        eval_cache(cache)
        processed += n_to_process
        mx.clear_cache()

    tail = prompt[processed:][None]
    hidden = language_model.model(tail, cache=cache)
    logits = last_logits(language_model, hidden)
    mx.eval(logits)
    return cache, logits


def prefill_for_mtp(
    language_model: Any,
    prompt_ids: list[int],
    prefill_step_size: int,
) -> tuple[list[Any], Any, mx.array, mx.array, int]:
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    cache = language_model.make_cache()

    processed = 0
    while len(prompt) - processed > 1:
        remaining = (len(prompt) - processed) - 1
        n_to_process = min(prefill_step_size, remaining)
        input_tokens = prompt[processed : processed + n_to_process][None]
        _ = language_model.model(input_tokens, cache=cache)
        eval_cache(cache)
        processed += n_to_process
        mx.clear_cache()

    tail = prompt[processed:][None]
    out = language_model(tail, cache=cache, return_hidden=True, return_shared_kv=True)
    prompt_logits = out.logits[:, -1, :].astype(mx.float32)
    draft_hidden = language_model.speculative_draft_hidden(out.hidden_states[-1][:, -1:, :])
    kv_offset = _mtp_cache_offset_max(cache)
    mx.eval(prompt_logits, draft_hidden)
    return cache, out, prompt_logits, draft_hidden, kv_offset


def decode_one(language_model: Any, cache: list[Any], token: int) -> mx.array:
    input_token = mx.array([token], dtype=mx.int32)[None]
    hidden = language_model.model(input_token, cache=cache)
    logits = last_logits(language_model, hidden)
    mx.eval(logits)
    return logits


def generate_tokens(
    language_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    prefill_step_size: int,
) -> tuple[list[int], mx.array]:
    cache, logits = prefill_and_next_logits(language_model, prompt_ids, prefill_step_size)
    generated: list[int] = []
    token = argmax_token(logits)
    for idx in range(max_new_tokens):
        generated.append(token)
        if idx + 1 == max_new_tokens:
            break
        logits = decode_one(language_model, cache, token)
        token = argmax_token(logits)
    return generated, logits


def layer_last_trace_after_append(
    language_model: Any,
    prompt_ids: list[int],
    append_token: int,
    prefill_step_size: int,
) -> mx.array:
    cache, _ = prefill_and_next_logits(language_model, prompt_ids, prefill_step_size)
    capture = []
    layer_count = len(language_model.model.layers)
    input_token = mx.array([append_token], dtype=mx.int32)[None]
    _ = language_model.model(
        input_token,
        cache=cache,
        capture_layer_ids=list(range(layer_count)),
        hidden_sink=capture,
    )
    stacked = mx.stack([h[:, -1, :].astype(mx.float32).reshape(-1) for h in capture], axis=0)
    mx.eval(stacked)
    return stacked


def layer0_stage_trace_after_append(
    language_model: Any,
    prompt_ids: list[int],
    append_token: int,
    prefill_step_size: int,
) -> mx.array:
    cache, _ = prefill_and_next_logits(language_model, prompt_ids, prefill_step_size)
    text_model = language_model.model
    layer = text_model.layers[0]
    input_token = mx.array([append_token], dtype=mx.int32)[None]

    h = text_model.embed_tokens(input_token) * text_model.embed_scale
    full_cache = cache + [None] * (len(text_model.layers) - len(cache))
    masks = text_model._make_masks(h, full_cache)
    per_layer_input = None
    if text_model.hidden_size_per_layer_input:
        per_layer_inputs = text_model.get_per_layer_inputs(input_token)
        per_layer_inputs = text_model.project_per_layer_inputs(h, per_layer_inputs)
        per_layer_input = per_layer_inputs[:, :, 0, :]

    stages = [h[:, -1, :].astype(mx.float32).reshape(-1)]

    residual = h
    h_norm = layer.input_layernorm(h)
    stages.append(h_norm[:, -1, :].astype(mx.float32).reshape(-1))
    attn, _shared_kv, _offset = layer.self_attn(h_norm, masks[0], full_cache[0])
    stages.append(attn[:, -1, :].astype(mx.float32).reshape(-1))
    attn_norm = layer.post_attention_layernorm(attn)
    stages.append(attn_norm[:, -1, :].astype(mx.float32).reshape(-1))
    h = residual + attn_norm
    stages.append(h[:, -1, :].astype(mx.float32).reshape(-1))

    residual = h
    ffn_norm = layer.pre_feedforward_layernorm(h)
    stages.append(ffn_norm[:, -1, :].astype(mx.float32).reshape(-1))
    ffn = layer.mlp(ffn_norm)
    stages.append(ffn[:, -1, :].astype(mx.float32).reshape(-1))
    ffn_normed = layer.post_feedforward_layernorm(ffn)
    stages.append(ffn_normed[:, -1, :].astype(mx.float32).reshape(-1))
    h = residual + ffn_normed
    stages.append(h[:, -1, :].astype(mx.float32).reshape(-1))

    if per_layer_input is not None:
        gate = layer.per_layer_input_gate(h)
        gate = mx.multiply(nn.gelu_approx(gate), per_layer_input)
        gate = layer.per_layer_projection(gate)
        gate = layer.post_per_layer_input_norm(gate)
        h = h + gate

    h = h * layer.layer_scalar
    stages.append(h[:, -1, :].astype(mx.float32).reshape(-1))

    stacked = mx.stack(stages, axis=0)
    mx.eval(stacked)
    return stacked


def save_case(
    model: Any,
    tokenizer: Any,
    token_count: int,
    out_dir: Path,
    max_new_tokens: int,
    top_k: int,
    prefill_step_size: int,
) -> None:
    language_model = model.language_model
    case = f"case_{token_count}"
    prompt_ids = build_prompt_ids(tokenizer, token_count)
    input_ids = mx.array(prompt_ids, dtype=mx.int32)
    mx.save(str(out_dir / f"{case}_input_ids.npy"), input_ids)

    cache, prompt_logits = prefill_and_next_logits(language_model, prompt_ids, prefill_step_size)
    first_token = argmax_token(prompt_logits)
    after_append_logits = decode_one(language_model, cache, first_token)
    after_append_greedy = argmax_token(after_append_logits)
    generated_tokens, _ = generate_tokens(
        language_model,
        prompt_ids,
        max_new_tokens,
        prefill_step_size,
    )
    layer_trace = layer_last_trace_after_append(
        language_model,
        prompt_ids,
        first_token,
        prefill_step_size,
    )
    stage_trace = layer0_stage_trace_after_append(
        language_model,
        prompt_ids,
        first_token,
        prefill_step_size,
    )
    mx.save(str(out_dir / f"{case}_expected_after_append_logits.npy"), after_append_logits)
    mx.save(str(out_dir / f"{case}_expected_layer_last_hiddens.npy"), layer_trace)
    mx.save(str(out_dir / f"{case}_expected_layer0_stage_last_hiddens.npy"), stage_trace)

    expected = {
        "case": case,
        "token_count": token_count,
        "prefill_step_size": prefill_step_size,
        "max_new_tokens": max_new_tokens,
        "prompt_tail_text": tokenizer_decode(tokenizer, prompt_ids[-64:]),
        "prompt_top_k": top_k_records(prompt_logits, tokenizer, top_k),
        "prompt_greedy_token": first_token,
        "prompt_greedy_text": tokenizer_decode(tokenizer, [first_token]),
        "generated_tokens": generated_tokens,
        "generated_text": tokenizer_decode(tokenizer, generated_tokens),
        "after_append_token": first_token,
        "after_append_top_k": top_k_records(after_append_logits, tokenizer, top_k),
        "after_append_greedy_token": after_append_greedy,
        "after_append_greedy_text": tokenizer_decode(tokenizer, [after_append_greedy]),
        "layer_trace_shape": list(layer_trace.shape),
        "layer0_stage_trace_shape": list(stage_trace.shape),
    }
    (out_dir / f"{case}_expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{case}: prompt_greedy={first_token} after_append_greedy={after_append_greedy} "
        f"generated={generated_tokens}"
    )


def save_drafter_case(
    model: Any,
    drafter: Any,
    tokenizer: Any,
    token_count: int,
    out_dir: Path,
    draft_tokens: int,
    prefill_step_size: int,
) -> None:
    if draft_tokens < 1:
        raise ValueError(f"draft_tokens must be >= 1, got {draft_tokens}")

    language_model = model.language_model
    case = f"case_{token_count}"
    input_path = out_dir / f"{case}_input_ids.npy"
    if input_path.exists():
        prompt_ids = [int(token) for token in np.asarray(mx.load(str(input_path))).tolist()]
    else:
        prompt_ids = build_prompt_ids(tokenizer, token_count)
        mx.save(str(input_path), mx.array(prompt_ids, dtype=mx.int32))

    cache, out, prompt_logits, draft_hidden, kv_offset = prefill_for_mtp(
        language_model,
        prompt_ids,
        prefill_step_size,
    )
    first_token = argmax_token(prompt_logits)
    draft_position = int(_mtp_draft_position(kv_offset))

    drafter.reset(model)
    drafter.set_shared_kv(
        out.shared_kv_states,
        kv_offset,
        position=draft_position,
        kv_valid_len=kv_offset,
    )

    position_ids = mx.array([[draft_position]], dtype=mx.int32)

    h_prev = draft_hidden
    tok = mx.array([[first_token]], dtype=mx.int32)
    drafted_tokens = []
    draft_step_top_k = []
    first_logits = None
    for step in range(draft_tokens):
        tok_embed = drafter._input_embed(tok) * drafter._input_embed_scale
        inputs_embeds = mx.concatenate([tok_embed, h_prev], axis=-1)
        h_prev, logits = drafter(inputs_embeds, out.shared_kv_states, position_ids)
        logits = logits.astype(mx.float32)
        mx.eval(h_prev, logits)
        if step == 0:
            first_logits = logits
        next_token = argmax_token(logits)
        drafted_tokens.append(next_token)
        draft_step_top_k.append(top_k_records(logits, tokenizer, 8))
        tok = mx.array([[next_token]], dtype=mx.int32)

    if first_logits is None:
        raise ValueError("drafter produced no logits")
    drafted = mx.array([drafted_tokens], dtype=mx.int32)
    mx.eval(drafted)

    verify_input = mx.concatenate(
        [mx.array([[first_token]], dtype=mx.int32), drafted],
        axis=1,
    )
    verify = language_model(verify_input, cache=cache, return_hidden=True, return_shared_kv=True)
    verified = mx.argmax(verify.logits, axis=-1).astype(mx.int32)
    mx.eval(verified)

    mx.save(str(out_dir / f"{case}_expected_drafter_first_logits.npy"), first_logits)
    expected = {
        "case": case,
        "token_count": token_count,
        "prefill_step_size": prefill_step_size,
        "draft_tokens_budget": draft_tokens,
        "kv_offset": kv_offset,
        "draft_position": draft_position,
        "first_token": first_token,
        "first_token_text": tokenizer_decode(tokenizer, [first_token]),
        "draft_tokens": drafted_tokens,
        "draft_step_top_k": draft_step_top_k,
        "verified_tokens": token_list(verified),
        "draft_first_top_k": draft_step_top_k[0],
    }
    (out_dir / f"{case}_expected_drafter_round.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{case}: drafter first={first_token} draft={expected['draft_tokens']} "
        f"verify={expected['verified_tokens']}"
    )


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("set GEMMA4_LONG_CONTEXT_MODEL or pass --model")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    if args.draft_tokens < 1:
        raise SystemExit("--draft-tokens must be >= 1")
    token_counts = args.tokens or DEFAULT_TOKEN_COUNTS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model: {args.model}")
    model, processor = load(args.model)
    tokenizer = processor.tokenizer
    drafter = None
    if args.drafter:
        print(f"loading drafter: {args.drafter}")
        drafter, kind = load_drafter(args.drafter, kind="mtp")
        if kind != "mtp":
            raise SystemExit(f"expected MTP drafter, got {kind}")

    for token_count in token_counts:
        save_case(
            model,
            tokenizer,
            token_count,
            args.out_dir,
            args.max_new_tokens,
            args.top_k,
            args.prefill_step_size,
        )
        if drafter is not None:
            save_drafter_case(
                model,
                drafter,
                tokenizer,
                token_count,
                args.out_dir,
                args.draft_tokens,
                args.prefill_step_size,
            )


if __name__ == "__main__":
    main()
