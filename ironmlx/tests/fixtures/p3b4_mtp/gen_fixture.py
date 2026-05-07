"""Generate P3b4 MTP fixtures.

Independent re-implementation of vllm-mlx's `_MTPModule.mtp_forward` algorithm
(`/Volumes/Dev/vllm-mlx/vllm_mlx/patches/qwen3_5_mtp.py:369-391`) using `mlx.core`
primitives only. Outputs `.npy` files alongside this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

EXPECTED_MLX_VERSION = "0.31.1"
_mlx_version = mx.__version__
if _mlx_version != EXPECTED_MLX_VERSION:
    raise SystemExit(
        f"mlx version mismatch: got {_mlx_version}, expected "
        f"{EXPECTED_MLX_VERSION}. Bump and regenerate the .npy fixtures."
    )

OUT_DIR = Path(__file__).parent

# ---- Small synthetic config (matches MtpConfig in p3b4_mtp.rs) ----
B = 1
S = 4
HIDDEN = 32
HQ = 4
HKV = 2
D = 8
INTERMEDIATE = 64
NUM_MTP_LAYERS = 1
RMS_EPS = 1e-6
PARTIAL = 1.0
ROT_DIM = int(D * PARTIAL) & ~1   # 8
HALF = ROT_DIM // 2               # 4
SECTIONS = [2, 1, 1]              # sum = HALF
THETA = 1e7


def _build_inv_freq() -> mx.array:
    idx = mx.arange(0, HALF, dtype=mx.float32)
    return mx.exp(-(idx * (2.0 / ROT_DIM)) * float(np.log(THETA)))


def _build_position_ids() -> mx.array:
    one = mx.arange(0, S, dtype=mx.int32).reshape((1, 1, S))
    return mx.broadcast_to(one, (3, B, S))


def _ref_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    pos_f = position_ids.astype(mx.float32)
    pos_unsq = pos_f[..., None]
    inv_unsq = inv_freq.reshape((1, 1, 1, -1))
    freqs = pos_unsq * inv_unsq
    cos_per = mx.cos(freqs)
    sin_per = mx.sin(freqs)
    offsets = [0]
    for n in SECTIONS:
        offsets.append(offsets[-1] + n)
    cos_segs = []
    sin_segs = []
    for s, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        cos_segs.append(cos_per[s, :, :, lo:hi])
        sin_segs.append(sin_per[s, :, :, lo:hi])
    return mx.concatenate(cos_segs, axis=-1), mx.concatenate(sin_segs, axis=-1)


def _ref_apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    rot = x[..., :ROT_DIM]
    tail = x[..., ROT_DIM:]
    even = rot[..., 0::2]
    odd = rot[..., 1::2]
    c = cos[:, None, :, :]
    s = sin[:, None, :, :]
    rot_even = (even.astype(mx.float32) * c - odd.astype(mx.float32) * s).astype(x.dtype)
    rot_odd = (even.astype(mx.float32) * s + odd.astype(mx.float32) * c).astype(x.dtype)
    out_rot = mx.stack([rot_even, rot_odd], axis=-1).reshape(x.shape[:-1] + (ROT_DIM,))
    return mx.concatenate([out_rot, tail], axis=-1)


def _ref_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _ref_gated_attention(
    x: mx.array,
    q_w: mx.array, k_w: mx.array, v_w: mx.array, o_w: mx.array,
    q_norm_w: mx.array, k_norm_w: mx.array,
    cos: mx.array, sin: mx.array,
) -> mx.array:
    """Independent re-impl of Qwen3NextAttention (causal, no cache, no mask)."""
    q_full = x @ q_w.T
    k = x @ k_w.T
    v = x @ v_w.T
    q_per_head = q_full.reshape((B, S, HQ, D * 2))
    queries, gate = mx.split(q_per_head, 2, axis=-1)
    gate_flat = gate.reshape((B, S, HQ * D))
    queries = _ref_rms_norm(queries, q_norm_w, RMS_EPS)
    queries = queries.transpose(0, 2, 1, 3)
    k = k.reshape((B, S, HKV, D))
    k = _ref_rms_norm(k, k_norm_w, RMS_EPS)
    k = k.transpose(0, 2, 1, 3)
    v = v.reshape((B, S, HKV, D)).transpose(0, 2, 1, 3)
    queries = _ref_apply_rope(queries, cos, sin)
    k = _ref_apply_rope(k, cos, sin)
    scale = D ** -0.5
    sdpa_out = mx.fast.scaled_dot_product_attention(queries, k, v, scale=scale, mask="causal")
    sdpa_flat = sdpa_out.transpose(0, 2, 1, 3).reshape((B, S, HQ * D))
    gated = sdpa_flat * mx.sigmoid(gate_flat)
    return gated @ o_w.T


def _ref_mlp(
    x: mx.array,
    gate_w: mx.array, up_w: mx.array, down_w: mx.array,
) -> mx.array:
    """SwiGLU: down( silu(gate(x)) * up(x) )."""
    g = x @ gate_w.T
    u = x @ up_w.T
    g_sig = mx.sigmoid(g)
    activated = g * g_sig * u
    return activated @ down_w.T


def _ref_decoder_layer(
    x: mx.array,
    in_ln_w: mx.array,
    q_w: mx.array, k_w: mx.array, v_w: mx.array, o_w: mx.array,
    q_norm_w: mx.array, k_norm_w: mx.array,
    post_ln_w: mx.array,
    mlp_gate_w: mx.array, mlp_up_w: mx.array, mlp_down_w: mx.array,
    cos: mx.array, sin: mx.array,
) -> mx.array:
    """Mirrors ironmlx::nn::DecoderLayer::forward (full-attn path, no cache, no mask)."""
    normed_in = _ref_rms_norm(x, in_ln_w, RMS_EPS)
    attn = _ref_gated_attention(
        normed_in, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w, cos, sin,
    )
    h = x + attn
    normed_post = _ref_rms_norm(h, post_ln_w, RMS_EPS)
    mlp_out = _ref_mlp(normed_post, mlp_gate_w, mlp_up_w, mlp_down_w)
    return h + mlp_out


def _ref_mtp(
    hidden: mx.array,
    next_embeds: mx.array,
    pre_fc_norm_hidden_w: mx.array,
    pre_fc_norm_embedding_w: mx.array,
    fc_w: mx.array,
    # layer 0 weights
    in_ln_w: mx.array,
    q_w: mx.array, k_w: mx.array, v_w: mx.array, o_w: mx.array,
    q_norm_w: mx.array, k_norm_w: mx.array,
    post_ln_w: mx.array,
    mlp_gate_w: mx.array, mlp_up_w: mx.array, mlp_down_w: mx.array,
    norm_w: mx.array,
    cos: mx.array, sin: mx.array,
) -> mx.array:
    """Mirrors ironmlx::nn::Mtp::forward."""
    h = _ref_rms_norm(hidden, pre_fc_norm_hidden_w, RMS_EPS)
    e = _ref_rms_norm(next_embeds, pre_fc_norm_embedding_w, RMS_EPS)
    concat = mx.concatenate([e, h], axis=-1)  # [B, S, 2H]; ORDER [e, h] is pinned.
    x = concat @ fc_w.T                       # [B, S, H]
    x = _ref_decoder_layer(
        x, in_ln_w,
        q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
        post_ln_w,
        mlp_gate_w, mlp_up_w, mlp_down_w,
        cos, sin,
    )
    return _ref_rms_norm(x, norm_w, RMS_EPS)


def main() -> None:
    np.random.seed(46)

    inv_freq = _build_inv_freq()
    position_ids = _build_position_ids()
    cos, sin = _ref_cos_sin(position_ids, inv_freq)

    def randn(shape, dtype=mx.bfloat16, scale=0.1):
        a = np.random.randn(*shape).astype(np.float32) * scale
        return mx.array(a).astype(dtype)

    # Inputs.
    hidden = randn((B, S, HIDDEN), dtype=mx.bfloat16)
    next_embeds = randn((B, S, HIDDEN), dtype=mx.bfloat16)

    # Mtp top-level weights.
    pre_fc_norm_hidden_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    pre_fc_norm_embedding_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    fc_w = randn((HIDDEN, 2 * HIDDEN), dtype=mx.bfloat16)

    # Layer 0 weights.
    in_ln_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    q_w = randn((HQ * D * 2, HIDDEN), dtype=mx.bfloat16)
    k_w = randn((HKV * D, HIDDEN), dtype=mx.bfloat16)
    v_w = randn((HKV * D, HIDDEN), dtype=mx.bfloat16)
    o_w = randn((HIDDEN, HQ * D), dtype=mx.bfloat16)
    q_norm_w = randn((D,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    k_norm_w = randn((D,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    post_ln_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    mlp_gate_w = randn((INTERMEDIATE, HIDDEN), dtype=mx.bfloat16)
    mlp_up_w = randn((INTERMEDIATE, HIDDEN), dtype=mx.bfloat16)
    mlp_down_w = randn((HIDDEN, INTERMEDIATE), dtype=mx.bfloat16)
    norm_w = randn((HIDDEN,), dtype=mx.float32, scale=0.5) + mx.array([1.0])

    out = _ref_mtp(
        hidden, next_embeds,
        pre_fc_norm_hidden_w, pre_fc_norm_embedding_w, fc_w,
        in_ln_w,
        q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
        post_ln_w,
        mlp_gate_w, mlp_up_w, mlp_down_w,
        norm_w,
        cos, sin,
    )

    mx.eval(
        cos, sin, hidden, next_embeds,
        pre_fc_norm_hidden_w, pre_fc_norm_embedding_w, fc_w,
        in_ln_w, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
        post_ln_w, mlp_gate_w, mlp_up_w, mlp_down_w, norm_w,
        out,
    )

    def save(name: str, arr) -> None:
        path = OUT_DIR / f"{name}.npy"
        mx.save(str(path), arr)
        print(f"  wrote {path.name}: shape={arr.shape} dtype={arr.dtype}")

    save("input_hidden", hidden)
    save("input_next_embeds", next_embeds)
    save("input_position_ids", position_ids)
    save("input_inv_freq", inv_freq)
    save("pre_fc_norm_hidden_weight", pre_fc_norm_hidden_w)
    save("pre_fc_norm_embedding_weight", pre_fc_norm_embedding_w)
    save("fc_weight", fc_w)
    save("layer0_input_layernorm_weight", in_ln_w)
    save("layer0_q_proj_weight", q_w)
    save("layer0_k_proj_weight", k_w)
    save("layer0_v_proj_weight", v_w)
    save("layer0_o_proj_weight", o_w)
    save("layer0_q_norm_weight", q_norm_w)
    save("layer0_k_norm_weight", k_norm_w)
    save("layer0_post_attention_layernorm_weight", post_ln_w)
    save("layer0_mlp_gate_proj_weight", mlp_gate_w)
    save("layer0_mlp_up_proj_weight", mlp_up_w)
    save("layer0_mlp_down_proj_weight", mlp_down_w)
    save("norm_weight", norm_w)
    save("expected_cos", cos)
    save("expected_sin", sin)
    save("expected_mtp_out", out)


if __name__ == "__main__":
    main()
