"""Generate P3b2 Gated Full Attention fixtures.

Independent re-implementation of mlx-lm's Qwen3NextAttention algorithm using
`mlx.core` primitives. Outputs `.npy` files alongside this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

# Pin MLX version. Bump and regenerate after upgrade.
EXPECTED_MLX_VERSION = "0.31.1"
_mlx_version = mx.__version__
if _mlx_version != EXPECTED_MLX_VERSION:
    raise SystemExit(
        f"mlx version mismatch: got {_mlx_version}, expected "
        f"{EXPECTED_MLX_VERSION}. Bump and regenerate the .npy fixtures."
    )

OUT_DIR = Path(__file__).parent

# ---- Small synthetic config ----
B = 1
S = 4
HQ = 4
HKV = 2
D = 8
HIDDEN = HQ * D  # 32
PARTIAL = 1.0
ROT_DIM = int(D * PARTIAL) & ~1  # 8
HALF = ROT_DIM // 2  # 4
SECTIONS = [2, 1, 1]  # sum = 4 = HALF
THETA = 1e7


def _build_inv_freq() -> mx.array:
    idx = mx.arange(0, HALF, dtype=mx.float32)
    return mx.exp(-(idx * (2.0 / ROT_DIM)) * float(np.log(THETA)))


def _build_position_ids() -> mx.array:
    one = mx.arange(0, S, dtype=mx.int32).reshape((1, 1, S))
    return mx.broadcast_to(one, (3, B, S))


def _ref_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    """Reference MRoPE cos/sin from RoPE first principles (standard
    `[B, S, ROT_DIM]` layout via `concat([freqs, freqs])`).

      freqs[s, b, t, i] = pos[s, b, t] * inv_freq[i]   for i in [0, HALF)
      freqs_t[b, t, i]  = freqs[slot_stream[i], b, t, i]  (MRoPE section select)
      emb               = concatenate([freqs_t, freqs_t], axis=-1)  -> [..., ROT_DIM]
      cos = cos(emb), sin = sin(emb)

    Output slot i draws its frequency from the modality stream owning section i;
    the source column index is the same `i`. For text-only prompts (identical
    streams) this collapses to a single stream, but the general selection is
    implemented to stay faithful to the multimodal algorithm. Returns the full
    `ROT_DIM` (not HALF) so downstream `rotate_half` reads the full rotated span.
    """
    pos_f = position_ids.astype(mx.float32)
    pos_unsq = pos_f[..., None]                 # [n_streams, B, S, 1]
    inv_unsq = inv_freq.reshape((1, 1, 1, -1))  # [1, 1, 1, HALF]
    freqs = pos_unsq * inv_unsq                 # [n_streams, B, S, HALF]

    n_streams = len(SECTIONS)
    slot_stream = [0] * HALF
    for s, sect_len in enumerate(SECTIONS):
        for k in range(sect_len):
            slot_stream[s + k * n_streams] = s

    freq_slots = [freqs[slot_stream[i], :, :, i : i + 1] for i in range(HALF)]
    freqs_t = mx.concatenate(freq_slots, axis=-1)  # [B, S, HALF]

    emb = mx.concatenate([freqs_t, freqs_t], axis=-1)  # [B, S, ROT_DIM]
    return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    """Standard RoPE rotate_half: rotate_half([x1, x2]) = [-x2, x1]."""
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _ref_apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply split-half RoPE rotation to `x`'s rotary slice; pass-through tail.

    Standard rotate_half formula (derived from RoPE math, not transcribed):
        y = x * cos + rotate_half(x) * sin
    per channel d in [0, ROT_DIM) with ROT_HALF = ROT_DIM/2:
        d <  ROT_HALF: y[d] = x[d]*cos[d] - x[d + ROT_HALF]*sin[d]
        d >= ROT_HALF: y[d] = x[d]*cos[d] + x[d - ROT_HALF]*sin[d]
    Channels d in [ROT_DIM, HEAD_DIM) pass through unchanged.
    """
    rot = x[..., :ROT_DIM]
    tail = x[..., ROT_DIM:]
    c = cos[:, None, :, :]  # [B, S, ROT_DIM] -> [B, 1, S, ROT_DIM]
    s = sin[:, None, :, :]
    rot_f = rot.astype(mx.float32)
    out_rot = (rot_f * c + _rotate_half(rot_f) * s).astype(x.dtype)
    return mx.concatenate([out_rot, tail], axis=-1)


def _ref_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Wrapper around MLX's fused rms_norm — same kernel as Rust's RmsNorm::forward."""
    return mx.fast.rms_norm(x, weight, eps)


def _ref_gated_attention(
    x: mx.array,
    q_w: mx.array,
    k_w: mx.array,
    v_w: mx.array,
    o_w: mx.array,
    q_norm_w: mx.array,
    k_norm_w: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    """Independent re-impl of Qwen3NextAttention (no cache, no mask, causal)."""
    q_full = x @ q_w.T
    k = x @ k_w.T
    v = x @ v_w.T

    q_per_head = q_full.reshape((B, S, HQ, D * 2))
    queries, gate = mx.split(q_per_head, 2, axis=-1)
    gate_flat = gate.reshape((B, S, HQ * D))

    queries = _ref_rms_norm(queries, q_norm_w, 1e-6)
    queries = queries.transpose(0, 2, 1, 3)

    k = k.reshape((B, S, HKV, D))
    k = _ref_rms_norm(k, k_norm_w, 1e-6)
    k = k.transpose(0, 2, 1, 3)

    v = v.reshape((B, S, HKV, D)).transpose(0, 2, 1, 3)

    queries = _ref_apply_rope(queries, cos, sin)
    k = _ref_apply_rope(k, cos, sin)

    scale = D**-0.5
    sdpa_out = mx.fast.scaled_dot_product_attention(queries, k, v, scale=scale, mask="causal")

    sdpa_flat = sdpa_out.transpose(0, 2, 1, 3).reshape((B, S, HQ * D))
    gated = sdpa_flat * mx.sigmoid(gate_flat)
    out = gated @ o_w.T
    return out


def main() -> None:
    np.random.seed(45)

    inv_freq = _build_inv_freq()
    position_ids = _build_position_ids()
    cos, sin = _ref_cos_sin(position_ids, inv_freq)

    def randn(shape, dtype=mx.bfloat16, scale=0.1):
        a = np.random.randn(*shape).astype(np.float32) * scale
        return mx.array(a).astype(dtype)

    x = randn((B, S, HIDDEN))
    q_w = randn((HQ * D * 2, HIDDEN))   # [64, 32]
    k_w = randn((HKV * D, HIDDEN))      # [16, 32]
    v_w = randn((HKV * D, HIDDEN))      # [16, 32]
    o_w = randn((HIDDEN, HQ * D))       # [32, 32]
    q_norm_w = randn((D,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    k_norm_w = randn((D,), dtype=mx.float32, scale=0.5) + mx.array([1.0])

    out = _ref_gated_attention(x, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w, cos, sin)

    mx.eval(cos, sin, x, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w, out)

    def save(name: str, arr) -> None:
        path = OUT_DIR / f"{name}.npy"
        mx.save(str(path), arr)
        print(f"  wrote {path.name}: shape={arr.shape} dtype={arr.dtype}")

    save("input_x", x)
    save("input_position_ids", position_ids)
    save("input_inv_freq", inv_freq)
    save("q_proj_weight", q_w)
    save("k_proj_weight", k_w)
    save("v_proj_weight", v_w)
    save("o_proj_weight", o_w)
    save("q_norm_weight", q_norm_w)
    save("k_norm_weight", k_norm_w)
    save("expected_cos", cos)
    save("expected_sin", sin)
    save("expected_gated_attn_out", out)


if __name__ == "__main__":
    main()
