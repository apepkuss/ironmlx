"""Generate P3b1 MRoPE fixtures from a Python reference implementation.

This is an independent re-implementation of the MRoPE algorithm using
mlx.core primitives -- it does NOT use mlx-lm's nn.RoPE (which is a no-op
for the mrope path). The reference matches the algorithm used by IronMLX.

Outputs go alongside this script as `.npy` files. Re-run to regenerate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

# Pin the MLX version this fixture was generated against. Bump this constant
# (and regenerate via `python gen_fixture.py`) whenever upgrading MLX so the
# Rust integration tests stay aligned with the exact numerical behavior of
# the Python reference.
EXPECTED_MLX_VERSION = "0.31.1"

_mlx_version = mx.__version__
if _mlx_version != EXPECTED_MLX_VERSION:
    raise SystemExit(
        f"mlx version mismatch: got {_mlx_version}, expected "
        f"{EXPECTED_MLX_VERSION}. Bump EXPECTED_MLX_VERSION in this file "
        f"and regenerate the .npy fixtures."
    )

OUT_DIR = Path(__file__).parent

# ---- Qwen3.5 MRoPE constants (per the model config) ----
HEAD_DIM = 256
PARTIAL = 0.25
ROT_DIM = int(HEAD_DIM * PARTIAL) & ~1  # 64
HALF = ROT_DIM // 2  # 32
SECTIONS = [11, 11, 10]
INTERLEAVED = True
THETA = 1e7
HQ = 64
HKV = 8
B = 1
S = 8


def build_inv_freq() -> mx.array:
    # inv_freq[i] = 1 / theta^(2i / rot_dim) for i in [0, half)
    idx = mx.arange(0, HALF, dtype=mx.float32)
    return mx.exp(-(idx * (2.0 / ROT_DIM)) * float(np.log(THETA)))


def build_position_ids() -> mx.array:
    # Text-only: 3 identical streams [0, 1, ..., S-1]
    one = mx.arange(0, S, dtype=mx.int32).reshape((1, 1, S))
    return mx.broadcast_to(one, (3, B, S))


def reference_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    """Reference MRoPE cos/sin -- independent re-implementation from RoPE first
    principles (standard `[B, S, rot_dim]` layout via `concat([freqs, freqs])`).

    Standard RoPE:
      freqs[s, b, t, i] = pos[s, b, t] * inv_freq[i]   for i in [0, HALF)
      emb               = concatenate([freqs_t, freqs_t], axis=-1)  -> [..., ROT_DIM]
      cos = cos(emb), sin = sin(emb)

    MRoPE adds a per-section stream selection: output slot `i` (for i in [0, HALF))
    draws its frequency from the modality stream that owns section `i`. The source
    column index into `freqs[stream]` is the SAME `i` as the destination slot. For
    text-only prompts the three streams carry identical position ids, so the
    selection collapses to a single stream -- but we implement the general
    selection anyway so the reference stays faithful to the multimodal algorithm.

    The full `ROT_DIM` (not HALF) is returned: `emb` duplicates the per-position
    `freqs_t` block so that downstream `rotate_half` reads cos[d]/sin[d] for the
    full rotated span. Returned shape is `[B, S, ROT_DIM]`.
    """
    pos_f = position_ids.astype(mx.float32)
    pos_unsq = pos_f[..., None]                 # [n_streams, B, S, 1]
    inv_unsq = inv_freq.reshape((1, 1, 1, -1))  # [1, 1, 1, HALF]
    freqs = pos_unsq * inv_unsq                 # [n_streams, B, S, HALF]

    # MRoPE per-section stream selection. slot_stream[i] is the stream index
    # whose freqs[stream, :, :, i] feeds output slot i. Sections are filled
    # round-robin across streams: stream s owns slots {s, s+n_streams, ...}
    # up to SECTIONS[s] entries (source/destination column index identical).
    n_streams = len(SECTIONS)
    slot_stream = [0] * HALF
    for s, sect_len in enumerate(SECTIONS):
        for k in range(sect_len):
            slot_stream[s + k * n_streams] = s

    freq_slots = [freqs[slot_stream[i], :, :, i : i + 1] for i in range(HALF)]
    freqs_t = mx.concatenate(freq_slots, axis=-1)  # [B, S, HALF]

    # Duplicate to full ROT_DIM: emb = concat([freqs_t, freqs_t]).
    emb = mx.concatenate([freqs_t, freqs_t], axis=-1)  # [B, S, ROT_DIM]
    cos = mx.cos(emb)
    sin = mx.sin(emb)
    return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    """Standard RoPE `rotate_half`: split the last axis in two and rotate.

      rotate_half([x1, x2]) = [-x2, x1]   (x1, x2 each ROT_DIM/2 wide)
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def reference_apply(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply split-half RoPE rotation to `x` (Q or K), tail pass-through.

    Standard `rotate_half` formula (independent of any Rust impl):
        y = x * cos + rotate_half(x) * sin
    expanded per channel d in [0, ROT_DIM) with ROT_HALF = ROT_DIM/2:
        d <  ROT_HALF: y[d] = x[d]*cos[d] - x[d + ROT_HALF]*sin[d]
        d >= ROT_HALF: y[d] = x[d]*cos[d] + x[d - ROT_HALF]*sin[d]
    Channels d in [ROT_DIM, HEAD_DIM) pass through unchanged.
    """
    # x: [B, H, S, HEAD_DIM]; cos/sin: [B, S, ROT_DIM] (fp32).
    rot = x[..., :ROT_DIM]
    tail = x[..., ROT_DIM:]

    # Broadcast cos/sin across heads: [B, S, ROT_DIM] -> [B, 1, S, ROT_DIM].
    c = cos[:, None, :, :]
    s = sin[:, None, :, :]

    rot_f = rot.astype(mx.float32)
    out_rot = (rot_f * c + rotate_half(rot_f) * s).astype(x.dtype)
    return mx.concatenate([out_rot, tail], axis=-1)


def main() -> None:
    np.random.seed(42)

    inv_freq = build_inv_freq()
    pos = build_position_ids()
    cos, sin = reference_cos_sin(pos, inv_freq)

    # Random Q, K (bf16 to match Qwen3.5)
    q_np = np.random.randn(B, HQ, S, HEAD_DIM).astype(np.float32)
    k_np = np.random.randn(B, HKV, S, HEAD_DIM).astype(np.float32)
    q = mx.array(q_np).astype(mx.bfloat16)
    k = mx.array(k_np).astype(mx.bfloat16)

    q_rot = reference_apply(q, cos, sin)
    k_rot = reference_apply(k, cos, sin)

    # Force eval, then convert via mx.save (which writes .npy with MLX dtype encoding).
    mx.eval(cos, sin, q_rot, k_rot)

    def save(name: str, arr) -> None:
        path = OUT_DIR / f"{name}.npy"
        mx.save(str(path), arr)
        print(f"  wrote {path.name}: shape={arr.shape} dtype={arr.dtype}")

    save("input_q", q)
    save("input_k", k)
    save("input_position_ids", pos)
    save("input_inv_freq", inv_freq)
    save("expected_cos", cos)
    save("expected_sin", sin)
    save("expected_q_rot", q_rot)
    save("expected_k_rot", k_rot)

    # ---- expected_attn_out: rotary -> SDPA path against fixture inputs ----
    #
    # The Rust e2e test runs the same path: load q/k/v + cos/sin from these
    # fixtures, run Mrope::apply, then mlx::fast::scaled_dot_product_attention,
    # and compare against expected_attn_out. We do NOT include o_proj here
    # (would require Qwen3.5 weights) — the Rust test mirrors this scope.

    np.random.seed(43)
    v_np = np.random.randn(B, HKV, S, HEAD_DIM).astype(np.float32)
    v = mx.array(v_np).astype(mx.bfloat16)
    save("input_v", v)

    scale = 1.0 / float(np.sqrt(HEAD_DIM))
    attn_out = mx.fast.scaled_dot_product_attention(
        q_rot, k_rot, v, scale=scale, mask="causal"
    )
    mx.eval(attn_out)
    save("expected_attn_out", attn_out)


if __name__ == "__main__":
    main()
