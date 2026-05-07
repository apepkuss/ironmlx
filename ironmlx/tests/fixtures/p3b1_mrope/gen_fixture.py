"""Generate P3b1 MRoPE fixtures from a Python reference implementation.

This is an independent re-implementation of the MRoPE algorithm using
mlx.core primitives -- it does NOT use mlx-lm's nn.RoPE (which is a no-op
for the mrope path). The reference matches the algorithm specified in
docs/superpowers/specs/2026-05-07-ironmlx-p3b1-mrope-finish-design.md.

Outputs go alongside this script as `.npy` files. Re-run to regenerate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

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
    """Reference MRoPE cos/sin -- independent re-implementation per the spec."""
    pos_f = position_ids.astype(mx.float32)
    pos_unsq = pos_f[..., None]                # [3, B, S, 1]
    inv_unsq = inv_freq.reshape((1, 1, 1, -1)) # [1, 1, 1, half]
    freqs = pos_unsq * inv_unsq                 # [3, B, S, half]
    cos_per = mx.cos(freqs)
    sin_per = mx.sin(freqs)

    # 3-section concat along last axis
    offsets = [0]
    for n in SECTIONS:
        offsets.append(offsets[-1] + n)

    cos_segs = []
    sin_segs = []
    for s, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        cos_segs.append(cos_per[s, :, :, lo:hi])
        sin_segs.append(sin_per[s, :, :, lo:hi])
    cos = mx.concatenate(cos_segs, axis=-1)
    sin = mx.concatenate(sin_segs, axis=-1)
    return cos, sin


def reference_apply(
    x: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    """Apply interleaved rotation to `x` (Q or K), tail pass-through."""
    # x: [B, H, S, HEAD_DIM]
    rot = x[..., :ROT_DIM]
    tail = x[..., ROT_DIM:]

    # Interleaved: even (2p) and odd (2p+1) channels form pairs sharing cos[p], sin[p].
    even = rot[..., 0::2]  # [B, H, S, HALF]
    odd = rot[..., 1::2]

    # Broadcast cos/sin: [B, S, HALF] -> [B, 1, S, HALF]
    c = cos[:, None, :, :]
    s = sin[:, None, :, :]

    rot_even = (even.astype(mx.float32) * c - odd.astype(mx.float32) * s).astype(x.dtype)
    rot_odd = (even.astype(mx.float32) * s + odd.astype(mx.float32) * c).astype(x.dtype)

    # Re-interleave
    out_rot = mx.stack([rot_even, rot_odd], axis=-1).reshape(x.shape[:-1] + (ROT_DIM,))
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


if __name__ == "__main__":
    main()
