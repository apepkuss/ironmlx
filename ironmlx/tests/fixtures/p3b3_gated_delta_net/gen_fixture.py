"""Generate P3b3 GatedDeltaNet fixtures.

Independent re-implementation using `mlx.core` primitives, NOT mlx-lm's
Metal kernel (avoids circular validation).
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

# ---- Small synthetic config (Dk=32 minimum due to Metal kernel constraint) ----
B, S = 1, 4
HV = 4
HK = 2
DK = 32
DV = 32
HIDDEN = HV * DV  # 128
KEY_DIM = HK * DK  # 64
VALUE_DIM = HV * DV  # 128
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 64+64+128 = 256
CONV_KERNEL = 4
EPS = 1e-6


def _ref_gated_delta_step_ops(q, k, v, g, beta, state):
    """Sequential ops-based step (per mlx-lm gated_delta._gated_delta_step_ops)."""
    decay = g[..., None, None]
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)
    return y.astype(q.dtype), state


def _ref_softplus(x):
    return mx.where(x > 20, x, mx.logaddexp(mx.zeros_like(x), x))


def _ref_compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * _ref_softplus(a + dt_bias))


def _ref_silu(x):
    return x * mx.sigmoid(x)


def _ref_gated_delta_net(
    x, qkv_w, z_w, a_w, b_w, conv_w, norm_w, out_w, A_log, dt_bias
):
    """Independent ref impl of GatedDeltaNet (no cache, no mask)."""
    qkv = x @ qkv_w.T
    z = x @ z_w.T
    a = x @ a_w.T
    b = x @ b_w.T

    conv_state = mx.zeros((B, CONV_KERNEL - 1, CONV_DIM), dtype=qkv.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = mx.conv1d(conv_input, conv_w, stride=1, padding=0, groups=CONV_DIM)
    conv_out = _ref_silu(conv_out)

    q_flat = conv_out[..., :KEY_DIM]
    k_flat = conv_out[..., KEY_DIM:2*KEY_DIM]
    v_flat = conv_out[..., 2*KEY_DIM:]

    q = q_flat.reshape(B, S, HK, DK)
    k = k_flat.reshape(B, S, HK, DK)
    v = v_flat.reshape(B, S, HV, DV)

    inv_scale = DK ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, EPS)
    k = inv_scale * mx.fast.rms_norm(k, None, EPS)

    g = _ref_compute_g(A_log, a, dt_bias)  # [B, S, HV]
    beta = mx.sigmoid(b)                   # [B, S, HV]

    # GQA repeat
    repeat = HV // HK
    q_rep = mx.repeat(q, repeat, axis=-2)
    k_rep = mx.repeat(k, repeat, axis=-2)

    state = mx.zeros((B, HV, DV, DK), dtype=mx.float32)
    ys = []
    for t in range(S):
        y_t, state = _ref_gated_delta_step_ops(
            q_rep[:, t], k_rep[:, t], v[:, t], g[:, t], beta[:, t], state,
        )
        ys.append(y_t)
    y = mx.stack(ys, axis=1)  # [B, S, HV, DV]

    z_per_head = z.reshape(B, S, HV, DV)
    y_normed = mx.fast.rms_norm(y, norm_w, EPS)
    z_silu = _ref_silu(z_per_head.astype(mx.float32))
    y_normed_f32 = y_normed.astype(mx.float32)
    out_per_head = (z_silu * y_normed_f32).astype(y.dtype)

    out_flat = out_per_head.reshape(B, S, HIDDEN)
    out = out_flat @ out_w.T
    return out


def main():
    np.random.seed(46)

    def randn(shape, dtype=mx.bfloat16, scale=0.1):
        a = np.random.randn(*shape).astype(np.float32) * scale
        return mx.array(a).astype(dtype)

    x = randn((B, S, HIDDEN))
    qkv_w = randn((CONV_DIM, HIDDEN))
    z_w = randn((VALUE_DIM, HIDDEN))
    a_w = randn((HV, HIDDEN))
    b_w = randn((HV, HIDDEN))
    conv_w = randn((CONV_DIM, CONV_KERNEL, 1))
    norm_w = randn((DV,), dtype=mx.float32, scale=0.5) + mx.array([1.0])
    out_w = randn((HIDDEN, VALUE_DIM))
    A_log = randn((HV,), dtype=mx.float32, scale=1.0)
    dt_bias = randn((HV,), dtype=mx.float32, scale=0.5) + mx.array([1.0])

    out = _ref_gated_delta_net(
        x, qkv_w, z_w, a_w, b_w, conv_w, norm_w, out_w, A_log, dt_bias,
    )
    mx.eval(x, qkv_w, z_w, a_w, b_w, conv_w, norm_w, out_w, A_log, dt_bias, out)

    def save(name, arr):
        path = OUT_DIR / f"{name}.npy"
        mx.save(str(path), arr)
        print(f"  wrote {path.name}: shape={arr.shape} dtype={arr.dtype}")

    save("input_x", x)
    save("qkv_proj_weight", qkv_w)
    save("z_proj_weight", z_w)
    save("a_proj_weight", a_w)
    save("b_proj_weight", b_w)
    save("conv1d_weight", conv_w)
    save("norm_weight", norm_w)
    save("out_proj_weight", out_w)
    save("A_log", A_log)
    save("dt_bias", dt_bias)
    save("expected_output", out)


if __name__ == "__main__":
    main()
