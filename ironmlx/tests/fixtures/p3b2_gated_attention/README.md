# P3b2 Gated Full Attention fixtures

Reference data for `nn::GatedAttention` numerical-correctness tests.

The reference is an independent re-implementation of the Qwen3-Next gated
full attention algorithm using `mlx.core` primitives. It mirrors mlx-lm's
`Qwen3NextAttention` algorithm but does NOT call mlx-lm directly — this
keeps the reference free of any patching / monkey-patching surprises.

Small-scale synthetic config: B=1, S=4, Hq=4, Hkv=2, D=8, hidden=Hq*D=32,
partial_rotary_factor=1.0, sections=[2,1,1] (sum to ROT_PAIRS=4).

## Regenerate

Requires the same `mlx` Python version pinned in `gen_fixture.py`. Re-run
after any algorithmic change to the reference.

```bash
cd ironmlx/tests/fixtures/p3b2_gated_attention
python gen_fixture.py
```

Generated `.npy` files (committed to git, ~10 KB total):

| File | Shape | Dtype |
|---|---|---|
| `input_x.npy` | `[1, 4, 32]` | bf16 |
| `input_position_ids.npy` | `[3, 1, 4]` | i32 |
| `input_inv_freq.npy` | `[4]` | fp32 |
| `q_proj_weight.npy` | `[64, 32]` | bf16 |
| `k_proj_weight.npy` | `[16, 32]` | bf16 |
| `v_proj_weight.npy` | `[16, 32]` | bf16 |
| `o_proj_weight.npy` | `[32, 32]` | bf16 |
| `q_norm_weight.npy` | `[8]` | fp32 |
| `k_norm_weight.npy` | `[8]` | fp32 |
| `expected_cos.npy` | `[1, 4, 8]` | fp32 |
| `expected_sin.npy` | `[1, 4, 8]` | fp32 |
| `expected_gated_attn_out.npy` | `[1, 4, 32]` | fp32 |

> **Note on output dtype**: `expected_gated_attn_out.npy` is fp32 even though
> `input_x.npy` is bf16 because `mx.fast.rms_norm(bf16_input, fp32_weight)`
> promotes the output to fp32 (Metal kernel type-promotion rule). The
> q_norm/k_norm weights are fp32 here, so the SDPA→sigmoid→o_proj chain runs
> in fp32 from that point on. The Rust `GatedAttention::forward` follows the
> same MLX backend, so dtype matches.
