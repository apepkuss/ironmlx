# P3b3 GatedDeltaNet fixtures

Reference data for `nn::GatedDeltaNet` numerical-correctness tests.

The reference is an **independent re-implementation** of the gated delta
algorithm using `mlx.core` primitives (the `gated_delta_ops`-style sequential
loop, NOT mlx-lm's Metal kernel). This avoids circular validation if our
Metal kernel and mlx-lm's Metal kernel share the same bug.

Small-scale synthetic config (Dk=32 minimum due to Metal kernel constraint
`n_per_t = Dk/32 >= 1`):
- B=1, S=4, num_v_heads=4, num_k_heads=2,
- head_k_dim=32, head_v_dim=32, hidden=128, conv_kernel=4, eps=1e-6.

## Regenerate

Requires the `mlx` Python version pinned in `gen_fixture.py`. Re-run after
any algorithmic change to the reference.

```bash
cd ironmlx/tests/fixtures/p3b3_gated_delta_net
python gen_fixture.py
```

Generated `.npy` files (committed to git, ~25-30 KB total):

| File | Shape | Dtype |
|---|---|---|
| `input_x.npy` | `[1, 4, 128]` | bf16 |
| `qkv_proj_weight.npy` | `[256, 128]` | bf16 |
| `z_proj_weight.npy` | `[128, 128]` | bf16 |
| `a_proj_weight.npy` | `[4, 128]` | bf16 |
| `b_proj_weight.npy` | `[4, 128]` | bf16 |
| `conv1d_weight.npy` | `[256, 4, 1]` | bf16 |
| `norm_weight.npy` | `[32]` | fp32 |
| `out_proj_weight.npy` | `[128, 128]` | bf16 |
| `A_log.npy` | `[4]` | fp32 |
| `dt_bias.npy` | `[4]` | fp32 |
| `expected_output.npy` | `[1, 4, 128]` | (varies; see Note) |

> **Note on output dtype**: `mx.fast.rms_norm(bf16, fp32_weight)` promotes to
> fp32, which propagates through the rest of the chain. The integration test
> asserts the dtype matches whatever the fixture produces (likely fp32).
