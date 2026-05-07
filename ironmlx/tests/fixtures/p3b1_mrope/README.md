# P3b1 MRoPE fixtures

Reference data for Qwen3.5 MRoPE numerical-correctness tests.

The reference implementation is an independent re-implementation of the
MRoPE algorithm (per the project spec) using `mlx.core` primitives. It is
NOT a passthrough of mlx-lm's `nn.RoPE` — mlx-lm's current `mrope` path is
a no-op that ignores `mrope_section` and falls through to standard RoPE.

## Regenerate

Requires `mlx` Python package (matches the version pinned at the top of
`gen_fixture.py`). Re-run after any algorithmic change to the reference.

```bash
cd ironmlx/tests/fixtures/p3b1_mrope
python gen_fixture.py
```

Generated `.npy` files (committed to git, ~800KB total):

| File | Shape | Dtype |
|---|---|---|
| `input_q.npy` | `[1, 64, 8, 256]` | bf16 |
| `input_k.npy` | `[1, 8, 8, 256]` | bf16 |
| `input_position_ids.npy` | `[3, 1, 8]` | i32 |
| `input_inv_freq.npy` | `[32]` | fp32 |
| `expected_cos.npy` | `[1, 8, 32]` | fp32 |
| `expected_sin.npy` | `[1, 8, 32]` | fp32 |
| `expected_q_rot.npy` | `[1, 64, 8, 256]` | bf16 |
| `expected_k_rot.npy` | `[1, 8, 8, 256]` | bf16 |

`expected_attn_out.npy` and `input_v.npy` are added in T4.
