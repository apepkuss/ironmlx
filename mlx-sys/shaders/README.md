# Prefill kernels for Apple GPU generation 17+

Compiled by `../build.rs` into `prefill_qmm_mtile.metallib` and
`prefill_d256_nax.metallib` and embedded as `mlx_sys::shaders`:

    xcrun metal -std=metal4.0 -O2 -fno-fast-math -mmacosx-version-min=26.2 -I vendor/<mlx>/include -c <source> -o <name>.air
    xcrun metallib <name>.air -o <name>.metallib

| library | source | MLX kernel headers | sha256 with Apple metal 32023.883 |
| --- | --- | --- | --- |
| QMM M-tile (`ironmlx_qmm_bm128_aligned`) | `qmm_mtile_probe.metal` | `vendor/mlx-0.32.2` | 961833aaa3cc02ab8c41d25aad79cc5ab0c6bbe12da2851f5bc21511ede2b633 |
| D256 NAX attention (prefill `ironmlx_probe_dsplit_bf16`, split-K decode/verify `ironmlx_splitk_d256_pass1_wm{1,2,4}` + `ironmlx_splitk_d256_pass2`) | `nax_dsplit_probe.metal` | `vendor/mlx-0.32.3` | 6230c1e4e1742e57301611ba2ef9aea973986e5ab18b52d18e22905c7c904688 |

The source file names are part of the compiled output (static-initializer
symbol), so they are kept as qualified. `vendor/` holds only the MLX kernel
headers these sources include (MIT, see each `LICENSE`):

- `vendor/mlx-0.32.2`: from the pinned MLX fork commit
  73ad5df20cb30be4192e5c4d0ae8130674773427 (version 0.32.2), the MLX this
  project links (`scripts/release-config.sh`).
- `vendor/mlx-0.32.3`: from upstream MLX tag v0.32.3 (commit
  64ea011cb65f14d9ce2737e60db9a4ae91ed7441), which adds
  `attention_nax_dsplit`; only these headers come from it, the linked MLX
  library stays the pinned fork.

Both directories are listed in `compliance/native-dependencies.json`;
`scripts/verify-third-party-materials.sh` checks every file against the
listed MLX commit.
