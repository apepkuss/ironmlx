# Prefill kernels for Apple GPU generation 17+

Compiled by `../build.rs` into `prefill_qmm_mtile.metallib` and
`prefill_d256_nax.metallib` and embedded as `mlx_sys::shaders`:

    xcrun metal -std=metal4.0 -O2 -fno-fast-math -mmacosx-version-min=26.2 -I vendor/<mlx>/include -c <source> -o <name>.air
    xcrun metallib <name>.air -o <name>.metallib

| library | source | MLX kernel headers | sha256 with Apple metal 32023.883 |
| --- | --- | --- | --- |
| QMM M-tile (`ironmlx_qmm_bm128_aligned`) | `qmm_mtile_probe.metal` | `vendor/mlx-0.32.2` | 961833aaa3cc02ab8c41d25aad79cc5ab0c6bbe12da2851f5bc21511ede2b633 |
| D256 NAX attention (`ironmlx_probe_dsplit_bf16`) | `nax_dsplit_probe.metal` | `vendor/mlx-0.32.3` | 7d9281498c7fcc4047e47fc347c4ee64f8af62786476d0ebb4290fb1db9567c9 |

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
