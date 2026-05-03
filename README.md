# cxx-mlx

Rust bindings to [Apple MLX](https://github.com/ml-explore/mlx) via the [cxx](https://cxx.rs) crate.

**Status:** P0 scaffold (zeros + eval + shape only). See `docs/superpowers/specs/` for the full design.

## Requirements

- macOS, Apple Silicon
- Rust 1.94+
- Prebuilt MLX 0.32+ at `$MLX_DIR` (see `docs/superpowers/plans/2026-05-03-cxx-mlx-p0-scaffold.md` for build instructions)

## Quickstart

```rust
use mlx::{Array, Dtype};

let a = Array::zeros(&[2, 3], Dtype::Float32);
assert_eq!(a.shape(), vec![2, 3]);
a.eval().unwrap();
```
