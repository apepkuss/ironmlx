# cxx-mlx

Rust bindings to [Apple MLX](https://github.com/ml-explore/mlx) via the [cxx](https://cxx.rs) crate.

**Status:** P0 scaffold — `Array::zeros` + `shape`/`dtype`/`size`/`ndim` + `eval`. Full design in [`docs/superpowers/specs/`](docs/superpowers/specs/).

## Requirements

- macOS, Apple Silicon
- Rust 1.94+
- Prebuilt MLX 0.32+ at `$MLX_DIR`

## Quickstart

Build MLX once (any prefix you like):

```bash
cd /path/to/mlx
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/mlx
make -j$(sysctl -n hw.ncpu) && make install
export MLX_DIR=$HOME/.local/mlx
```

Then in your project:

```rust
use mlx::{Array, Dtype};

fn main() -> mlx::Result<()> {
    let a = Array::zeros(&[2, 3], Dtype::Float32);
    println!("shape={:?} dtype={:?} size={}", a.shape(), a.dtype(), a.size());
    a.eval()?;
    Ok(())
}
```

## Roadmap

- ✅ **P0** — scaffold (zeros + eval + shape)
- ⏳ **P1** — `Array` + core ops (arithmetic, reduction, indexing, shape ops, random)
- ⏳ **P2** — `fast` (rms_norm, layer_norm, rope, sdpa) + io (safetensors/gguf load) + transforms
- ⏳ **P3** — quantization + compile + LLM inference example

## Architecture

Two-crate workspace following the standard Rust FFI convention:

- [`mlx-sys`](mlx-sys/) — raw cxx FFI bindings + hand-written C++ shim that flattens MLX templates and overloads into cxx-friendly free functions
- [`mlx`](mlx/) — safe, idiomatic Rust API on top of `mlx-sys`

The `mlx-sys` `build.rs` locates a prebuilt MLX install via `$MLX_DIR` (it never writes to that prefix, so it's safe to share with sibling Rust/Python/Swift projects). A `bundled` feature for source-built MLX is planned for P2.
