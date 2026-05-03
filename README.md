# cxx-mlx

Rust bindings to [Apple MLX](https://github.com/ml-explore/mlx) via the [cxx](https://cxx.rs) crate.

**Status:** P1a — Array foundation (`zeros`/`from_slice<T>`/`item<T>`/`to_vec<T>`/`Clone`/`Debug`/`Send`/`SmallVec` shape) covering 10 dtypes incl. `f16`/`bf16`. Full design in [`docs/superpowers/specs/`](docs/superpowers/specs/).

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
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2])?;
    println!("{a:?}");
    let v: Vec<f32> = a.to_vec()?;
    println!("values: {v:?}");
    Ok(())
}
```

## Threading

`mlx::Array` implements `Send` but **not** `Sync`. Internally, MLX's
`mlx::core::array` is backed by a `std::shared_ptr` whose refcount is atomic,
so transferring ownership across threads is safe. However, MLX's "const"
methods (e.g. `set_status`, `attach_event`, the lazy→available transition
in `is_available`) mutate `ArrayDesc` without synchronization, so two
threads concurrently holding `&Array` to the same instance is a data race.

To share an array between threads, clone it (cheap — MLX refcounts the
underlying storage):

```rust
let a = mlx::Array::zeros(&[2, 3], mlx::Dtype::Float32)?;
let b = a.clone();   // Atomic refcount++; tensor data is not copied.
std::thread::spawn(move || {
    let _ = b.shape();
});
# Ok::<(), mlx::Error>(())
```

Avoid wrapping in `Arc<Mutex<Array>>` unless you genuinely need shared
mutable access — `clone` is almost always the right answer.

## Roadmap

- ✅ **P0** — scaffold (zeros + eval + shape)
- ✅ **P1a** — Array foundation (Element trait, 10 dtypes, from_slice/item/to_vec, Clone/Debug, Send, SmallVec shape)
- ⏳ **P1b** — core ops (arithmetic, reduction, indexing, shape ops)
- ⏳ **P1c** — random (key + uniform/normal/categorical)
- ⏳ **P2** — `fast` (rms_norm, layer_norm, rope, sdpa) + io (safetensors/gguf load) + transforms
- ⏳ **P3** — quantization + compile + LLM inference example

## Architecture

Two-crate workspace following the standard Rust FFI convention:

- [`mlx-sys`](mlx-sys/) — raw cxx FFI bindings + hand-written C++ shim that flattens MLX templates and overloads into cxx-friendly free functions
- [`mlx`](mlx/) — safe, idiomatic Rust API on top of `mlx-sys`

The `mlx-sys` `build.rs` locates a prebuilt MLX install via `$MLX_DIR` (it never writes to that prefix, so it's safe to share with sibling Rust/Python/Swift projects). A `bundled` feature for source-built MLX is planned for P2.
