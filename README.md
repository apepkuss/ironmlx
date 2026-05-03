# cxx-mlx

Rust bindings to [Apple MLX](https://github.com/ml-explore/mlx) via the [cxx](https://cxx.rs) crate.

**Status:** P1b2a — full op surface for inference primitives: 6 shape ops (`reshape` with `-1` inference, `transpose`/`transpose_axes`, `broadcast_to`, `concatenate`, `stack`, `split_n`/`split_at`) + 5 reductions (`sum`/`mean`/`max`/`min`/`argmax` via `IntoAxes` sealed trait + `All` marker) + `matmul`. Compose `softmax`/`gelu`/`silu`. Built on P1b1 operators. Full design in [`docs/superpowers/specs/`](docs/superpowers/specs/).

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

## Operators

`mlx::Array` supports the standard arithmetic operators with all 4 reference combinations (`a + b`, `&a + b`, `a + &b`, `&a + &b`) and scalar RHS for any `Element` type. Operators return `Result<Array>` because broadcasting validation, dtype mismatch, or MLX-side errors all surface as recoverable Rust errors:

```rust
use mlx::{Array, Dtype};

fn main() -> mlx::Result<()> {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3])?;
    let b = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3])?;

    // Binary ops with all reference combos
    let _r1 = (&a + &b)?;          // most common
    let _r2 = (&a * 2.0_f32)?;     // scalar RHS

    // Chained unary (free fn or method form)
    let _y = (&a.exp()? - 1.0_f32)?;
    let _z = mlx::ops::sigmoid(&a)?;

    // Negation
    let _n = (-&a)?;
    Ok(())
}
```

NumPy-style broadcasting is validated in Rust before the FFI call; incompatible shapes return `Err(Error::BroadcastMismatch { lhs, rhs })` with structured fields rather than an opaque MLX exception string.

**No scalar LHS** (`1.0 - &a`): blocked by Rust's orphan rule. Equivalent expressions: `(-&a)? + 1.0_f32`, or `Array::from_slice(&[1.0_f32], &[])? - a`.

> **Tip:** Always type-suffix scalar literals (`1.0_f32`, not bare `1.0`). Without a suffix, Rust infers `f64`, and most arrays in inference workloads are `f32`/`f16`/`bf16`. Mixing `f64` scalars into a non-`f64` op surfaces as `Err(Error::Mlx("..."))` at runtime, not at compile time. The same applies to integer literals: prefer `1_i32` over `1`.

Available unary ops: `exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `square`, `rsqrt`, `erf`, `reciprocal` — sufficient to compose `softmax`, `gelu` (via `0.5 * x * (1 + erf(x / sqrt(2)))`), and `silu` once P1b2 adds the needed reductions.

## Reductions, Shape, Matmul

Reductions accept axes via the `IntoAxes` trait — pass `mlx::All` to reduce all axes, an `i32` for a single axis, or any of `&[i32]` / `Vec<i32>` / `[i32; N]` for multiple axes:

```rust
use mlx::{Array, All, ops};

fn main() -> mlx::Result<()> {
    let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let _total = ops::sum(&x, All, false)?;          // scalar 21.0
    let _row_sums = x.sum(-1, false)?;               // [6.0, 15.0]
    let _row_sums_kd = x.sum(-1, true)?;             // [[6.0], [15.0]]

    let _reshaped = x.reshape(&[3, 2])?;             // [3, 2]
    let _auto = x.reshape(&[2, -1])?;                // -1 inferred → [2, 3]
    let _t = x.t()?;                                 // [3, 2] (transpose)

    // Matmul covers 2D, batched, and broadcasting on batch dims.
    let q = Array::from_slice(&[0.0_f32; 24], &[2, 3, 4])?;
    let k = Array::from_slice(&[0.0_f32; 24], &[2, 4, 3])?;
    let _scores = q.matmul(&k)?;                     // [2, 3, 3]

    Ok(())
}
```

`softmax`, `gelu`, and `silu` compose directly atop these ops — see [`mlx/tests/p1b2a_compose.rs`](mlx/tests/p1b2a_compose.rs) for the canonical implementations.

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
- ✅ **P1b1** — operators + element-wise unary + broadcasting
- ✅ **P1b2a** — shape ops + reduction + matmul (compose softmax/gelu/silu)
- ⏳ **P1b2b** — indexing (take/gather/where/slice) + SDPA integration test
- ⏳ **P1c** — random (key + uniform/normal/categorical)
- ⏳ **P2** — `fast` (rms_norm, layer_norm, rope, sdpa) + io (safetensors/gguf load) + transforms
- ⏳ **P3** — quantization + compile + LLM inference example

## Architecture

Two-crate workspace following the standard Rust FFI convention:

- [`mlx-sys`](mlx-sys/) — raw cxx FFI bindings + hand-written C++ shim that flattens MLX templates and overloads into cxx-friendly free functions
- [`mlx`](mlx/) — safe, idiomatic Rust API on top of `mlx-sys`

The `mlx-sys` `build.rs` locates a prebuilt MLX install via `$MLX_DIR` (it never writes to that prefix, so it's safe to share with sibling Rust/Python/Swift projects). A `bundled` feature for source-built MLX is planned for P2.
