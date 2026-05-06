# cxx-mlx

Rust bindings to [Apple MLX](https://github.com/ml-explore/mlx) via the [cxx](https://cxx.rs) crate.

**Status:** 🎉 **P6 complete** — `mlx::compile` 闭包 JIT 绑定 (`compile()` + `CompiledFn::invoke` + global controls). 用户可把任意 Rust 闭包传给 MLX 进行图追踪 + 融合.

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

> **Gotcha:** `.t()` reverses **all** dims. For 4D attention (`Q @ K^T` where `Q`/`K` are `[B, H, S, D]`), use `k.transpose_axes(&[0, 1, 3, 2])` to swap just the last two dims, not `k.t()` (which would yield `[D, S, H, B]`). See [`matmul_using_t_for_attention`](mlx/tests/p1b2a_matmul.rs).

## Indexing & SDPA

`mlx::ops::where_` (trailing underscore, since `where` is a Rust keyword) selects element-wise from two arrays based on a condition mask. `take` / `take_along_axis` index along an axis (NumPy / PyTorch semantics). `slice` and `slice_strided` extract sub-arrays Python-style:

```rust
use mlx::{ops, Array};

fn main() -> mlx::Result<()> {
    let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let cond = Array::from_slice(&[1_u8, 0, 1, 0, 1, 0], &[2, 3])?;
    let zeros = Array::from_slice(&[0.0_f32; 6], &[2, 3])?;

    let _picked = ops::where_(&cond, &a, &zeros)?;             // element-wise select
    let idx = Array::from_slice(&[0_u32, 2], &[2])?;
    let _cols = a.take(&idx, 1)?;                              // [2, 2] — pick cols 0 and 2
    let _sub = a.slice(&[0, 1], &[2, 3])?;                     // [2, 2] — rows 0..2, cols 1..3
    Ok(())
}
```

A complete SDPA (scaled dot-product attention) implementation composing matmul, transpose, mask-add, softmax, and matmul lives in [`mlx/tests/p1b2b_sdpa.rs`](mlx/tests/p1b2b_sdpa.rs). It's the canonical test that all of P1 (P0 + P1a + P1b1 + P1b2a + P1b2b) integrates correctly. P2b's [`fast::scaled_dot_product_attention`](mlx/tests/p2b_fast.rs) matches these numerics via a fused Metal kernel.

## Streams & Devices

`Device::cpu()` / `Device::gpu(index)` are the supported devices on Apple
Silicon. Streams are MLX's execution queues — work on different streams may
run concurrently. The default stream of the default device is used unless
explicitly overridden:

```rust
use mlx::{Array, Device, Dtype};

fn main() -> mlx::Result<()> {
    println!("default device: {:?}", mlx::default_device());
    println!("gpu count: {}", mlx::device_count(mlx::DeviceType::Gpu));

    let _arr = Array::zeros(&[2, 3], Dtype::Float32)?;

    // Optional: switch streams (thread-local).
    let s = mlx::new_stream(Device::gpu(0))?;
    mlx::set_default_stream(s);

    Ok(())
}
```

### Async evaluation

`async_eval` returns a runtime-agnostic `Future`. It works under any
executor — tokio, async-std, smol, or `futures_lite::future::block_on`:

```rust
use mlx::{Array, Dtype};

# #[tokio::main]
# async fn main() -> mlx::Result<()> {
let a = Array::zeros(&[1024], Dtype::Float32)?;
let b = Array::zeros(&[1024], Dtype::Float32)?;

// Submit one or many arrays; await when ready.
mlx::async_eval(&[&a, &b]).await?;

// Or single-array convenience method:
let c = Array::zeros(&[256], Dtype::Float32)?;
c.async_eval().await?;
# Ok(())
# }
```

**Cancellation note**: dropping a future without awaiting does NOT cancel
the submitted MLX work — MLX has no cancellation primitive. The work runs
to completion in the background. Subsequent ops on the same arrays will
implicitly synchronize.

For sync contexts (no executor), use `mlx::synchronize()` (default stream)
or `mlx::synchronize_stream(s)` (explicit stream) to block.

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
- ✅ **P1b2b** — indexing (take/take_along_axis/where/slice/gather) + u16/u32/u64 dtypes + SDPA integration
- 🎉 **P1 complete** — full inference primitives ready
- ✅ **P2a** — Stream / Device foundation + runtime-agnostic async_eval
- ✅ **P2b** — `fast` ops (rms_norm / layer_norm / rope int+array offset / sdpa) — 12 integration tests
- ✅ **P2c** — `io` (safetensors / gguf / npy + Reader/Writer streams) — 18 integration tests
- ✅ **P3** — `quantization` (quantize/dequantize/quantized_matmul/qqmm/gather_qmm/fp8) — 8 integration tests
- ✅ **P4** — `random` (key/seed/split + 17 distributions including categorical) — 23 integration tests
- ✅ **P5** — `ops` 补漏 (8 matmul family ops) — 9 integration tests
- ✅ **P6** — `compile` (closure JIT via extern "Rust" callback + ArrayVec opaque + CompiledFn) — 9 integration tests
- ⏳ LLM inference example

## Architecture

Two-crate workspace following the standard Rust FFI convention:

- [`mlx-sys`](mlx-sys/) — raw cxx FFI bindings + hand-written C++ shim that flattens MLX templates and overloads into cxx-friendly free functions
- [`mlx`](mlx/) — safe, idiomatic Rust API on top of `mlx-sys`

The `mlx-sys` `build.rs` locates a prebuilt MLX install via `$MLX_DIR` (it never writes to that prefix, so it's safe to share with sibling Rust/Python/Swift projects). A `bundled` feature for source-built MLX is planned for P2.
