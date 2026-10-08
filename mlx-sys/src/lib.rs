//! Raw FFI bindings to MLX C++.
//!
//! This crate is the `-sys` half of the IronMLX MLX bindings. For a safe, idiomatic API,
//! depend on the `mlx` crate instead.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("mlx-sys only supports macOS on Apple Silicon (aarch64-apple-darwin)");

mod bridge;

pub use bridge::array;
pub use bridge::compile;
pub use bridge::conv;
pub use bridge::einsum;
pub use bridge::fast;
pub use bridge::fft;
pub use bridge::io;
pub use bridge::memory;
pub use bridge::metal;
pub use bridge::quantization;
pub use bridge::random;
pub use bridge::stream;
pub use bridge::transforms;

/// Precompiled prefill kernel libraries for Apple GPU generation 17+, built
/// from `shaders/` by `build.rs`.
pub mod shaders {
    /// QMM M-tile prefill library (kernel `ironmlx_qmm_bm128_aligned`).
    pub const PREFILL_QMM_MTILE_METALLIB: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/prefill_qmm_mtile.metallib"));
    /// D256 NAX prefill attention library (kernel `ironmlx_probe_dsplit_bf16`).
    pub const PREFILL_D256_NAX_METALLIB: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/prefill_d256_nax.metallib"));
}
