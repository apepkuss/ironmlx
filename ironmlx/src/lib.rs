//! ironmlx — local LLM inference on Apple Silicon, backed by [`mlx`] (cxx-mlx).
//!
//! ## Layered structure
//!
//! - [`nn`] — neural-network primitives shared across model families
//!   (Linear, Embedding, RMSNorm, MRoPE, attention helpers, MLP).
//! - [`core`] — generation infrastructure that's model-agnostic
//!   (KV cache with prefix sharing, sampler, tokenizer, weight loading).
//! - [`models`] — one self-contained directory per model architecture.
//!   Each model owns its config, layer structs, and forward path.
//!   Shared components stay in [`nn`] / [`core`].
//! - [`cli`] — command-line interface (top-level binary in `main.rs`).
//!
//! ## Performance principles
//!
//! Every model implementation must:
//! - Use `mlx::fast::scaled_dot_product_attention` for attention hot paths
//!   (fused Metal kernel) — never compose softmax + matmul by hand.
//! - Compile per-layer forward via `mlx::compile::compile(.., ShapeMode::Shapeless)`
//!   so prefill and decode share one optimized graph.
//! - Operate on quantized weights directly via `quantized_matmul` —
//!   never dequantize on the inference hot path.
//! - Pre-allocate KV cache pages and advance a position pointer per token;
//!   never `concatenate` per decode step.
//! - Use `_on(stream)` variants to dispatch independent ops to different
//!   streams when concurrency is genuinely available
//!   (e.g. prefill stream vs decode stream).
//! - Defer `eval()` until the end of a layer / step; let MLX fuse the graph.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("ironmlx only supports macOS on Apple Silicon (aarch64-apple-darwin)");

pub mod cli;
pub mod core;
pub mod models;
pub mod nn;

pub use anyhow::{Error, Result};

pub use core::{ChatTemplate, KVCache, Loader, Message, QuantMeta, Sampler, Tokenizer};
