//! Generation infrastructure that's model-agnostic.
//!
//! Components added incrementally:
//! - **P1** — `loader` (safetensors tree → name→Array map), `tokenizer`
//!   (HuggingFace `tokenizers` integration), `sampler`
//!   (greedy / temperature / top-p / top-k).
//! - **P2** — `cache` (block-style + prefix-aware KV cache, vllm-inspired).
//! - **P3+** — `generate` loop wiring (prefill + decode driver).

// pub mod cache;
// pub mod generate;
// pub mod loader;
// pub mod sampler;
// pub mod tokenizer;
