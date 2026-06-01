//! Standard Llama-family GQA dense model (`model_type = "llama"`).
//!
//! Supports MiniCPM5-1B and other `LlamaForCausalLM` checkpoints: plain GQA
//! attention + standard RoPE + SwiGLU MLP + RMSNorm, separate `lm_head`. No
//! MoE / MLA / sliding-window / Q-K norm. Text-only.

mod attention;
pub mod config;
mod decoder_layer;
mod model;

pub use config::LlamaConfig;
pub use model::LlamaModel;
