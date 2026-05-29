//! Model architectures.
//!
//! Each architecture lives in its own directory. Sharing between
//! architectures happens via [`crate::nn`], [`crate::core`], and explicitly
//! shared model modules. Qwen3.6 MoE is a named facade over the shared MoE-VL
//! kernel because the checkpoint declares the same HF architecture and tensor
//! graph while adding Qwen3.6-specific config validation and quant metadata.
//!
//! Planned (in implementation order):
//! - **P3-P4** — `qwen3_5` (Dense): hybrid gated-delta + gated full attention,
//!   MRoPE, RMSNormGated, MTP layer, 4-bit quantized weights.
//! - **P5** — `qwen3_5_moe` (MoE variant): adds SparseMoeBlock; otherwise
//!   reuses qwen3_5 attention / norm primitives via local copies (modules
//!   stay independent — no cross-model imports).
//! - **P6** — `qwen3_5_vl` (multimodal): adds vision encoder + cross-modal
//!   token routing.
//! - **Gemma4** — dense text-only language path with per-layer inputs,
//!   sliding/full attention, KV sharing, and tied output projection.

pub mod gemma4;
pub mod qwen3_5;
pub mod qwen3_5_moe;
pub mod qwen3_6_moe;
pub mod vision;

pub use gemma4::{Gemma4Config, Gemma4Model, Gemma4TextConfig};
pub use qwen3_5::{Qwen35Config, Qwen35Model, Qwen35TextModel, RopeParams};
pub use qwen3_5_moe::{
    Qwen35MoeConfig, Qwen35MoeModel, Qwen35MoeTextModel, RopeParams as MoeRopeParams,
};
pub use qwen3_6_moe::{is_qwen36_moe_config, Qwen36MoeConfig, Qwen36MoeModel, Qwen36MoeTextModel};

// pub mod qwen3_5_vl;

#[cfg(test)]
mod tests {
    #[test]
    fn shared_vision_module_exports_vision_tower() {
        fn assert_type<T>() {}
        assert_type::<super::vision::VisionTower>();
    }
}
