//! Model architectures.
//!
//! Each execution architecture lives in its own directory. Sharing between
//! architectures happens via [`crate::nn`], [`crate::core`], and explicitly
//! shared model modules. Modules for checkpoint-specific validation facades
//! may live next to architecture modules when they preserve a public core API
//! without changing the execution graph.
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
//! - **Qwen3.6 MoE facade** — `qwen3_6_moe` validates Qwen3.6 MoE checkpoint
//!   structure and exposes Qwen3.6-specific regression hooks while delegating
//!   numeric execution to the `qwen3_5_moe` architecture.
//! - **MiniCPM-V-4.6 facade** — `minicpmv4_6` adapts the MiniCPM-V-4.6
//!   `text_config` (Qwen3.5-text backbone) onto the `qwen3_5` dense execution
//!   graph for text-only inference. The SigLIP vision tower is not yet
//!   implemented; image inputs are out of scope.

pub mod architecture;
pub mod dflash2;
pub mod diffusion_gemma;
pub mod gemma4;
pub mod glm4_moe_lite;
pub mod llama;
pub mod minicpmv4_6;
pub mod qwen3_5;
pub mod qwen3_5_moe;
pub mod qwen3_6_moe;
pub mod vision;

pub use architecture::ModelArchitecture;
pub use dflash2::{DFlash2Config, DFlash2DraftModel};
pub use diffusion_gemma::{
    DiffusionGemmaConfig, DiffusionGemmaGenerateEvent, DiffusionGemmaGenerationConfig,
    DiffusionGemmaModel, DiffusionGemmaTextConfig,
};
pub use gemma4::{Gemma4AssistantConfig, Gemma4Config, Gemma4Model, Gemma4TextConfig};
pub use glm4_moe_lite::{Glm4MoeLiteConfig, Glm4MoeLiteModel};
pub use llama::{LlamaConfig, LlamaModel};
pub use minicpmv4_6::MiniCpmV46Model;
pub use qwen3_5::{Qwen35Config, Qwen35Model, Qwen35TextModel, RopeParams};
pub use qwen3_5_moe::{
    Qwen35MoeConfig, Qwen35MoeModel, Qwen35MoeMtp, Qwen35MoeMtpConfig, Qwen35MoeTextModel,
    RopeParams as MoeRopeParams,
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
