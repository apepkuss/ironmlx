//! Gemma4 Dense model support.
//!
//! Scope: `model_type=gemma4` and `model_type=gemma4_unified` with
//! `text_config.enable_moe_block=false`. Vision supports image + text prompts.
//! Audio/video and Gemma4 MoE are intentionally out of scope.

mod attention;
mod config;
mod cross_modal;
mod decoder_layer;
pub(crate) mod drafter;
pub mod image_processor;
mod mlp;
mod model;
mod ops;
mod profile;
mod quant_fusion;
mod rope;
mod text_model;
pub(crate) mod vision;

pub use config::{
    Gemma4AssistantConfig, Gemma4Config, Gemma4LayerKind, Gemma4RopeParams, Gemma4TextConfig,
    Gemma4VisionConfig,
};
pub(crate) use drafter::{draft_position_for_shared_kv, shared_kv_row_view_on};
pub use drafter::{
    Gemma4AssistantModel, Gemma4DrafterActiveKvRuntime, Gemma4DrafterGenerationStream,
    Gemma4DrafterPrefixCache, Gemma4DrafterTraceWindow,
};
pub use model::Gemma4Model;
pub use text_model::{Gemma4SharedKvStates, Gemma4TextModel};

#[cfg(test)]
mod tests {
    use mlx::Dtype;

    #[test]
    fn drafter_sliding_mask_is_bidirectional() {
        let mask = super::drafter::build_bidirectional_swa_mask_for_test(
            2,
            4,
            6,
            3,
            None,
            0,
            Dtype::Float32,
        )
        .unwrap()
        .expect("window requires a mask");
        let values: Vec<f32> = mask.to_vec().unwrap();
        let k_len = 6usize;
        let at = |q: usize, k: usize| values[q * k_len + k];

        assert!(at(0, 0).is_infinite() && at(0, 0).is_sign_negative());
        assert_eq!(at(0, 2), 0.0);
        assert_eq!(at(0, 5), 0.0);
        assert_eq!(at(1, 3), 0.0);
        assert_eq!(at(1, 5), 0.0);
    }
}
