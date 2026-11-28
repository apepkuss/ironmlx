//! Inkling 2 Mini: native looped MoE text and image inference.
pub mod config;
pub mod image_processor;
mod layers;
mod metal;
mod model;
mod nvfp4;
mod ops;
pub(crate) mod oq;
mod vision;
pub use config::Inkling2Config;
pub use model::Inkling2Model;

/// B1 prefill defaults measured on Apple GPUs. Explicit chunk sizes and
/// calibrated scheduler profiles take precedence at the call sites.
pub fn b1_prefill_chunk_size(oq: bool) -> Option<usize> {
    if mlx::default_device().device_type != mlx::DeviceType::Gpu {
        return None;
    }
    if oq {
        Some(2048)
    } else {
        layers::m5_available().then_some(512)
    }
}

pub fn is_oq_checkpoint(config: &serde_json::Value) -> bool {
    matches!(
        config["model_type"].as_str(),
        Some("inkling2" | "inkling2_loop")
    ) && config["quantization"]["model.embed"].is_object()
}

/// The converted checkpoint retains per-expert NVFP4/MXFP8 metadata in its
/// tensors, independently of the affine quantization of ordinary projections.
pub(crate) fn validate_experts(
    weights: &crate::core::weights::WeightMap,
    config: &serde_json::Value,
) -> crate::Result<std::collections::HashSet<String>> {
    let config = Inkling2Config::from_value(config)?;
    let c = &config.text_config;
    let mut prefixes = std::collections::HashSet::new();
    for i in 0..c.num_hidden_layers {
        for (name, input, output) in [
            ("gate_up_proj", c.hidden_size, 2 * c.intermediate_size),
            ("down_proj", c.intermediate_size, c.hidden_size),
        ] {
            let prefix = format!("model.language_model.layers.{i}.mlp.experts.{name}");
            layers::ExpertLinear::load(weights, &prefix, input, output, c.n_routed_experts)?;
            prefixes.insert(prefix);
        }
    }
    Ok(prefixes)
}
