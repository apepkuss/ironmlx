//! Conservative discovery metadata without loading model weights.

use super::runtime_config::EngineModelConfig;
use ironmlx_lm::models::ModelArchitecture;
use serde_json::Value;

/// Conservative output budget advertised when a causal model has no trusted
/// independent output-only limit and no deployment override.
pub const DEFAULT_ADVERTISED_MAX_OUTPUT_TOKENS: usize = 4_096;

/// Resolve the output budget published by model discovery.
///
/// `context_window` is the combined input/output capacity, not an output-only
/// limit. A configured deployment default therefore wins, otherwise discovery
/// advertises a conservative 4K budget. Both paths leave at least half of a
/// multi-token window available for input so clients cannot mistake the total
/// context capacity for a directly usable per-request output budget.
pub fn advertised_max_output_tokens(
    context_window: Option<usize>,
    configured_default: Option<usize>,
) -> Option<usize> {
    let context_window = context_window?;
    let input_aware_cap = (context_window / 2).max(1);
    Some(
        configured_default
            .unwrap_or(DEFAULT_ADVERTISED_MAX_OUTPUT_TOKENS)
            .min(input_aware_cap),
    )
}

/// Resolve the same total-token ceiling used by causal admission. Missing or
/// invalid metadata is deliberately not replaced with a generation default.
pub(crate) fn configured_context_window(model: &EngineModelConfig) -> Option<usize> {
    if model.audio.is_some() || model.capabilities.runtime_kind != "causal" {
        return None;
    }
    let cache_cap = model
        .scheduler_runtime_profile
        .as_ref()?
        .config
        .max_cache_cap;
    let raw = std::fs::read(model.path.join("config.json")).ok()?;
    let config: Value = serde_json::from_slice(&raw).ok()?;
    context_window_from_config(&config, cache_cap)
}

fn context_window_from_config(config: &Value, cache_cap: usize) -> Option<usize> {
    // Follow each loader's layout; in particular, do not fall back to a VLM's
    // top-level or vision capacity when text_config is absent or malformed.
    let text = match ModelArchitecture::from_config_value(config).ok()? {
        ModelArchitecture::Llama | ModelArchitecture::Glm4MoeLite => config,
        ModelArchitecture::Qwen35Dense
        | ModelArchitecture::Qwen35Moe
        | ModelArchitecture::Gemma4
        | ModelArchitecture::MiniCpmV46 => config.get("text_config")?,
        ModelArchitecture::DiffusionGemma => return None,
    };
    // ModelMeta uses i32, so values outside that range are not usable limits.
    let model_cap = i32::try_from(text.get("max_position_embeddings")?.as_u64()?).ok()?;
    let cap = cache_cap.min(usize::try_from(model_cap).ok()?);
    (cap > 0).then_some(cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn capacity_uses_loader_layout_and_effective_cache_limit() {
        for kind in ["llama", "glm4_moe_lite"] {
            let config = json!({"model_type": kind, "max_position_embeddings": 8192,
                "text_config": {"max_position_embeddings": 64}});
            assert_eq!(context_window_from_config(&config, 32768), Some(8192));
            assert_eq!(context_window_from_config(&config, 4096), Some(4096));
        }
        for kind in [
            "qwen3_5",
            "qwen3_5_moe",
            "gemma4",
            "gemma4_unified",
            "minicpmv4_6",
        ] {
            let config = json!({"model_type": kind, "max_position_embeddings": 64,
                "text_config": {"max_position_embeddings": 262144}});
            assert_eq!(context_window_from_config(&config, 32768), Some(32768));
            assert_eq!(context_window_from_config(&config, 524288), Some(262144));
        }
    }

    #[test]
    fn unknown_or_invalid_capacity_is_omitted() {
        for value in [
            json!(null),
            json!(0),
            json!(-1),
            json!(1.5),
            json!("8192"),
            json!(2147483648_u64),
        ] {
            let config = json!({"model_type": "llama", "max_position_embeddings": value});
            assert_eq!(context_window_from_config(&config, 32768), None);
        }
        for config in [
            json!({"model_type": "llama"}),
            json!({"model_type": "qwen3_5", "max_position_embeddings": 8192}),
            json!({"model_type": "gemma4", "text_config": null, "max_position_embeddings": 8192}),
            json!({"model_type": "unknown", "max_position_embeddings": 8192}),
            json!({"model_type": "diffusion_gemma", "text_config": {"max_position_embeddings": 8192}}),
        ] {
            assert_eq!(context_window_from_config(&config, 32768), None);
        }
        assert_eq!(
            context_window_from_config(
                &json!({"model_type": "llama", "max_position_embeddings": 8192}),
                0
            ),
            None
        );
    }

    #[test]
    fn advertised_output_budget_prefers_configuration_then_uses_safe_fallback() {
        assert_eq!(
            advertised_max_output_tokens(Some(262_144), None),
            Some(4_096)
        );
        assert_eq!(
            advertised_max_output_tokens(Some(262_144), Some(8_192)),
            Some(8_192)
        );
        assert_eq!(advertised_max_output_tokens(Some(4_096), None), Some(2_048));
        assert_eq!(
            advertised_max_output_tokens(Some(4_096), Some(32_768)),
            Some(2_048)
        );
        assert_eq!(advertised_max_output_tokens(Some(1), None), Some(1));
        assert_eq!(advertised_max_output_tokens(None, Some(8_192)), None);
    }
}
