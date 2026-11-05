use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::BTreeSet,
    path::{Component, Path},
};

#[derive(Clone, Debug, Deserialize)]
pub(super) struct StackConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    #[serde(default = "theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Value,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub eoi_token_index: i32,
    #[serde(default)]
    pub vocab_size: i32,
}
fn theta() -> f32 {
    10000.
}
#[derive(Clone, Debug, Deserialize)]
pub(super) struct Config {
    pub model_type: String,
    pub num_codebooks: i32,
    pub vocab_size: i32,
    pub backbone_config: StackConfig,
    pub depth_decoder_config: StackConfig,
    pub text_encoder_config: StackConfig,
    #[serde(default)]
    pub quantization: Option<Value>,
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        ensure!(c.model_type == "breeze_tts", "expected breeze_tts");
        ensure!(
            c.num_codebooks == 16 && c.vocab_size == 2051,
            "unsupported Breeze codebook layout"
        );
        let b = &c.backbone_config;
        let d = &c.depth_decoder_config;
        let t = &c.text_encoder_config;
        ensure!(
            (
                b.hidden_size,
                b.intermediate_size,
                b.num_hidden_layers,
                b.num_attention_heads,
                b.num_key_value_heads,
                b.head_dim
            ) == (2048, 6144, 28, 16, 8, 128),
            "unsupported Breeze backbone"
        );
        ensure!(
            (
                d.hidden_size,
                d.intermediate_size,
                d.num_hidden_layers,
                d.num_attention_heads,
                d.num_key_value_heads,
                d.head_dim
            ) == (1024, 8192, 12, 8, 2, 128),
            "unsupported Breeze depth decoder"
        );
        ensure!(
            (
                t.hidden_size,
                t.intermediate_size,
                t.num_hidden_layers,
                t.num_attention_heads,
                t.num_key_value_heads,
                t.head_dim,
                t.vocab_size,
                t.eoi_token_index
            ) == (1152, 6912, 26, 4, 1, 256, 262158, 256000),
            "unsupported Breeze text encoder"
        );
        ensure!(
            t.layer_types.len() == t.num_hidden_layers
                && t.layer_types
                    .iter()
                    .all(|v| matches!(v.as_str(), "full_attention" | "sliding_attention")),
            "invalid Breeze text attention pattern"
        );
        ensure!(
            t.sliding_window == Some(512),
            "unsupported Breeze text window"
        );
        for s in [b, d, t] {
            ensure!(
                s.rms_norm_eps.is_finite()
                    && s.rms_norm_eps > 0.
                    && s.rope_theta.is_finite()
                    && s.rope_theta > 0.,
                "invalid Breeze normalization/RoPE"
            );
        }
        if let Some(q) = &c.quantization {
            ensure!(
                q["mode"] == "mxfp8" && q["bits"] == 8 && q["group_size"] == 32,
                "Breeze supports BF16 or MXFP8 group_size=32"
            );
        }
        Ok(c)
    }
}
pub(super) fn shards(path: &Path) -> Result<Vec<String>> {
    let index: Value =
        serde_json::from_slice(&std::fs::read(path.join("model.safetensors.index.json"))?)?;
    let map = index["weight_map"]
        .as_object()
        .context("missing Breeze weight_map")?;
    ensure!(!map.is_empty(), "empty Breeze weight index");
    let mut files = BTreeSet::new();
    for name in map.values() {
        let name = name.as_str().context("invalid Breeze shard name")?;
        ensure!(
            Path::new(name)
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
                && name.ends_with(".safetensors"),
            "unsafe Breeze shard path"
        );
        files.insert(name.to_owned());
    }
    Ok(files.into_iter().collect())
}
/// Inspect the fixed native Breeze profile without initializing MLX.
pub fn preflight(path: &Path) -> Result<u64> {
    Config::load(path)?;
    let c: Value =
        serde_json::from_slice(&std::fs::read(path.join("audio_tokenizer/config.json"))?)?;
    ensure!(
        c["model_type"] == "qwen3_tts_tokenizer_12hz"
            && c["input_sample_rate"] == 24000
            && c["output_sample_rate"] == 24000
            && c["encoder_valid_num_quantizers"] == 16
            && c["decode_upsample_rate"] == 1920,
        "unsupported Breeze codec"
    );
    let mut bytes = 0u64;
    for f in shards(path)?
        .iter()
        .map(String::as_str)
        .chain(["audio_tokenizer/model.safetensors"])
    {
        let meta = std::fs::metadata(path.join(f))
            .with_context(|| format!("missing Breeze resource {f}"))?;
        ensure!(meta.is_file() && meta.len() > 8, "empty Breeze weight {f}");
        bytes = bytes
            .checked_add(meta.len())
            .context("Breeze weight size overflow")?;
    }
    for f in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "audio_tokenizer/preprocessor_config.json",
    ] {
        ensure!(path.join(f).is_file(), "missing Breeze resource {f}");
    }
    Ok(bytes)
}

/// Metadata-only probe used before the App downloads model weights.
pub fn preflight_model_metadata(path: &Path) -> Result<Value> {
    let c = Config::load(path)?;
    let quantization = c
        .quantization
        .map(|_| serde_json::json!({"mode":"mxfp8","bits":8,"group_size":32,"override_count":0}));
    Ok(
        serde_json::json!({"model_type":"breeze_tts","artifact_role":"tts","quantization":quantization}),
    )
}
