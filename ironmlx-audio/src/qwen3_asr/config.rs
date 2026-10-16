use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub thinker_config: ThinkerConfig,
    pub quantization: Quantization,
    pub support_languages: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
    pub audio_token_id: u32,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AudioConfig {
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: i32,
    pub encoder_ffn_dim: i32,
    pub d_model: i32,
    pub n_window: usize,
    pub output_dim: i32,
    pub n_window_infer: usize,
    pub downsample_hidden_size: i32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Quantization {
    pub mode: String,
    pub bits: i32,
    pub group_size: i32,
    #[serde(default)]
    pub override_count: usize,
}

#[derive(Debug, Serialize)]
pub struct Metadata {
    pub model_type: &'static str,
    pub artifact_role: &'static str,
    pub quantization: Quantization,
}

impl Config {
    pub fn load(directory: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)
            .context("invalid Qwen3-ASR config.json")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "qwen3_asr",
            "unsupported model_type: {}",
            self.model_type
        );
        let a = &self.thinker_config.audio_config;
        let t = &self.thinker_config.text_config;
        let profile = (
            a.d_model,
            a.encoder_layers,
            a.encoder_attention_heads,
            a.encoder_ffn_dim,
            t.hidden_size,
            t.intermediate_size,
        );
        ensure!(
            matches!(
                profile,
                (896, 18, 14, 3584, 1024, 3072) | (1024, 24, 16, 4096, 2048, 6144)
            ),
            "unsupported Qwen3-ASR encoder/decoder profile: {profile:?}"
        );
        ensure!(
            a.num_mel_bins == 128
                && a.n_window == 50
                && a.n_window_infer == 800
                && a.downsample_hidden_size == 480
                && a.output_dim == t.hidden_size,
            "unsupported Qwen3-ASR audio configuration"
        );
        ensure!(
            t.num_hidden_layers == 28
                && t.num_attention_heads == 16
                && t.num_key_value_heads == 8
                && t.head_dim == 128
                && t.vocab_size == 151936
                && t.tie_word_embeddings
                && t.max_position_embeddings == 65536
                && t.rms_norm_eps == 1e-6
                && t.rope_theta == 1_000_000.,
            "unsupported Qwen3-ASR text configuration"
        );
        ensure!(
            self.thinker_config.audio_token_id == 151676
                && self.thinker_config.audio_start_token_id == 151669
                && self.thinker_config.audio_end_token_id == 151670,
            "invalid Qwen3-ASR audio token IDs"
        );
        ensure!(
            self.quantization.mode == "affine"
                && self.quantization.bits == 8
                && self.quantization.group_size == 64,
            "unsupported Qwen3-ASR quantization: expected affine 8bit, group_size 64"
        );
        ensure!(
            !self.support_languages.is_empty(),
            "missing Qwen3-ASR supported languages"
        );
        Ok(())
    }

    pub fn language(&self, language: Option<&str>) -> Result<Option<String>> {
        let Some(value) = language else {
            return Ok(None);
        };
        let name = match value {
            "zh" => "Chinese",
            "en" => "English",
            "yue" => "Cantonese",
            "ar" => "Arabic",
            "de" => "German",
            "fr" => "French",
            "es" => "Spanish",
            "pt" => "Portuguese",
            "id" => "Indonesian",
            "it" => "Italian",
            "ko" => "Korean",
            "ru" => "Russian",
            "th" => "Thai",
            "vi" => "Vietnamese",
            "ja" => "Japanese",
            "tr" => "Turkish",
            "hi" => "Hindi",
            "ms" => "Malay",
            "nl" => "Dutch",
            "sv" => "Swedish",
            "da" => "Danish",
            "fi" => "Finnish",
            "pl" => "Polish",
            "cs" => "Czech",
            "fil" => "Filipino",
            "fa" => "Persian",
            "el" => "Greek",
            "hu" => "Hungarian",
            "mk" => "Macedonian",
            "ro" => "Romanian",
            other => other,
        };
        match self
            .support_languages
            .iter()
            .find(|s| s.eq_ignore_ascii_case(name))
        {
            Some(value) => Ok(Some(value.clone())),
            None => bail!("unsupported ASR language: {value}"),
        }
    }
}

pub fn preflight_model_metadata(directory: &Path) -> Result<Metadata> {
    let config = Config::load(directory)?;
    let processor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("preprocessor_config.json"))?)?;
    ensure!(
        processor["feature_extractor_type"] == "WhisperFeatureExtractor"
            && processor["feature_size"] == 128
            && processor["n_fft"] == 400
            && processor["hop_length"] == 160,
        "invalid Qwen3-ASR preprocessor configuration"
    );
    super::tokenizer::load(directory)?;
    Ok(Metadata {
        model_type: "qwen3_asr",
        artifact_role: "asr",
        quantization: config.quantization,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const SMALL: &str = include_str!("../../tests/fixtures/qwen3_asr/0.6B-config.json");
    const LARGE: &str = include_str!("../../tests/fixtures/qwen3_asr/1.7B-config.json");

    #[test]
    fn both_published_profiles_and_languages_are_supported() -> Result<()> {
        for source in [SMALL, LARGE] {
            let config: Config = serde_json::from_str(source)?;
            config.validate()?;
            assert_eq!(config.language(Some("zh"))?.as_deref(), Some("Chinese"));
            assert_eq!(config.language(Some("en"))?.as_deref(), Some("English"));
            assert_eq!(config.language(Some("yue"))?.as_deref(), Some("Cantonese"));
            assert_eq!(config.language(None)?, None);
            assert!(config.language(Some("not-a-language")).is_err());
        }
        Ok(())
    }
    #[test]
    fn incompatible_profiles_are_rejected_before_weights() -> Result<()> {
        for (key, value) in [("bits", 4), ("group_size", 32)] {
            let mut config: Config = serde_json::from_str(SMALL)?;
            if key == "bits" {
                config.quantization.bits = value;
            } else {
                config.quantization.group_size = value;
            }
            assert!(config.validate().is_err());
        }
        let mut config: Config = serde_json::from_str(LARGE)?;
        config.thinker_config.text_config.head_dim = 64;
        assert!(config.validate().is_err());
        Ok(())
    }
}
