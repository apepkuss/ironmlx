use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub num_speakers: usize,
    pub dtype: String,
    pub output_subsampling_factor: usize,
    pub encoder_config: serde_json::Value,
    pub modules_config: serde_json::Value,
    pub processor_config: serde_json::Value,
    pub quantization: Option<serde_json::Value>,
    pub quantization_config: Option<serde_json::Value>,
}
impl Config {
    pub fn load(directory: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)
            .context("invalid Nemotron diarization config")?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "nemotron_diarization",
            "unsupported diarization model_type"
        );
        ensure!(self.num_speakers == 8 && self.dtype == "bfloat16" && self.output_subsampling_factor == 1 && self.quantization.is_none() && self.quantization_config.is_none(), "unsupported Nemotron diarization precision or output profile; expected BF16, 8 speakers and native output resolution");
        let expected = serde_json::json!({
            "feat_in":128,"d_model":512,"n_layers":31,"n_heads":8,"subsampling_factor":8,
            "ff_expansion":4.0,"qkv_bias":false,"qk_norm":false,"pre_block_norm":true,
            "xscaling":false,"rope_base":10000.0,"rotary_fraction":1.0
        });
        for (k, v) in expected.as_object().unwrap() {
            ensure!(
                self.encoder_config.get(k) == Some(v),
                "unsupported Nemotron encoder setting {k}"
            );
        }
        let expected = serde_json::json!({"num_speakers":8,"fc_d_model":512,"tf_d_model":192,"subsampling_factor":8,"chunk_len":340,"fifo_len":40,"spkcache_len":264,"spkcache_update_period":300,"chunk_left_context":0,"chunk_right_context":40,"spkcache_sil_frames_per_spk":1,"pred_score_threshold":0.25,"max_index":99999,"scores_boost_latest":0.05,"sil_threshold":0.2,"strong_boost_rate":0.75,"weak_boost_rate":1.5,"min_pos_scores_rate":0.5,"use_aosc":true,"use_learnable_sil_emb":true,"use_activity_head":true});
        for (k, v) in expected.as_object().unwrap() {
            ensure!(
                self.modules_config.get(k) == Some(v),
                "unsupported Nemotron cache/head setting {k}"
            );
        }
        let expected = serde_json::json!({"feature_size":128,"sampling_rate":16000,"hop_length":160,"n_fft":512,"win_length":400,"preemphasis":0.97,"pad_to":16});
        for (k, v) in expected.as_object().unwrap() {
            ensure!(
                self.processor_config.get(k) == Some(v),
                "unsupported Nemotron preprocessor setting {k}"
            );
        }
        Ok(())
    }
}
#[derive(Serialize)]
pub struct Metadata {
    pub model_type: &'static str,
    pub artifact_role: &'static str,
}
pub fn preflight_model_metadata(directory: &Path) -> Result<Metadata> {
    Config::load(directory)?;
    Ok(Metadata {
        model_type: "nemotron_diarization",
        artifact_role: "diarization",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = include_str!("../../tests/fixtures/nemotron/config.json");
    #[test]
    fn published_bf16_profile_and_unknown_architecture_gates() -> Result<()> {
        let config: Config = serde_json::from_str(SOURCE)?;
        config.validate()?;
        for field in [
            "dtype",
            "num_speakers",
            "output_subsampling_factor",
            "quantization",
        ] {
            let mut value: serde_json::Value = serde_json::from_str(SOURCE)?;
            value[field] = match field {
                "dtype" => "float16".into(),
                "quantization" => serde_json::json!({"bits":8}),
                _ => 0.into(),
            };
            assert!(serde_json::from_value::<Config>(value)?.validate().is_err());
        }
        for (section, key) in [
            ("encoder_config", "n_layers"),
            ("modules_config", "chunk_len"),
            ("processor_config", "sampling_rate"),
        ] {
            let mut value: serde_json::Value = serde_json::from_str(SOURCE)?;
            value[section][key] = 0.into();
            assert!(serde_json::from_value::<Config>(value)?.validate().is_err());
        }
        Ok(())
    }
}
