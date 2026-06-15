use anyhow::anyhow;
use serde_json::Value;

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// Qwen3.5 dense execution graph. This also covers Qwen3.6 dense
    /// checkpoints that declare `model_type = "qwen3_5"`.
    Qwen35Dense,
    /// Qwen3.5 MoE execution graph. This also covers Qwen3.6 MoE checkpoints
    /// that declare `model_type = "qwen3_5_moe"`.
    Qwen35Moe,
    Gemma4,
    Glm4MoeLite,
    /// Standard Llama-family GQA dense decoder. MiniCPM5-1B ships
    /// `model_type = "llama"` / `architectures = ["LlamaForCausalLM"]` and is a
    /// plain GQA dense checkpoint (no MoE / MLA / sliding-window / Q-K norm).
    Llama,
    /// MiniCPM-V-4.6 (`model_type = "minicpmv4_6"`). A vision-language model
    /// whose language backbone is Qwen3.5-text verbatim
    /// (`text_config.model_type = "qwen3_5_text"`; mlx-vlm derives its
    /// `LanguageModel` from `Qwen35LanguageModel`). The text path runs on the
    /// shared [`Qwen35Dense`](Self::Qwen35Dense) execution graph; the SigLIP
    /// vision tower + resampler are NOT yet supported, so image inputs are out
    /// of scope (text-only).
    MiniCpmV46,
    /// DiffusionGemma block-diffusion architecture (`model_type =
    /// "diffusion_gemma"`). Text-only support uses a dedicated serial
    /// canvas-denoising generation path rather than the causal-LM scheduler.
    DiffusionGemma,
}

impl ModelArchitecture {
    pub const EXPECTED_MODEL_TYPES: &'static str =
        "'qwen3_5', 'qwen3_5_moe', 'gemma4', 'glm4_moe_lite', 'llama', 'minicpmv4_6', or 'diffusion_gemma'";

    pub fn from_config_value(raw: &Value) -> Result<Self> {
        let model_type = raw
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("config.json missing model_type"))?;
        Self::from_model_type(model_type)
    }

    pub fn from_model_type(model_type: &str) -> Result<Self> {
        match model_type {
            "qwen3_5" => Ok(Self::Qwen35Dense),
            "qwen3_5_moe" => Ok(Self::Qwen35Moe),
            "gemma4" => Ok(Self::Gemma4),
            "glm4_moe_lite" => Ok(Self::Glm4MoeLite),
            "llama" => Ok(Self::Llama),
            "minicpmv4_6" => Ok(Self::MiniCpmV46),
            "diffusion_gemma" => Ok(Self::DiffusionGemma),
            other => Err(anyhow!(
                "unsupported model_type: {other} (expected {})",
                Self::EXPECTED_MODEL_TYPES
            )),
        }
    }

    pub fn model_type(self) -> &'static str {
        match self {
            Self::Qwen35Dense => "qwen3_5",
            Self::Qwen35Moe => "qwen3_5_moe",
            Self::Gemma4 => "gemma4",
            Self::Glm4MoeLite => "glm4_moe_lite",
            Self::Llama => "llama",
            Self::MiniCpmV46 => "minicpmv4_6",
            Self::DiffusionGemma => "diffusion_gemma",
        }
    }
}
