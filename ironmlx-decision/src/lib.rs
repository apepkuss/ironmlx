//! Native MLX inference for typed decision checkpoints.

pub mod contract;
pub mod laya;
pub mod openai;

pub use contract::{
    DecisionRequest, DecisionResponse, Question, Scores, ScoringRequest, TokenBudgetExceeded,
};
pub use laya::{system_one_response, Laya};

/// Recognize the supported checkpoint without allocating model weights.
pub fn is_laya_checkpoint(dir: &std::path::Path) -> anyhow::Result<bool> {
    let path = dir.join("mlx_config.json");
    if !path.is_file() {
        return Ok(false);
    }
    let config: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    Ok(config["format"] == "laya-mlx"
        && config["format_version"] == 1
        && config["repository"] == contract::MULTILINGUAL_MODEL_ID)
}

mod settings;
pub use settings::{ComputeDevice, ComputeDtype, DecisionSettings};
