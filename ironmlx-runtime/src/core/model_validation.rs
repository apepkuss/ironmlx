//! Native model compatibility validation without admin request/response types.
use crate::core::speculative::{MtpDraftTokensArg, MtpSpeculativeConfig};
use crate::Result;
use anyhow::Context;
use ironmlx_core::sampler::Sampler;
use std::path::Path;
use {
    ironmlx_lm::models::Gemma4AssistantConfig, ironmlx_lm::models::Gemma4Config,
    ironmlx_lm::models::ModelArchitecture, ironmlx_lm::models::Qwen35Config,
    ironmlx_lm::models::Qwen35MoeConfig,
};
pub const MTP_BASE_MODEL_NOT_FOUND_CODE: &str = "mtp_base_model_not_found";
pub const MTP_INCOMPATIBLE_CODE: &str = "mtp_incompatible";
pub const MTP_INVALID_CONFIG_CODE: &str = "mtp_invalid_config";
pub const MTP_INVALID_DRAFT_TOKENS_CODE: &str = "mtp_invalid_draft_tokens";
pub const MTP_INVALID_MODEL_TYPE_CODE: &str = "mtp_invalid_model_type";
pub const MTP_MODEL_NOT_FOUND_CODE: &str = "mtp_model_not_found";
pub const MTP_OK_CODE: &str = "ok";
pub const MTP_UNSUPPORTED_ARCHITECTURE_CODE: &str = "mtp_unsupported_architecture";
#[derive(Debug)]
pub struct MtpCompatibility {
    pub compatible: bool,
    pub reason_code: &'static str,
    pub message: String,
    pub draft_tokens: Option<usize>,
}

impl MtpCompatibility {
    fn compatible(draft_tokens: usize) -> Self {
        Self {
            compatible: true,
            reason_code: MTP_OK_CODE,
            message: "MTP weights are compatible with this model.".to_string(),
            draft_tokens: Some(draft_tokens),
        }
    }

    fn not_compatible(
        reason_code: &'static str,
        message: impl Into<String>,
        draft_tokens: Option<usize>,
    ) -> Self {
        Self {
            compatible: false,
            reason_code,
            message: message.into(),
            draft_tokens,
        }
    }
}

pub fn validate_mtp_pair(
    model_dir: &Path,
    mtp_model_dir: &Path,
    mtp_draft_tokens: Option<usize>,
) -> Result<MtpCompatibility> {
    if !model_dir.is_dir() {
        return Ok(MtpCompatibility::not_compatible(
            MTP_BASE_MODEL_NOT_FOUND_CODE,
            format!(
                "Base model directory does not exist: {}",
                model_dir.display()
            ),
            None,
        ));
    }
    if !mtp_model_dir.is_dir() {
        return Ok(MtpCompatibility::not_compatible(
            MTP_MODEL_NOT_FOUND_CODE,
            format!(
                "MTP model directory does not exist: {}",
                mtp_model_dir.display()
            ),
            None,
        ));
    }

    let base_raw = match read_config_json(model_dir) {
        Ok(raw) => raw,
        Err(error) => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            ));
        }
    };
    let model_type = base_raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let architecture = match ModelArchitecture::from_model_type(model_type) {
        Ok(architecture) => architecture,
        Err(error) => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_UNSUPPORTED_ARCHITECTURE_CODE,
                format!("{error:#}"),
                None,
            ));
        }
    };
    match architecture {
        ModelArchitecture::Qwen35Dense
        | ModelArchitecture::Qwen35Moe
        | ModelArchitecture::Gemma4 => {}
        _ => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_UNSUPPORTED_ARCHITECTURE_CODE,
                "MTP currently supports Qwen/Gemma4 models only.",
                None,
            ));
        }
    }

    let mtp_raw = match read_config_json(mtp_model_dir) {
        Ok(raw) => raw,
        Err(error) => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            ));
        }
    };
    let mtp_model_type = mtp_raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let valid_mtp_type = match architecture {
        ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe => {
            mtp_model_type == "qwen3_5_mtp"
        }
        ModelArchitecture::Gemma4 => {
            matches!(
                mtp_model_type,
                "gemma4_assistant" | "gemma4_unified_assistant"
            )
        }
        _ => unreachable!("MTP architecture was filtered above"),
    };
    if !valid_mtp_type {
        let expected = match architecture {
            ModelArchitecture::Qwen35Dense | ModelArchitecture::Qwen35Moe => "qwen3_5_mtp",
            ModelArchitecture::Gemma4 => "gemma4_assistant or gemma4_unified_assistant",
            _ => unreachable!("MTP architecture was filtered above"),
        };
        return Ok(MtpCompatibility::not_compatible(
            MTP_INVALID_MODEL_TYPE_CODE,
            format!("Expected MTP model_type={expected}, got {mtp_model_type}"),
            None,
        ));
    }

    match architecture {
        ModelArchitecture::Qwen35Dense => {
            if let Some(response) = validate_qwen35_dense_mtp_config(&base_raw, &mtp_raw)? {
                return Ok(response);
            }
        }
        ModelArchitecture::Qwen35Moe => {
            if let Some(response) = validate_qwen35_moe_mtp_config(&base_raw, &mtp_raw)? {
                return Ok(response);
            }
        }
        ModelArchitecture::Gemma4 => {
            if let Some(response) = validate_gemma4_assistant_mtp_config(&base_raw, &mtp_raw)? {
                return Ok(response);
            }
        }
        _ => unreachable!("MTP architecture was filtered above"),
    }

    let base_lineage = match mtp_model_lineage(model_dir, &base_raw) {
        Ok(lineage) => lineage,
        Err(error) => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            ));
        }
    };
    let mtp_lineage = match mtp_model_lineage(mtp_model_dir, &mtp_raw) {
        Ok(lineage) => lineage,
        Err(error) => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            ));
        }
    };
    match (base_lineage.as_deref(), mtp_lineage.as_deref()) {
        (Some(base), Some(mtp)) if base == mtp => {}
        (Some(base), Some(mtp)) => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_INCOMPATIBLE_CODE,
                format!("MTP model lineage mismatch: base={base} mtp={mtp}"),
                None,
            ));
        }
        _ => {
            return Ok(MtpCompatibility::not_compatible(
                MTP_INCOMPATIBLE_CODE,
                "MTP model lineage could not be established from model metadata.",
                None,
            ));
        }
    }

    let draft_tokens = crate::core::speculative::resolve_mtp_draft_tokens(
        &base_raw,
        mtp_draft_tokens
            .map(MtpDraftTokensArg::Explicit)
            .unwrap_or(MtpDraftTokensArg::Omitted),
    );
    if let Err(error) = MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy()) {
        return Ok(MtpCompatibility::not_compatible(
            MTP_INVALID_DRAFT_TOKENS_CODE,
            format!("{error:#}"),
            Some(draft_tokens),
        ));
    }
    Ok(MtpCompatibility::compatible(draft_tokens))
}

fn validate_qwen35_dense_mtp_config(
    base_raw: &serde_json::Value,
    mtp_raw: &serde_json::Value,
) -> Result<Option<MtpCompatibility>> {
    let base_cfg = match qwen35_config_from_raw(base_raw) {
        Ok(config) => config,
        Err(error) => {
            return Ok(Some(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            )));
        }
    };
    let mtp_cfg = match Qwen35Config::from_mtp_config_value(mtp_raw) {
        Ok(config) => config,
        Err(error) => {
            return Ok(Some(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            )));
        }
    };
    if let Err(error) = mtp_cfg.mtp_config() {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    if let Err(error) = base_cfg.ensure_mtp_compatible(&mtp_cfg) {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    Ok(None)
}

fn validate_qwen35_moe_mtp_config(
    base_raw: &serde_json::Value,
    mtp_raw: &serde_json::Value,
) -> Result<Option<MtpCompatibility>> {
    let base_cfg = match qwen35_moe_config_from_raw(base_raw) {
        Ok(config) => config,
        Err(error) => {
            return Ok(Some(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            )));
        }
    };
    let mtp_cfg = match Qwen35MoeConfig::from_mtp_config_value(mtp_raw) {
        Ok(config) => config,
        Err(error) => {
            return Ok(Some(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            )));
        }
    };
    if let Err(error) = mtp_cfg.mtp_config() {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    if let Err(error) = base_cfg.ensure_mtp_compatible(&mtp_cfg) {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    Ok(None)
}

fn validate_gemma4_assistant_mtp_config(
    base_raw: &serde_json::Value,
    mtp_raw: &serde_json::Value,
) -> Result<Option<MtpCompatibility>> {
    let mut base_cfg: Gemma4Config = match serde_json::from_value(base_raw.clone())
        .context("failed to deserialize Gemma4Config")
    {
        Ok(config) => config,
        Err(error) => {
            return Ok(Some(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            )));
        }
    };
    if let Err(error) = base_cfg.validate_and_finalize() {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    let mut mtp_cfg: Gemma4AssistantConfig = match serde_json::from_value(mtp_raw.clone())
        .context("failed to deserialize Gemma4AssistantConfig")
    {
        Ok(config) => config,
        Err(error) => {
            return Ok(Some(MtpCompatibility::not_compatible(
                MTP_INVALID_CONFIG_CODE,
                format!("{error:#}"),
                None,
            )));
        }
    };
    if let Err(error) = mtp_cfg.validate_and_finalize() {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INVALID_CONFIG_CODE,
            format!("{error:#}"),
            None,
        )));
    }
    let expected_assistant_type = match base_cfg.model_type.as_str() {
        "gemma4" => "gemma4_assistant",
        "gemma4_unified" => "gemma4_unified_assistant",
        _ => unreachable!("Gemma4Config validation filtered model type"),
    };
    if mtp_cfg.model_type != expected_assistant_type {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!(
                "Gemma4 base model_type={} requires assistant model_type={expected_assistant_type}, got {}",
                base_cfg.model_type, mtp_cfg.model_type
            ),
            None,
        )));
    }
    if mtp_cfg.backbone_hidden_size != base_cfg.text_config.hidden_size {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!(
                "Gemma4 assistant backbone_hidden_size={} must match base hidden_size={}",
                mtp_cfg.backbone_hidden_size, base_cfg.text_config.hidden_size
            ),
            None,
        )));
    }
    if mtp_cfg.text_config.vocab_size != base_cfg.text_config.vocab_size {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            format!(
                "Gemma4 assistant vocab_size={} must match base vocab_size={}",
                mtp_cfg.text_config.vocab_size, base_cfg.text_config.vocab_size
            ),
            None,
        )));
    }
    if !mtp_cfg.text_config.all_layers_share_external_kv() {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            "Gemma4 assistant must share K/V for every drafter layer".to_owned(),
            None,
        )));
    }

    let mut base_has_sliding = false;
    let mut base_has_full = false;
    for idx in 0..base_cfg.text_config.num_hidden_layers as usize {
        match base_cfg.text_config.layer_kind(idx) {
            ironmlx_lm::models::gemma4::Gemma4LayerKind::Sliding => base_has_sliding = true,
            ironmlx_lm::models::gemma4::Gemma4LayerKind::Full => base_has_full = true,
        }
    }
    let mut assistant_has_sliding = false;
    let mut assistant_has_full = false;
    for idx in 0..mtp_cfg.text_config.num_hidden_layers as usize {
        match mtp_cfg.text_config.layer_kind(idx) {
            ironmlx_lm::models::gemma4::Gemma4LayerKind::Sliding => assistant_has_sliding = true,
            ironmlx_lm::models::gemma4::Gemma4LayerKind::Full => assistant_has_full = true,
        }
    }
    if (base_has_sliding && !assistant_has_sliding) || (base_has_full && !assistant_has_full) {
        return Ok(Some(MtpCompatibility::not_compatible(
            MTP_INCOMPATIBLE_CODE,
            "Gemma4 assistant layer_types must cover every layer type used by the base model"
                .to_owned(),
            None,
        )));
    }
    Ok(None)
}

pub fn read_config_json(model_dir: &Path) -> Result<serde_json::Value> {
    let path = model_dir.join("config.json");
    let data = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))
}

/// Resolve the model identity that an MTP artifact was trained against.
///
/// Execution graph identifiers are intentionally broad: `qwen3_5`, for
/// example, covers Qwen3.5, Qwen3.6, and Qwen3.8. Weight compatibility must
/// therefore also retain the model generation and parameter variant encoded in
/// the source model identity.
fn mtp_model_lineage(model_dir: &Path, raw: &serde_json::Value) -> Result<Option<String>> {
    let text = raw.get("text_config");
    for identity in [
        raw.get("base_model_name_or_path")
            .and_then(serde_json::Value::as_str),
        raw.get("_name_or_path").and_then(serde_json::Value::as_str),
        text.and_then(|value| value.get("base_model_name_or_path"))
            .and_then(serde_json::Value::as_str),
        text.and_then(|value| value.get("_name_or_path"))
            .and_then(serde_json::Value::as_str),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(lineage) = normalized_mtp_lineage(identity) {
            return Ok(Some(lineage));
        }
    }

    let manifest_path = model_dir.join(".ironmlx-snapshot.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let data = std::fs::read(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&data)
        .with_context(|| format!("parsing {}", manifest_path.display()))?;
    Ok(manifest
        .get("repo_id")
        .and_then(serde_json::Value::as_str)
        .and_then(normalized_mtp_lineage))
}

fn normalized_mtp_lineage(identity: &str) -> Option<String> {
    let leaf = identity
        .trim()
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default();
    let normalized = leaf.to_ascii_lowercase();
    let tokens = normalized
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '.'))
        .filter(|token| !token.is_empty() && !is_mtp_packaging_marker(token))
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join("-"))
    }
}

fn is_mtp_packaging_marker(token: &str) -> bool {
    if matches!(
        token,
        "mtp"
            | "assistant"
            | "drafter"
            | "mlx"
            | "qat"
            | "optiq"
            | "awq"
            | "gptq"
            | "gguf"
            | "quantized"
            | "bfloat16"
            | "float16"
    ) {
        return true;
    }
    if token
        .strip_suffix("bit")
        .is_some_and(|bits| bits.parse::<u8>().is_ok())
    {
        return true;
    }
    ["mxfp", "int", "fp", "bf"].into_iter().any(|prefix| {
        token
            .strip_prefix(prefix)
            .is_some_and(|bits| bits.parse::<u8>().is_ok())
    })
}

fn qwen35_config_from_raw(raw: &serde_json::Value) -> Result<Qwen35Config> {
    let text_config = raw
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("config.json missing text_config field"))?;
    let mut cfg: Qwen35Config = serde_json::from_value(text_config.clone())
        .context("failed to deserialize Qwen35Config from text_config")?;
    if let Some(vision_config) = raw.get("vision_config") {
        cfg.vision_config = Some(
            serde_json::from_value(vision_config.clone())
                .context("failed to deserialize VisionConfig")?,
        );
    }
    Ok(cfg)
}

fn qwen35_moe_config_from_raw(raw: &serde_json::Value) -> Result<Qwen35MoeConfig> {
    Qwen35MoeConfig::from_raw_config_value(raw)
}
