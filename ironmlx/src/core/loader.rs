//! Model loader — opens a directory containing `config.json` +
//! `tokenizer_config.json` + safetensors weights, exposes tensor lookup
//! by full key and parsed quantization metadata.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context};
use mlx::Array;
use serde::Deserialize;

use crate::Result;

/// Quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Affine quantization (scale + offset per group).
    Affine,
}

/// Quantization metadata parsed from `config.json`.
#[derive(Debug, Clone, Copy)]
pub struct QuantMeta {
    /// Group size for per-group quantization parameters.
    pub group_size: i32,
    /// Bits per quantized weight (4, 6, 8).
    pub bits: i32,
    /// Quantization scheme.
    pub mode: QuantMode,
}

/// HF `eos_token_id` field — single int or list of ints.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    /// Single end-of-sequence token id.
    Single(u32),
    /// Multiple end-of-sequence token ids.
    Multi(Vec<u32>),
}

/// Subset of `tokenizer_config.json` fields ironmlx cares about.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenizerConfig {
    /// Jinja chat template string.
    #[serde(default)]
    pub chat_template: Option<String>,
    /// EOS token literal (e.g. `"</s>"`).
    #[serde(default)]
    pub eos_token: Option<String>,
    /// BOS token literal.
    #[serde(default)]
    pub bos_token: Option<String>,
    /// Pad token literal.
    #[serde(default)]
    pub pad_token: Option<String>,
    /// EOS token id(s) — single or list.
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
}

/// Mmap-eager safetensors model loader. Owns all tensor data + parsed config.
pub struct Loader {
    tensors: HashMap<String, Array>,
    quant: Option<QuantMeta>,
    tokenizer_config: TokenizerConfig,
    config_raw: serde_json::Value,
    model_dir: std::path::PathBuf,
}

impl Loader {
    /// Open a directory containing `config.json`, `tokenizer_config.json`,
    /// and `model.safetensors` (single-file) or `model.safetensors.index.json`
    /// (sharded). All weights are mmap-loaded eagerly.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_raw: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("opening {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;

        let tok_path = model_dir.join("tokenizer_config.json");
        let tokenizer_config: TokenizerConfig = if tok_path.exists() {
            serde_json::from_reader(
                std::fs::File::open(&tok_path)
                    .with_context(|| format!("opening {}", tok_path.display()))?,
            )
            .with_context(|| format!("parsing {}", tok_path.display()))?
        } else {
            TokenizerConfig::default()
        };

        let quant = parse_quant_meta(&config_raw)?;

        let tensors = load_safetensors(model_dir)?;

        Ok(Self {
            tensors,
            quant,
            tokenizer_config,
            config_raw,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// Returns tensor by full key, or error if absent.
    pub fn tensor(&self, key: &str) -> Result<&Array> {
        self.tensors
            .get(key)
            .ok_or_else(|| anyhow!("Loader: missing tensor key `{key}`"))
    }

    /// Returns tensor by full key, or None if absent.
    pub fn tensor_opt(&self, key: &str) -> Option<&Array> {
        self.tensors.get(key)
    }

    /// True iff the key is present.
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Iterator over all loaded tensor keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Quantization metadata, or None if model is not quantized.
    pub fn quant_meta(&self) -> Option<QuantMeta> {
        self.quant
    }

    /// Parse model-specific config struct via serde.
    pub fn config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        Ok(serde_json::from_value(self.config_raw.clone())?)
    }

    /// Raw `serde_json::Value` of the parsed `config.json`. Used by model-
    /// specific code that needs to navigate nested keys (e.g. `text_config`)
    /// without a wrapping struct.
    pub fn config_raw_value(&self) -> &serde_json::Value {
        &self.config_raw
    }

    /// Tokenizer config (chat template, eos token, etc.).
    pub fn tokenizer_config(&self) -> &TokenizerConfig {
        &self.tokenizer_config
    }

    /// Path to the model directory.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}

fn parse_quant_meta(config_raw: &serde_json::Value) -> Result<Option<QuantMeta>> {
    let Some(q) = config_raw
        .get("quantization")
        .or_else(|| config_raw.get("quantization_config"))
    else {
        return Ok(None);
    };
    let group_size_i64 = q
        .get("group_size")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("quantization.group_size missing or non-int"))?;
    let group_size =
        i32::try_from(group_size_i64).context("quantization.group_size out of i32 range")?;
    let bits_i64 = q
        .get("bits")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("quantization.bits missing or non-int"))?;
    let bits = i32::try_from(bits_i64).context("quantization.bits out of i32 range")?;
    let mode_str = q.get("mode").and_then(|m| m.as_str()).unwrap_or("affine");
    let mode = match mode_str {
        "affine" => QuantMode::Affine,
        other => return Err(anyhow!("unsupported quantization.mode `{other}`")),
    };
    Ok(Some(QuantMeta {
        group_size,
        bits,
        mode,
    }))
}

fn load_safetensors(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let single = model_dir.join("model.safetensors");
    let sharded = model_dir.join("model.safetensors.index.json");

    if single.exists() {
        let (tensors, _meta) = mlx::io::load_safetensors(
            single
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 path: {}", single.display()))?,
        )
        .map_err(|e| anyhow!("load_safetensors {}: {e}", single.display()))?;
        return Ok(tensors);
    }

    if sharded.exists() {
        let idx_text = std::fs::read_to_string(&sharded)?;
        let idx: serde_json::Value = serde_json::from_str(&idx_text)?;
        let weight_map = idx
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("safetensors index missing weight_map"))?;

        let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for v in weight_map.values() {
            if let Some(s) = v.as_str() {
                shards.insert(s.to_owned());
            }
        }

        let mut all = HashMap::new();
        for shard_name in shards {
            let shard_path = model_dir.join(&shard_name);
            let (tensors, _meta) = mlx::io::load_safetensors(
                shard_path
                    .to_str()
                    .ok_or_else(|| anyhow!("non-utf8 path: {}", shard_path.display()))?,
            )
            .map_err(|e| anyhow!("load_safetensors {}: {e}", shard_path.display()))?;
            all.extend(tensors);
        }
        return Ok(all);
    }

    Err(anyhow!(
        "no model.safetensors or model.safetensors.index.json in {}",
        model_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_quant_meta_affine_4bit() {
        let cfg = json!({
            "quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
        });
        let q = parse_quant_meta(&cfg).unwrap().expect("quant");
        assert_eq!(q.group_size, 64);
        assert_eq!(q.bits, 4);
        assert_eq!(q.mode, QuantMode::Affine);
    }

    #[test]
    fn parse_quant_meta_falls_back_to_quantization_config() {
        let cfg = json!({
            "quantization_config": { "group_size": 128, "bits": 8, "mode": "affine" }
        });
        let q = parse_quant_meta(&cfg).unwrap().expect("quant");
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 128);
    }

    #[test]
    fn parse_quant_meta_returns_none_when_absent() {
        let cfg = json!({ "model_type": "qwen3_5" });
        assert!(parse_quant_meta(&cfg).unwrap().is_none());
    }

    #[test]
    fn parse_quant_meta_errors_on_unknown_mode() {
        let cfg = json!({
            "quantization": { "group_size": 64, "bits": 4, "mode": "fp8" }
        });
        assert!(parse_quant_meta(&cfg).is_err());
    }

    #[test]
    fn eos_token_id_single_or_multi() {
        let s: EosTokenId = serde_json::from_str("42").unwrap();
        assert!(matches!(s, EosTokenId::Single(42)));
        let m: EosTokenId = serde_json::from_str("[1, 2, 3]").unwrap();
        match m {
            EosTokenId::Multi(v) => assert_eq!(v, vec![1, 2, 3]),
            _ => panic!("expected Multi"),
        }
    }
}
