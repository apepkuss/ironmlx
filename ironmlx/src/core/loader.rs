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

        let mut tensors = load_safetensors(model_dir)?;

        Self::sanitize(&mut tensors, &config_raw)?;

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

    /// HF Qwen3.5 sanitize aligned with mlx-lm `qwen3_5.py::Model::sanitize` +
    /// `TextModel::sanitize`.
    ///
    /// Mutates `weights` in place:
    /// 0. Drop `vision_tower.*` keys (vision encoder not used for LLM-only inference).
    ///    Strip `language_model.` prefix from all remaining keys so that downstream
    ///    code can use plain `model.*` paths (e.g. `model.embed_tokens.weight`).
    /// 1. Strips `mtp.*` keys (the dedicated MTP head — see P8c).
    /// 2. If `text_config.tie_word_embeddings`, drops `lm_head.{weight,scales,biases}`.
    /// 3. `transpose_axes [0, 2, 1]` on `conv1d.weight` tensors whose last dim != 1
    ///    (HF stores them as `[out, in, k]`; cxx-mlx Conv1d wants `[out, k, in]`).
    /// 4. Adds `1.0` to all 1-D RmsNorm weights at known suffixes when either
    ///    `mtp.*` was present OR an unsanitized conv1d was detected — the HF
    ///    "offset gamma" convention.
    fn sanitize(
        weights: &mut HashMap<String, Array>,
        config_raw: &serde_json::Value,
    ) -> Result<()> {
        // 0. Drop vision_tower.* keys unconditionally (vision encoder not needed
        //    for LLM-only inference regardless of checkpoint layout).
        weights.retain(|k, _| !k.starts_with("vision_tower."));

        // Strip language_model. prefix when present so downstream code can use
        // plain model.* paths.  This matches the multimodal Qwen3.5 checkpoint
        // layout where text-model weights sit under language_model.*.
        if weights.keys().any(|k| k.starts_with("language_model.")) {
            let old: HashMap<String, Array> = std::mem::take(weights);
            for (k, v) in old {
                let new_key = k
                    .strip_prefix("language_model.")
                    .map_or(k.clone(), str::to_owned);
                weights.insert(new_key, v);
            }
        }

        // Detection BEFORE mutation (after prefix strip so keys are in model.* form).
        let has_mtp = weights.keys().any(|k| k.contains("mtp."));
        let has_unsanitized_conv1d = weights.iter().any(|(k, v)| {
            k.ends_with("conv1d.weight") && v.shape().as_slice().last().copied().unwrap_or(1) != 1
        });
        let should_shift_norm = has_mtp || has_unsanitized_conv1d;

        // 1. Strip mtp.*
        weights.retain(|k, _| !k.contains("mtp."));

        // 2. Strip lm_head if tied.
        let tie = config_raw
            .get("text_config")
            .and_then(|tc| tc.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if tie {
            weights.remove("lm_head.weight");
            weights.remove("lm_head.scales");
            weights.remove("lm_head.biases");
        }

        // 3. conv1d.weight transpose_axes [0, 2, 1] if old form.
        let conv1d_keys: Vec<String> = weights
            .iter()
            .filter(|(k, v)| {
                k.ends_with("conv1d.weight")
                    && v.shape().as_slice().len() == 3
                    && v.shape().as_slice().last().copied() != Some(1)
            })
            .map(|(k, _)| k.clone())
            .collect();
        for k in conv1d_keys {
            let v = weights.get(&k).expect("key just collected").clone();
            // HF [out, in, k] → cxx-mlx [out, k, in] : axes permutation [0, 2, 1].
            let moved = mlx::ops::shape::transpose_axes(&v, &[0_i32, 2, 1][..])?;
            weights.insert(k, moved);
        }

        // 4. RMSNorm +1.0 shift if triggered.
        if should_shift_norm {
            const NORM_SUFFIXES: &[&str] = &[
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                ".q_norm.weight",
                ".k_norm.weight",
            ];
            const NORM_EXACT: &[&str] = &["model.norm.weight"];
            let keys_to_shift: Vec<String> = weights
                .iter()
                .filter(|(k, v)| {
                    v.shape().as_slice().len() == 1
                        && (NORM_SUFFIXES.iter().any(|s| k.ends_with(s))
                            || NORM_EXACT.iter().any(|s| k == s))
                })
                .map(|(k, _)| k.clone())
                .collect();
            for k in keys_to_shift {
                let v = weights.get(&k).expect("key just collected").clone();
                let shifted = &v + 1.0_f32;
                weights.insert(k, shifted);
            }
        }
        Ok(())
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

    fn empty_text_config() -> serde_json::Value {
        serde_json::json!({"text_config": {}})
    }
    fn tied_text_config() -> serde_json::Value {
        serde_json::json!({"text_config": {"tie_word_embeddings": true}})
    }

    #[test]
    fn sanitize_strips_mtp_keys_and_shifts_norm() {
        let mut w: HashMap<String, Array> = HashMap::new();
        // mtp.* presence triggers should_shift_norm
        let mtp_arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert(
            "mtp.layers.0.input_layernorm.weight".into(),
            mtp_arr.clone(),
        );
        // a main-model norm at a known suffix
        let norm_arr: Array = (&[0.5_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert(
            "model.layers.0.input_layernorm.weight".into(),
            norm_arr.clone(),
        );

        Loader::sanitize(&mut w, &empty_text_config()).unwrap();

        // mtp.* is gone.
        assert!(!w.contains_key("mtp.layers.0.input_layernorm.weight"));
        // main-model norm got +1.0 shift.
        let shifted = w.get("model.layers.0.input_layernorm.weight").unwrap();
        let v: Vec<f32> = shifted.to_vec().unwrap();
        for x in v {
            assert!((x - 1.5).abs() < 1e-6, "expected 1.5 (0.5+1.0), got {x}");
        }
    }

    #[test]
    fn sanitize_conv1d_moveaxis_when_3d_last_not_one() {
        let mut w: HashMap<String, Array> = HashMap::new();
        // shape [out=2, in=3, k=4] → after transpose_axes [0,2,1] → [2, 4, 3]
        let data: Vec<f32> = (0..(2 * 3 * 4)).map(|i| i as f32).collect();
        let arr: Array = (data.as_slice(), &[2_i32, 3, 4][..]).try_into().unwrap();
        w.insert("model.layers.0.linear_attn.conv1d.weight".into(), arr);

        Loader::sanitize(&mut w, &empty_text_config()).unwrap();

        let after = w.get("model.layers.0.linear_attn.conv1d.weight").unwrap();
        assert_eq!(after.shape().as_slice(), &[2, 4, 3]);
    }

    #[test]
    fn sanitize_strips_lm_head_when_tied() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let h: Array = (&[0.0_f32; 4][..], (2_i32, 2)).try_into().unwrap();
        w.insert("lm_head.weight".into(), h.clone());
        w.insert("lm_head.scales".into(), h.clone());
        w.insert("model.embed_tokens.weight".into(), h);

        Loader::sanitize(&mut w, &tied_text_config()).unwrap();

        assert!(!w.contains_key("lm_head.weight"));
        assert!(!w.contains_key("lm_head.scales"));
        // embed_tokens preserved.
        assert!(w.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    fn sanitize_no_norm_shift_when_neither_trigger() {
        let mut w: HashMap<String, Array> = HashMap::new();
        // No mtp.*, conv1d already in correct form.
        let conv: Array = (&[0.0_f32; 8][..], &[2_i32, 4, 1][..]).try_into().unwrap();
        w.insert("layers.0.linear_attn.conv1d.weight".into(), conv);
        let norm: Array = (&[0.5_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert("model.norm.weight".into(), norm);

        Loader::sanitize(&mut w, &empty_text_config()).unwrap();

        let n = w.get("model.norm.weight").unwrap();
        let v: Vec<f32> = n.to_vec().unwrap();
        for x in v {
            assert!((x - 0.5).abs() < 1e-6, "norm should stay at 0.5, got {x}");
        }
    }

    #[test]
    fn sanitize_drops_vision_tower_keys() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        // vision_tower.* should be dropped.
        w.insert("vision_tower.encoder.layers.0.weight".into(), arr.clone());
        w.insert("vision_tower.patch_embed.proj.weight".into(), arr.clone());
        // plain model.* key should be preserved.
        w.insert("model.embed_tokens.weight".into(), arr.clone());

        Loader::sanitize(&mut w, &empty_text_config()).unwrap();

        assert!(
            !w.contains_key("vision_tower.encoder.layers.0.weight"),
            "vision_tower key must be dropped"
        );
        assert!(
            !w.contains_key("vision_tower.patch_embed.proj.weight"),
            "vision_tower key must be dropped"
        );
        assert!(
            w.contains_key("model.embed_tokens.weight"),
            "plain model.* key must be preserved"
        );
    }

    #[test]
    fn sanitize_strips_language_model_prefix() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        // language_model. prefix should be stripped.
        w.insert(
            "language_model.model.embed_tokens.weight".into(),
            arr.clone(),
        );
        w.insert(
            "language_model.model.layers.0.self_attn.q_proj.weight".into(),
            arr.clone(),
        );
        // vision_tower mixed in should also be dropped.
        w.insert("vision_tower.foo.weight".into(), arr.clone());

        Loader::sanitize(&mut w, &empty_text_config()).unwrap();

        // prefix-stripped keys must exist.
        assert!(
            w.contains_key("model.embed_tokens.weight"),
            "language_model. prefix must be stripped"
        );
        assert!(
            w.contains_key("model.layers.0.self_attn.q_proj.weight"),
            "language_model. prefix must be stripped"
        );
        // original prefixed keys must not exist.
        assert!(
            !w.contains_key("language_model.model.embed_tokens.weight"),
            "original prefixed key must be removed"
        );
        // vision_tower must be dropped.
        assert!(
            !w.contains_key("vision_tower.foo.weight"),
            "vision_tower key must be dropped"
        );
    }
}
