//! Model loader — opens a directory containing `config.json` +
//! `tokenizer_config.json` + safetensors weights, exposes tensor lookup
//! by full key and parsed quantization metadata.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype};
use serde::Deserialize;

use crate::Result;

/// Quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Affine quantization (scale + offset per group).
    Affine,
    /// OptiQ mixed-bit quantization backed by MLX affine packed tensors.
    OptiQ,
    /// OCP microscaling 4-bit floating-point quantization.
    Mxfp4,
    /// OCP microscaling 8-bit floating-point quantization.
    Mxfp8,
}

impl QuantMode {
    pub(crate) fn mlx_mode(self) -> &'static str {
        match self {
            Self::Affine | Self::OptiQ => "affine",
            Self::Mxfp4 => "mxfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }

    pub(crate) fn uses_affine_storage(self) -> bool {
        matches!(self, Self::Affine | Self::OptiQ)
    }

    pub(crate) fn output_dtype(self, scales_dtype: Dtype, biases_dtype: Option<Dtype>) -> Dtype {
        match self {
            Self::Affine | Self::OptiQ => biases_dtype.unwrap_or(scales_dtype),
            Self::Mxfp4 | Self::Mxfp8 => Dtype::Bfloat16,
        }
    }
}

/// Quantization metadata parsed from `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantMeta {
    /// Group size for per-group quantization parameters.
    pub group_size: i32,
    /// Bits per quantized weight (2, 4, 8).
    pub bits: i32,
    /// Quantization scheme.
    pub mode: QuantMode,
}

impl QuantMeta {
    pub(crate) fn validate_storage(
        self,
        prefix: &str,
        weight: &Array,
        scales: &Array,
        biases: Option<&Array>,
    ) -> Result<()> {
        if matches!(self.mode, QuantMode::Mxfp4 | QuantMode::Mxfp8) {
            if weight.dtype() != Dtype::Uint32 {
                return Err(anyhow!(
                    "{prefix}: {} packed weight must have dtype uint32, got {:?}",
                    self.mode.mlx_mode(),
                    weight.dtype()
                ));
            }
            if scales.dtype() != Dtype::Uint8 {
                return Err(anyhow!(
                    "{prefix}: {} scales must have dtype uint8, got {:?}",
                    self.mode.mlx_mode(),
                    scales.dtype()
                ));
            }
            if biases.is_some() {
                return Err(anyhow!(
                    "{prefix}: {} storage must not contain affine quantization biases",
                    self.mode.mlx_mode()
                ));
            }
        }
        Ok(())
    }
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
    quant_overrides: HashMap<String, QuantMeta>,
    tokenizer_config: TokenizerConfig,
    config_raw: serde_json::Value,
    model_dir: std::path::PathBuf,
}

#[derive(Debug, Clone, Copy)]
enum SanitizeMode {
    Text { keep_vision_tower: bool },
    Mtp,
    Gemma4Drafter,
}

impl Loader {
    /// Open a directory containing `config.json`, `tokenizer_config.json`,
    /// and `model.safetensors` (single-file) or `model.safetensors.index.json`
    /// (sharded). All weights are mmap-loaded eagerly.
    ///
    /// `vision_tower.*` keys are **dropped** during sanitize — use
    /// [`Loader::open_multimodal`] when the vision encoder weights are needed.
    pub fn open(model_dir: &Path) -> Result<Self> {
        Self::open_impl(
            model_dir,
            SanitizeMode::Text {
                keep_vision_tower: false,
            },
        )
    }

    /// Like [`Loader::open`] but retains `vision_tower.*` keys so that the
    /// VisionTower can load its weights from the same Loader instance.
    /// Used for multimodal (VL) inference paths.
    pub fn open_multimodal(model_dir: &Path) -> Result<Self> {
        Self::open_impl(
            model_dir,
            SanitizeMode::Text {
                keep_vision_tower: true,
            },
        )
    }

    /// Open a standalone Qwen MTP draft-head checkpoint.
    ///
    /// These repositories are not full language models: their weights live at
    /// root paths such as `fc.weight`, `layers.0.*`, and `norm.weight`.
    pub fn open_mtp(model_dir: &Path) -> Result<Self> {
        Self::open_impl(model_dir, SanitizeMode::Mtp)
    }

    /// Open a standalone Gemma4 assistant/drafter checkpoint.
    ///
    /// Gemma4 assistant repositories are independent models that consume the
    /// base Gemma4 model's shared K/V states. They use root tensors such as
    /// `pre_projection.*`, `post_projection.*`, `masked_embedding.*`, and a
    /// small `model.*` drafter backbone.
    pub fn open_gemma4_drafter(model_dir: &Path) -> Result<Self> {
        Self::open_impl(model_dir, SanitizeMode::Gemma4Drafter)
    }

    fn open_impl(model_dir: &Path, sanitize_mode: SanitizeMode) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_raw: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("opening {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;

        let tok_path = model_dir.join("tokenizer_config.json");
        let mut tokenizer_config: TokenizerConfig = if tok_path.exists() {
            serde_json::from_reader(
                std::fs::File::open(&tok_path)
                    .with_context(|| format!("opening {}", tok_path.display()))?,
            )
            .with_context(|| format!("parsing {}", tok_path.display()))?
        } else {
            TokenizerConfig::default()
        };

        // Some HF checkpoints (e.g. Qwen3.5-4B-MLX-4bit) store the chat
        // template in a standalone `chat_template.jinja` instead of inlining
        // it in `tokenizer_config.json`. Fall back to that file when the JSON
        // field is absent.
        if tokenizer_config.chat_template.is_none() {
            let jinja_path = model_dir.join("chat_template.jinja");
            if jinja_path.exists() {
                let tmpl = std::fs::read_to_string(&jinja_path)
                    .with_context(|| format!("reading {}", jinja_path.display()))?;
                tokenizer_config.chat_template = Some(tmpl);
            }
        }

        let optiq_metadata = load_optiq_metadata(model_dir)?;
        let quant = parse_quant_meta_with_optiq(&config_raw, optiq_metadata.as_ref())?;
        let quant_overrides = match quant {
            Some(global) => {
                parse_quant_overrides_with_optiq(&config_raw, global, optiq_metadata.as_ref())?
            }
            None => HashMap::new(),
        };

        let mut tensors = load_safetensors(model_dir)?;
        if matches!(
            sanitize_mode,
            SanitizeMode::Text {
                keep_vision_tower: true
            }
        ) {
            load_optiq_vision_sidecar(model_dir, &config_raw, &mut tensors)?;
        }

        match sanitize_mode {
            SanitizeMode::Text { keep_vision_tower } => {
                Self::sanitize(&mut tensors, &config_raw, keep_vision_tower)?;
            }
            SanitizeMode::Mtp => Self::sanitize_mtp(&mut tensors, &config_raw)?,
            SanitizeMode::Gemma4Drafter => {
                Self::sanitize_gemma4_drafter(&mut tensors, &config_raw)?
            }
        }

        // Eagerly evaluate all tensors on the loading thread so that no lazy
        // stream-tagged computation remains in the weight arrays.  This
        // eliminates a thread-safety hazard: if any sanitize step produced a
        // lazy result (e.g. transpose_axes, +1.0 norm-shift), its primitive is
        // tagged with the current thread's default MLX stream.  That stream's
        // CommandEncoder lives in *this* thread's thread_local map.  A later
        // inference call on a different thread (e.g. tokio blocking-pool)
        // would find no encoder for that stream index and panic with
        // "There is no Stream(gpu, N) in current thread."
        //
        // After eval() all tensors are plain data buffers — subsequent threads
        // can read them without any stream dependency.
        {
            let refs: Vec<&Array> = tensors.values().collect();
            mlx::transforms::eval(&refs).context("Loader::open: eager eval of weights")?;
        }

        Ok(Self {
            tensors,
            quant,
            quant_overrides,
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

    /// Total storage bytes of the sanitized, eagerly loaded tensors.
    pub(crate) fn loaded_tensor_bytes(&self) -> usize {
        tensor_storage_bytes(&self.tensors)
    }

    /// Quantization metadata, or None if model is not quantized.
    pub fn quant_meta(&self) -> Option<QuantMeta> {
        self.quant
    }

    /// Quantization metadata for a tensor prefix. Per-prefix overrides in
    /// `config.json` take precedence over the global quantization metadata.
    pub fn quant_meta_for(&self, prefix: &str) -> Option<QuantMeta> {
        self.quant_overrides.get(prefix).copied().or(self.quant)
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

    /// HF checkpoint sanitize aligned with model-specific text-only loading.
    ///
    /// Mutates `weights` in place:
    /// 0. Drop non-text tower keys when the caller requests text-only loading.
    ///    Strip `language_model.` prefix from all remaining keys so that downstream
    ///    code can use plain `model.*` paths (e.g. `model.embed_tokens.weight`).
    ///    Also drops `vit_merger.*` and `merger.*` (MiniCPM-V-4.6 resampler weights,
    ///    not needed for text-only inference).
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
        keep_vision_tower: bool,
    ) -> Result<()> {
        let is_qwen35 = is_qwen35_offset_gamma_model(config_raw);

        // 0. Drop non-text tower/embedder keys unless caller explicitly
        //    requests the vision path. Audio is not supported by any ironmlx
        //    path yet, so it is always discarded before conv/norm detection.
        if keep_vision_tower {
            weights.retain(|k, _| !k.starts_with("audio_tower.") && !k.starts_with("embed_audio."));
        } else {
            weights.retain(|k, _| {
                !k.starts_with("vision_tower.")
                    && !k.starts_with("vision_embedder.")
                    && !k.starts_with("audio_tower.")
                    && !k.starts_with("embed_vision.")
                    && !k.starts_with("embed_audio.")
                    && !k.starts_with("model.encoder.vision_tower.")
                    && !k.starts_with("model.encoder.embed_vision.")
                    && !k.starts_with("vit_merger.")
                    && !k.starts_with("merger.")
            });
        }

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
        let should_shift_norm = is_qwen35 && (has_mtp || has_unsanitized_conv1d);

        // 1. Strip mtp.*
        weights.retain(|k, _| !k.contains("mtp."));

        // 2. Strip lm_head if tied.
        let tie = config_raw
            .get("text_config")
            .and_then(|tc| tc.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .or_else(|| {
                config_raw
                    .get("tie_word_embeddings")
                    .and_then(|v| v.as_bool())
            })
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
            shift_offset_gamma_norms(weights, &["model.norm.weight"])?;
        }
        Ok(())
    }

    /// Sanitize a standalone Qwen MTP draft-head checkpoint.
    ///
    /// New mlx-community MTP repos store the MTP head at root paths, not under
    /// `mtp.*`; these root keys must be retained. Qwen's offset-gamma RmsNorm
    /// convention still applies to every MTP norm tensor.
    fn sanitize_mtp(
        weights: &mut HashMap<String, Array>,
        config_raw: &serde_json::Value,
    ) -> Result<()> {
        let model_type = config_raw
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("MTP config.json missing model_type"))?;
        if model_type != "qwen3_5_mtp" {
            return Err(anyhow!(
                "Loader::open_mtp expected model_type=qwen3_5_mtp, got {model_type}"
            ));
        }

        weights.retain(|k, _| {
            !k.starts_with("vision_tower.")
                && !k.starts_with("audio_tower.")
                && !k.starts_with("embed_vision.")
                && !k.starts_with("embed_audio.")
                && !k.starts_with("vit_merger.")
                && !k.starts_with("merger.")
        });

        shift_offset_gamma_norms(
            weights,
            &[
                "pre_fc_norm_hidden.weight",
                "pre_fc_norm_embedding.weight",
                "norm.weight",
            ],
        )
    }

    /// Sanitize a standalone Gemma4 assistant/drafter checkpoint.
    fn sanitize_gemma4_drafter(
        weights: &mut HashMap<String, Array>,
        config_raw: &serde_json::Value,
    ) -> Result<()> {
        let model_type = config_raw
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Gemma4 drafter config.json missing model_type"))?;
        if !matches!(model_type, "gemma4_assistant" | "gemma4_unified_assistant") {
            return Err(anyhow!(
                "Loader::open_gemma4_drafter expected model_type=gemma4_assistant or gemma4_unified_assistant, got {model_type}"
            ));
        }

        weights.retain(|k, _| {
            !k.starts_with("vision_tower.")
                && !k.starts_with("vision_embedder.")
                && !k.starts_with("audio_tower.")
                && !k.starts_with("embed_vision.")
                && !k.starts_with("embed_audio.")
                && !k.starts_with("vit_merger.")
                && !k.starts_with("merger.")
        });

        let tie = config_raw
            .get("text_config")
            .and_then(|tc| tc.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .or_else(|| {
                config_raw
                    .get("tie_word_embeddings")
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false);
        if tie {
            weights.remove("lm_head.weight");
            weights.remove("lm_head.scales");
            weights.remove("lm_head.biases");
        }

        if let Some(ordering) = weights.remove("masked_embedding.token_ordering") {
            let ordering = mlx::ops::cast::astype(&ordering, Dtype::Int32)?;
            weights.insert("masked_embedding.token_ordering".to_owned(), ordering);
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

fn tensor_storage_bytes(tensors: &HashMap<String, Array>) -> usize {
    tensors.values().fold(0usize, |total, tensor| {
        total.saturating_add(tensor.size().saturating_mul(tensor.dtype().byte_size()))
    })
}

fn quant_config_value(config_raw: &serde_json::Value) -> Option<&serde_json::Value> {
    config_raw
        .get("quantization")
        .or_else(|| config_raw.get("quantization_config"))
}

#[cfg(test)]
fn parse_quant_meta(config_raw: &serde_json::Value) -> Result<Option<QuantMeta>> {
    parse_quant_meta_with_optiq(config_raw, None)
}

fn parse_quant_meta_with_optiq(
    config_raw: &serde_json::Value,
    optiq_metadata: Option<&serde_json::Value>,
) -> Result<Option<QuantMeta>> {
    if let Some(metadata) = optiq_metadata {
        validate_optiq_metadata(metadata)?;
    }
    let Some(q) = quant_config_value(config_raw) else {
        if optiq_metadata.is_some() {
            return Err(anyhow!(
                "optiq_metadata.json present but config.json has no quantization metadata"
            ));
        }
        return Ok(None);
    };
    let force_mode = optiq_metadata.map(|_| QuantMode::OptiQ);
    Ok(Some(parse_quant_meta_value_with_mode(
        q,
        None,
        force_mode,
        "quantization",
    )?))
}

#[cfg(test)]
fn parse_quant_overrides(
    config_raw: &serde_json::Value,
    global: QuantMeta,
) -> Result<HashMap<String, QuantMeta>> {
    parse_quant_overrides_with_optiq(config_raw, global, None)
}

fn parse_quant_overrides_with_optiq(
    config_raw: &serde_json::Value,
    global: QuantMeta,
    optiq_metadata: Option<&serde_json::Value>,
) -> Result<HashMap<String, QuantMeta>> {
    let Some(q) = quant_config_value(config_raw) else {
        return Ok(HashMap::new());
    };
    let q_obj = q
        .as_object()
        .ok_or_else(|| anyhow!("quantization must be a JSON object"))?;
    let mut overrides = HashMap::new();

    for (key, value) in q_obj {
        if value.as_object().is_none()
            || (value.get("bits").is_none() && value.get("group_size").is_none())
        {
            continue;
        }
        let prefix = normalize_quant_prefix(key);
        let force_mode = (global.mode == QuantMode::OptiQ).then_some(QuantMode::OptiQ);
        let meta = parse_quant_meta_value_with_mode(
            value,
            Some(global.mode),
            force_mode,
            &format!("quantization.{key}"),
        )?;
        overrides.insert(prefix, meta);
    }

    if let Some(metadata) = optiq_metadata {
        let per_layer = optiq_per_layer(metadata)?;
        for (key, value) in per_layer {
            let prefix = normalize_quant_prefix(key);
            let meta = parse_quant_meta_value_with_mode(
                value,
                Some(global.mode),
                Some(QuantMode::OptiQ),
                &format!("optiq_metadata.per_layer.{key}"),
            )?;
            if let Some(existing) = overrides.get(&prefix) {
                if *existing != meta {
                    return Err(anyhow!(
                        "quantization.{prefix} conflicts with optiq_metadata.per_layer.{key}: config={existing:?}, optiq_metadata={meta:?}"
                    ));
                }
            } else {
                overrides.insert(prefix, meta);
            }
        }
    }

    Ok(overrides)
}

fn parse_quant_meta_value_with_mode(
    q: &serde_json::Value,
    default_mode: Option<QuantMode>,
    force_mode: Option<QuantMode>,
    context_name: &str,
) -> Result<QuantMeta> {
    let group_size_i64 = q
        .get("group_size")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("{context_name}.group_size missing or non-int"))?;
    let group_size = i32::try_from(group_size_i64)
        .with_context(|| format!("{context_name}.group_size out of i32 range"))?;
    let bits_i64 = q
        .get("bits")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("{context_name}.bits missing or non-int"))?;
    let bits =
        i32::try_from(bits_i64).with_context(|| format!("{context_name}.bits out of i32 range"))?;
    let mode = match force_mode {
        Some(QuantMode::OptiQ) => match q.get("mode").and_then(|m| m.as_str()) {
            Some("affine") | None => QuantMode::OptiQ,
            Some(other) => {
                return Err(anyhow!(
                    "unsupported {context_name}.mode `{other}` for OptiQ affine storage"
                ));
            }
        },
        Some(mode) => mode,
        None => match q.get("mode").and_then(|m| m.as_str()) {
            Some("affine") => QuantMode::Affine,
            Some("mxfp4") => QuantMode::Mxfp4,
            Some("mxfp8") => QuantMode::Mxfp8,
            Some(other) => return Err(anyhow!("unsupported {context_name}.mode `{other}`")),
            None => default_mode.unwrap_or(QuantMode::Affine),
        },
    };
    validate_quant_meta_contract(mode, bits, group_size, context_name)?;
    Ok(QuantMeta {
        group_size,
        bits,
        mode,
    })
}

fn validate_quant_meta_contract(
    mode: QuantMode,
    bits: i32,
    group_size: i32,
    context_name: &str,
) -> Result<()> {
    match mode {
        QuantMode::Affine | QuantMode::OptiQ if matches!(bits, 2 | 4 | 8) => Ok(()),
        QuantMode::Affine | QuantMode::OptiQ => Err(anyhow!(
            "unsupported {context_name}.bits `{bits}` for {mode:?} quantization; supported bits are 2, 4, and 8"
        )),
        QuantMode::Mxfp4 if bits == 4 && group_size == 32 => Ok(()),
        QuantMode::Mxfp4 => Err(anyhow!(
            "{context_name}.mode `mxfp4` requires bits=4 and group_size=32, got bits={bits} and group_size={group_size}"
        )),
        QuantMode::Mxfp8 if bits == 8 && group_size == 32 => Ok(()),
        QuantMode::Mxfp8 => Err(anyhow!(
            "{context_name}.mode `mxfp8` requires bits=8 and group_size=32, got bits={bits} and group_size={group_size}"
        )),
    }
}

fn load_optiq_metadata(model_dir: &Path) -> Result<Option<serde_json::Value>> {
    let path = model_dir.join("optiq_metadata.json");
    if !path.exists() {
        return Ok(None);
    }
    let metadata: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    validate_optiq_metadata(&metadata)?;
    Ok(Some(metadata))
}

fn validate_optiq_metadata(metadata: &serde_json::Value) -> Result<()> {
    let method = metadata
        .get("method")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("optiq_metadata.method missing or non-string"))?;
    if method != "optiq_mixed_precision" {
        return Err(anyhow!("unsupported optiq_metadata.method `{method}`"));
    }
    optiq_per_layer(metadata)?;
    Ok(())
}

fn optiq_per_layer(
    metadata: &serde_json::Value,
) -> Result<&serde_json::Map<String, serde_json::Value>> {
    metadata
        .get("per_layer")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("optiq_metadata.per_layer missing or non-object"))
}

fn load_optiq_vision_sidecar(
    model_dir: &Path,
    config_raw: &serde_json::Value,
    tensors: &mut HashMap<String, Array>,
) -> Result<()> {
    let Some(sidecar) = config_raw
        .get("optiq_vision")
        .and_then(|v| v.get("sidecar"))
        .and_then(|v| v.as_str())
    else {
        return Ok(());
    };
    let sidecar_path = model_dir.join(sidecar);
    let (sidecar_tensors, _meta) = mlx::io::load_safetensors(
        sidecar_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 path: {}", sidecar_path.display()))?,
    )
    .map_err(|e| anyhow!("load_safetensors {}: {e}", sidecar_path.display()))?;
    for (key, tensor) in sidecar_tensors {
        if tensors.insert(key.clone(), tensor).is_some() {
            return Err(anyhow!(
                "optiq vision sidecar `{sidecar}` duplicates tensor key `{key}`"
            ));
        }
    }
    Ok(())
}

fn normalize_quant_prefix(key: &str) -> String {
    key.strip_prefix("language_model.")
        .unwrap_or(key)
        .to_owned()
}

fn shift_offset_gamma_norms(weights: &mut HashMap<String, Array>, exact: &[&str]) -> Result<()> {
    const NORM_SUFFIXES: &[&str] = &[
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];
    let keys_to_shift: Vec<String> = weights
        .iter()
        .filter(|(k, v)| {
            v.shape().as_slice().len() == 1
                && (NORM_SUFFIXES.iter().any(|s| k.ends_with(s)) || exact.iter().any(|s| k == s))
        })
        .map(|(k, _)| k.clone())
        .collect();
    for k in keys_to_shift {
        let v = weights.get(&k).expect("key just collected").clone();
        let shifted = &v + 1.0_f32;
        weights.insert(k, shifted);
    }
    Ok(())
}

fn is_qwen35_offset_gamma_model(config_raw: &serde_json::Value) -> bool {
    let top = config_raw.get("model_type").and_then(|v| v.as_str());
    let text = config_raw
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(|v| v.as_str());
    top.is_some_and(|m| m == "qwen3_5" || m == "qwen3_5_moe")
        || text.is_some_and(|m| m == "qwen3_5_text" || m == "qwen3_5_moe_text")
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
    use mlx::Array;
    use serde_json::json;
    use serial_test::serial;
    use std::collections::HashMap;

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
    fn parse_quant_meta_affine_2bit() {
        let cfg = json!({
            "quantization": { "group_size": 64, "bits": 2, "mode": "affine" }
        });
        let q = parse_quant_meta(&cfg).unwrap().expect("quant");
        assert_eq!(q.group_size, 64);
        assert_eq!(q.bits, 2);
        assert_eq!(q.mode, QuantMode::Affine);
    }

    #[test]
    fn parse_quant_meta_mxfp4_exact_contract() {
        let cfg = json!({
            "quantization": { "group_size": 32, "bits": 4, "mode": "mxfp4" }
        });
        let q = parse_quant_meta(&cfg).unwrap().expect("quant");
        assert_eq!(q.group_size, 32);
        assert_eq!(q.bits, 4);
        assert_eq!(q.mode, QuantMode::Mxfp4);
        assert_eq!(q.mode.mlx_mode(), "mxfp4");
        assert!(!q.mode.uses_affine_storage());
        assert_eq!(q.mode.output_dtype(Dtype::Uint8, None), Dtype::Bfloat16);
    }

    #[test]
    fn parse_quant_meta_mxfp8_exact_contract() {
        let cfg = json!({
            "quantization_config": { "group_size": 32, "bits": 8, "mode": "mxfp8" }
        });
        let q = parse_quant_meta(&cfg).unwrap().expect("quant");
        assert_eq!(q.group_size, 32);
        assert_eq!(q.bits, 8);
        assert_eq!(q.mode, QuantMode::Mxfp8);
        assert_eq!(q.mode.mlx_mode(), "mxfp8");
        assert!(!q.mode.uses_affine_storage());
        assert_eq!(q.mode.output_dtype(Dtype::Uint8, None), Dtype::Bfloat16);
    }

    #[test]
    fn parse_quant_overrides_inherit_mxfp_mode() {
        let cfg = json!({
            "quantization": {
                "group_size": 32,
                "bits": 4,
                "mode": "mxfp4",
                "language_model.model.layers.0.mlp.down_proj": {
                    "group_size": 32,
                    "bits": 4
                }
            }
        });
        let global = parse_quant_meta(&cfg).unwrap().expect("global quant");
        let overrides = parse_quant_overrides(&cfg, global).unwrap();
        assert_eq!(
            overrides["model.layers.0.mlp.down_proj"].mode,
            QuantMode::Mxfp4
        );
    }

    #[test]
    fn parse_quant_meta_rejects_invalid_mxfp_parameters() {
        for (mode, bits, group_size) in [
            ("mxfp4", 8, 32),
            ("mxfp4", 4, 64),
            ("mxfp8", 4, 32),
            ("mxfp8", 8, 64),
        ] {
            let cfg = json!({
                "quantization": {
                    "group_size": group_size,
                    "bits": bits,
                    "mode": mode
                }
            });
            let err = parse_quant_meta(&cfg).expect_err("invalid MXFP contract must fail");
            let message = err.to_string();
            assert!(message.contains(mode), "unexpected error: {message}");
            assert!(message.contains("requires"), "unexpected error: {message}");
        }
    }

    #[test]
    fn mxfp_storage_requires_uint8_scales_without_quant_biases() {
        for mode in [QuantMode::Mxfp4, QuantMode::Mxfp8] {
            let bits = if mode == QuantMode::Mxfp4 { 4 } else { 8 };
            let meta = QuantMeta {
                group_size: 32,
                bits,
                mode,
            };
            let weight = Array::zeros((2, 2), Dtype::Uint32).unwrap();
            let scales = Array::zeros((2, 2), Dtype::Uint8).unwrap();
            meta.validate_storage("model.layers.0.mlp.down_proj", &weight, &scales, None)
                .expect("valid MXFP storage");

            let byte_weight = Array::zeros((2, 2), Dtype::Uint8).unwrap();
            let err = meta
                .validate_storage("model.layers.0.mlp.down_proj", &byte_weight, &scales, None)
                .expect_err("MXFP byte weights must fail");
            assert!(err.to_string().contains("uint32"));

            let float_scales = Array::zeros((2, 2), Dtype::Bfloat16).unwrap();
            let err = meta
                .validate_storage("model.layers.0.mlp.down_proj", &weight, &float_scales, None)
                .expect_err("MXFP float scales must fail");
            assert!(err.to_string().contains("uint8"));

            let biases = Array::zeros((2, 2), Dtype::Uint8).unwrap();
            let err = meta
                .validate_storage(
                    "model.layers.0.mlp.down_proj",
                    &weight,
                    &scales,
                    Some(&biases),
                )
                .expect_err("MXFP quant biases must fail");
            assert!(err.to_string().contains("must not contain"));
        }
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
    fn parse_quant_overrides_normalizes_language_model_prefix() {
        let cfg = json!({
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        });
        let global = parse_quant_meta(&cfg).unwrap().expect("global quant");
        let overrides = parse_quant_overrides(&cfg, global).unwrap();

        assert_eq!(
            overrides["model.layers.0.mlp.gate"],
            QuantMeta {
                group_size: 64,
                bits: 8,
                mode: QuantMode::Affine,
            }
        );
    }

    #[test]
    fn parse_quant_overrides_preserve_mixed_affine_bits() {
        let cfg = json!({
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "model.layers.0.mlp.down_proj": {
                    "group_size": 64,
                    "bits": 2
                },
                "model.layers.1.mlp.down_proj": {
                    "group_size": 128,
                    "bits": 8
                }
            }
        });
        let global = parse_quant_meta(&cfg).unwrap().expect("global quant");
        let overrides = parse_quant_overrides(&cfg, global).unwrap();

        assert_eq!(
            overrides["model.layers.0.mlp.down_proj"],
            QuantMeta {
                group_size: 64,
                bits: 2,
                mode: QuantMode::Affine,
            }
        );
        assert_eq!(
            overrides["model.layers.1.mlp.down_proj"],
            QuantMeta {
                group_size: 128,
                bits: 8,
                mode: QuantMode::Affine,
            }
        );
    }

    #[test]
    fn parse_quant_meta_with_optiq_metadata_marks_independent_mode() {
        let cfg = json!({
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.mlp.down_proj": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        });
        let optiq = json!({
            "method": "optiq_mixed_precision",
            "per_layer": {
                "language_model.model.layers.0.mlp.down_proj": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        });

        let global = parse_quant_meta_with_optiq(&cfg, Some(&optiq))
            .unwrap()
            .expect("global quant");
        let overrides = parse_quant_overrides_with_optiq(&cfg, global, Some(&optiq)).unwrap();

        assert_eq!(
            global,
            QuantMeta {
                group_size: 64,
                bits: 4,
                mode: QuantMode::OptiQ,
            }
        );
        assert_eq!(
            overrides["model.layers.0.mlp.down_proj"],
            QuantMeta {
                group_size: 64,
                bits: 8,
                mode: QuantMode::OptiQ,
            }
        );
    }

    #[test]
    fn parse_quant_overrides_accepts_optiq_per_layer_not_duplicated_in_config() {
        let cfg = json!({
            "quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
        });
        let optiq = json!({
            "method": "optiq_mixed_precision",
            "per_layer": {
                "language_model.model.layers.3.self_attn.q_proj": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        });
        let global = parse_quant_meta_with_optiq(&cfg, Some(&optiq))
            .unwrap()
            .expect("global quant");
        let overrides = parse_quant_overrides_with_optiq(&cfg, global, Some(&optiq)).unwrap();

        assert_eq!(
            overrides["model.layers.3.self_attn.q_proj"],
            QuantMeta {
                group_size: 64,
                bits: 8,
                mode: QuantMode::OptiQ,
            }
        );
    }

    #[test]
    fn parse_quant_overrides_rejects_conflicting_optiq_per_layer() {
        let cfg = json!({
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "model.layers.3.self_attn.q_proj": {
                    "group_size": 64,
                    "bits": 4
                }
            }
        });
        let optiq = json!({
            "method": "optiq_mixed_precision",
            "per_layer": {
                "model.layers.3.self_attn.q_proj": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        });
        let global = parse_quant_meta_with_optiq(&cfg, Some(&optiq))
            .unwrap()
            .expect("global quant");
        let err = parse_quant_overrides_with_optiq(&cfg, global, Some(&optiq))
            .expect_err("conflicting OptiQ metadata should fail");
        assert!(err.to_string().contains("conflicts with optiq_metadata"));
    }

    #[test]
    fn parse_quant_meta_rejects_unknown_optiq_method() {
        let cfg = json!({
            "quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
        });
        let optiq = json!({
            "method": "not_optiq",
            "per_layer": {}
        });
        let err = parse_quant_meta_with_optiq(&cfg, Some(&optiq))
            .expect_err("unknown optiq method should fail");
        assert!(err
            .to_string()
            .contains("unsupported optiq_metadata.method"));
    }

    #[test]
    #[serial(mlx_metal)]
    fn open_multimodal_loads_optiq_vision_sidecar_but_text_open_does_not() {
        let dir =
            std::env::temp_dir().join(format!("ironmlx-optiq-sidecar-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            json!({
                "model_type": "gemma4",
                "optiq_vision": {
                    "sidecar": "optiq_vision.safetensors",
                    "dtype": "bfloat16",
                    "n_tensors": 1
                }
            })
            .to_string(),
        )
        .unwrap();

        let main_tensor: Array = (&[1.0_f32][..], (1_i32,)).try_into().unwrap();
        let vision_tensor: Array = (&[2.0_f32][..], (1_i32,)).try_into().unwrap();
        let mut main = HashMap::new();
        main.insert(
            "language_model.model.embed_tokens.weight".to_owned(),
            main_tensor,
        );
        let mut sidecar = HashMap::new();
        sidecar.insert("vision_tower.patch_embed.weight".to_owned(), vision_tensor);
        let metadata = HashMap::new();
        mlx::io::save_safetensors(
            dir.join("model.safetensors").to_str().unwrap(),
            &main,
            &metadata,
        )
        .unwrap();
        mlx::io::save_safetensors(
            dir.join("optiq_vision.safetensors").to_str().unwrap(),
            &sidecar,
            &metadata,
        )
        .unwrap();

        let text_loader = Loader::open(&dir).unwrap();
        assert!(text_loader.contains("model.embed_tokens.weight"));
        assert!(!text_loader.contains("vision_tower.patch_embed.weight"));

        let multimodal_loader = Loader::open_multimodal(&dir).unwrap();
        assert!(multimodal_loader.contains("model.embed_tokens.weight"));
        assert!(multimodal_loader.contains("vision_tower.patch_embed.weight"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn quant_meta_for_prefers_override_then_falls_back_to_global() {
        let global = QuantMeta {
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        };
        let override_meta = QuantMeta {
            group_size: 64,
            bits: 8,
            mode: QuantMode::Affine,
        };
        let mut quant_overrides = HashMap::new();
        quant_overrides.insert("model.layers.0.mlp.gate".to_owned(), override_meta);
        let loader = Loader {
            tensors: HashMap::new(),
            quant: Some(global),
            quant_overrides,
            tokenizer_config: TokenizerConfig::default(),
            config_raw: json!({}),
            model_dir: std::path::PathBuf::new(),
        };

        assert_eq!(
            loader.quant_meta_for("model.layers.0.mlp.gate"),
            Some(override_meta)
        );
        assert_eq!(
            loader.quant_meta_for("model.layers.0.self_attn.q_proj"),
            Some(global)
        );
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
    fn parse_quant_meta_rejects_affine_6bit_for_now() {
        let cfg = json!({
            "quantization": { "group_size": 64, "bits": 6, "mode": "affine" }
        });
        let err = parse_quant_meta(&cfg).expect_err("6-bit affine should be rejected");
        assert!(
            err.to_string().contains("unsupported quantization.bits"),
            "{err:#}"
        );
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
    fn qwen35_text_config() -> serde_json::Value {
        serde_json::json!({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}})
    }
    fn gemma4_text_config() -> serde_json::Value {
        serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}})
    }
    fn gemma4_unified_text_config() -> serde_json::Value {
        serde_json::json!({"model_type": "gemma4_unified", "text_config": {"model_type": "gemma4_unified_text"}})
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

        Loader::sanitize(&mut w, &qwen35_text_config(), false).unwrap();

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
    fn sanitize_mtp_root_weights_preserves_root_keys_and_shifts_norms() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let norm: Array = (&[0.25_f32; 4][..], (4_i32,)).try_into().unwrap();
        for key in [
            "pre_fc_norm_hidden.weight",
            "pre_fc_norm_embedding.weight",
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
        ] {
            w.insert(key.to_owned(), norm.clone());
        }
        let fc: Array = (&[2.0_f32; 8][..], &[2_i32, 4][..]).try_into().unwrap();
        w.insert("fc.weight".to_owned(), fc);

        Loader::sanitize_mtp(&mut w, &serde_json::json!({"model_type": "qwen3_5_mtp"})).unwrap();

        assert!(w.contains_key("fc.weight"));
        for key in [
            "pre_fc_norm_hidden.weight",
            "pre_fc_norm_embedding.weight",
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
        ] {
            let shifted = w.get(key).unwrap();
            let values: Vec<f32> = shifted.to_vec().unwrap();
            for x in values {
                assert!((x - 1.25).abs() < 1e-6, "{key} should be shifted, got {x}");
            }
        }
    }

    #[test]
    fn sanitize_gemma4_drafter_preserves_assistant_roots_and_casts_ordering() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let weight: Array = (&[1.0_f32; 8][..], &[2_i32, 4][..]).try_into().unwrap();
        let ordering: Array = (&[3_i64, 1, 2][..], &[3_i32][..]).try_into().unwrap();
        w.insert("pre_projection.weight".to_owned(), weight.clone());
        w.insert("post_projection.weight".to_owned(), weight.clone());
        w.insert("model.embed_tokens.weight".to_owned(), weight.clone());
        w.insert(
            "masked_embedding.embedding.weight".to_owned(),
            weight.clone(),
        );
        w.insert("masked_embedding.token_ordering".to_owned(), ordering);
        w.insert("lm_head.weight".to_owned(), weight.clone());
        w.insert(
            "vision_tower.patch_embedder.input_proj.weight".to_owned(),
            weight,
        );

        Loader::sanitize_gemma4_drafter(
            &mut w,
            &serde_json::json!({
                "model_type": "gemma4_assistant",
                "tie_word_embeddings": true,
                "text_config": {"tie_word_embeddings": true}
            }),
        )
        .unwrap();

        assert!(w.contains_key("pre_projection.weight"));
        assert!(w.contains_key("post_projection.weight"));
        assert!(w.contains_key("model.embed_tokens.weight"));
        assert!(w.contains_key("masked_embedding.embedding.weight"));
        assert!(!w.contains_key("lm_head.weight"));
        assert!(!w.contains_key("vision_tower.patch_embedder.input_proj.weight"));
        assert_eq!(
            w.get("masked_embedding.token_ordering").unwrap().dtype(),
            mlx::Dtype::Int32
        );
    }

    #[test]
    fn sanitize_gemma4_drafter_rejects_non_assistant_model_type() {
        let mut w: HashMap<String, Array> = HashMap::new();

        let err =
            Loader::sanitize_gemma4_drafter(&mut w, &serde_json::json!({"model_type": "gemma4"}))
                .expect_err("base Gemma4 is not a drafter checkpoint");

        assert!(err.to_string().contains("gemma4_assistant"));
    }

    #[test]
    fn sanitize_conv1d_moveaxis_when_3d_last_not_one() {
        let mut w: HashMap<String, Array> = HashMap::new();
        // shape [out=2, in=3, k=4] → after transpose_axes [0,2,1] → [2, 4, 3]
        let data: Vec<f32> = (0..(2 * 3 * 4)).map(|i| i as f32).collect();
        let arr: Array = (data.as_slice(), &[2_i32, 3, 4][..]).try_into().unwrap();
        w.insert("model.layers.0.linear_attn.conv1d.weight".into(), arr);

        Loader::sanitize(&mut w, &empty_text_config(), false).unwrap();

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

        Loader::sanitize(&mut w, &tied_text_config(), false).unwrap();

        assert!(!w.contains_key("lm_head.weight"));
        assert!(!w.contains_key("lm_head.scales"));
        // embed_tokens preserved.
        assert!(w.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    fn tensor_storage_bytes_sums_real_loaded_tensor_buffers() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let f32_arr = Array::zeros((2_i32, 3), mlx::Dtype::Float32).unwrap();
        let bf16_arr = Array::zeros((5_i32,), mlx::Dtype::Bfloat16).unwrap();
        let u8_arr = Array::zeros((7_i32,), mlx::Dtype::Uint8).unwrap();
        w.insert("model.f32.weight".into(), f32_arr);
        w.insert("model.bf16.weight".into(), bf16_arr);
        w.insert("model.u8.weight".into(), u8_arr);

        assert_eq!(super::tensor_storage_bytes(&w), 6 * 4 + 5 * 2 + 7);
    }

    #[test]
    fn sanitize_no_norm_shift_when_neither_trigger() {
        let mut w: HashMap<String, Array> = HashMap::new();
        // No mtp.*, conv1d already in correct form.
        let conv: Array = (&[0.0_f32; 8][..], &[2_i32, 4, 1][..]).try_into().unwrap();
        w.insert("layers.0.linear_attn.conv1d.weight".into(), conv);
        let norm: Array = (&[0.5_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert("model.norm.weight".into(), norm);

        Loader::sanitize(&mut w, &empty_text_config(), false).unwrap();

        let n = w.get("model.norm.weight").unwrap();
        let v: Vec<f32> = n.to_vec().unwrap();
        for x in v {
            assert!((x - 0.5).abs() < 1e-6, "norm should stay at 0.5, got {x}");
        }
    }

    #[test]
    fn sanitize_gemma4_audio_conv_does_not_shift_text_norm() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let conv: Array = (&[0.0_f32; 2 * 3 * 5][..], &[2_i32, 3, 5][..])
            .try_into()
            .unwrap();
        w.insert("audio_tower.depthwise_conv1d.weight".into(), conv);
        let norm: Array = (&[0.5_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert("language_model.model.norm.weight".into(), norm);

        Loader::sanitize(&mut w, &gemma4_text_config(), false).unwrap();

        assert!(
            !w.contains_key("audio_tower.depthwise_conv1d.weight"),
            "text-only Gemma4 load must drop audio tower keys before conv handling"
        );
        let n = w.get("model.norm.weight").unwrap();
        let v: Vec<f32> = n.to_vec().unwrap();
        for x in v {
            assert!(
                (x - 0.5).abs() < 1e-6,
                "Gemma4 norm should stay at 0.5, got {x}"
            );
        }
    }

    #[test]
    fn sanitize_drops_vision_tower_keys() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        // vision_tower.* should be dropped when keep_vision_tower=false.
        w.insert("vision_tower.encoder.layers.0.weight".into(), arr.clone());
        w.insert("vision_tower.patch_embed.proj.weight".into(), arr.clone());
        // plain model.* key should be preserved.
        w.insert("model.embed_tokens.weight".into(), arr.clone());

        Loader::sanitize(&mut w, &empty_text_config(), false).unwrap();

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
    fn sanitize_drops_diffusion_gemma_encoder_vision_keys_for_text_only() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert(
            "model.encoder.vision_tower.encoder.layers.0.weight".into(),
            arr.clone(),
        );
        w.insert(
            "model.encoder.embed_vision.embedding_projection.weight".into(),
            arr.clone(),
        );
        w.insert(
            "model.encoder.language_model.layers.0.layer_scalar".into(),
            arr.clone(),
        );
        w.insert("model.decoder.embed_tokens.weight".into(), arr.clone());

        Loader::sanitize(&mut w, &empty_text_config(), false).unwrap();

        assert!(!w.contains_key("model.encoder.vision_tower.encoder.layers.0.weight"));
        assert!(!w.contains_key("model.encoder.embed_vision.embedding_projection.weight"));
        assert!(w.contains_key("model.encoder.language_model.layers.0.layer_scalar"));
        assert!(w.contains_key("model.decoder.embed_tokens.weight"));
    }

    #[test]
    fn sanitize_keeps_resampler_keys_when_multimodal() {
        // keep_vision_tower=true → vit_merger.* / merger.* must be retained.
        let mut w: HashMap<String, Array> = HashMap::new();
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert("vit_merger.pre_norm.weight".into(), arr.clone());
        w.insert("vit_merger.self_attn.q_proj.weight".into(), arr.clone());
        w.insert("merger.mlp.0.linear_1.weight".into(), arr.clone());
        w.insert("merger.mlp.0.pre_norm.weight".into(), arr.clone());
        w.insert("vision_tower.encoder.layers.0.weight".into(), arr.clone());
        w.insert("model.embed_tokens.weight".into(), arr.clone());

        Loader::sanitize(&mut w, &empty_text_config(), true).unwrap();

        assert!(
            w.contains_key("vit_merger.pre_norm.weight"),
            "vit_merger.* must be kept when keep_vision_tower=true"
        );
        assert!(
            w.contains_key("vit_merger.self_attn.q_proj.weight"),
            "vit_merger.* must be kept when keep_vision_tower=true"
        );
        assert!(
            w.contains_key("merger.mlp.0.linear_1.weight"),
            "merger.* must be kept when keep_vision_tower=true"
        );
        assert!(
            w.contains_key("merger.mlp.0.pre_norm.weight"),
            "merger.* must be kept when keep_vision_tower=true"
        );
        assert!(
            w.contains_key("vision_tower.encoder.layers.0.weight"),
            "vision_tower.* must be kept when keep_vision_tower=true"
        );
    }

    #[test]
    fn sanitize_drops_resampler_keys_when_text_only() {
        // keep_vision_tower=false → vit_merger.* / merger.* must be dropped.
        let mut w: HashMap<String, Array> = HashMap::new();
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        w.insert("vit_merger.pre_norm.weight".into(), arr.clone());
        w.insert("vit_merger.self_attn.q_proj.weight".into(), arr.clone());
        w.insert("merger.mlp.0.linear_1.weight".into(), arr.clone());
        w.insert("merger.mlp.0.pre_norm.weight".into(), arr.clone());
        w.insert("vision_tower.encoder.layers.0.weight".into(), arr.clone());
        w.insert("model.embed_tokens.weight".into(), arr.clone());

        Loader::sanitize(&mut w, &empty_text_config(), false).unwrap();

        assert!(
            !w.contains_key("vit_merger.pre_norm.weight"),
            "vit_merger.* must be dropped when keep_vision_tower=false"
        );
        assert!(
            !w.contains_key("vit_merger.self_attn.q_proj.weight"),
            "vit_merger.* must be dropped when keep_vision_tower=false"
        );
        assert!(
            !w.contains_key("merger.mlp.0.linear_1.weight"),
            "merger.* must be dropped when keep_vision_tower=false"
        );
        assert!(
            !w.contains_key("merger.mlp.0.pre_norm.weight"),
            "merger.* must be dropped when keep_vision_tower=false"
        );
        assert!(
            !w.contains_key("vision_tower.encoder.layers.0.weight"),
            "vision_tower.* must be dropped when keep_vision_tower=false"
        );
        assert!(
            w.contains_key("model.embed_tokens.weight"),
            "plain model.* key must be preserved"
        );
    }

    #[test]
    fn sanitize_keeps_vision_tower_keys_when_requested() {
        let mut w: HashMap<String, Array> = HashMap::new();
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();
        // vision_tower.* must be retained when keep_vision_tower=true.
        w.insert("vision_tower.patch_embed.proj.weight".into(), arr.clone());
        w.insert(
            "embed_vision.embedding_projection.weight".into(),
            arr.clone(),
        );
        // Audio is still unsupported and must not survive open_multimodal.
        w.insert("audio_tower.layers.0.weight".into(), arr.clone());
        w.insert(
            "embed_audio.embedding_projection.weight".into(),
            arr.clone(),
        );
        w.insert("model.embed_tokens.weight".into(), arr.clone());

        Loader::sanitize(&mut w, &empty_text_config(), true).unwrap();

        assert!(
            w.contains_key("vision_tower.patch_embed.proj.weight"),
            "vision_tower key must be kept when keep_vision_tower=true"
        );
        assert!(
            w.contains_key("embed_vision.embedding_projection.weight"),
            "embed_vision key must be kept when keep_vision_tower=true"
        );
        assert!(
            !w.contains_key("audio_tower.layers.0.weight"),
            "audio_tower key must be dropped even when keep_vision_tower=true"
        );
        assert!(
            !w.contains_key("embed_audio.embedding_projection.weight"),
            "embed_audio key must be dropped even when keep_vision_tower=true"
        );
        assert!(
            w.contains_key("model.embed_tokens.weight"),
            "plain model.* key must be preserved"
        );
    }

    #[test]
    fn sanitize_drops_unified_vision_embedder_keys_only_for_text_only() {
        let arr: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();

        let mut text_only: HashMap<String, Array> = HashMap::new();
        text_only.insert("vision_embedder.patch_dense.weight".into(), arr.clone());
        text_only.insert("vision_embedder.patch_ln1.weight".into(), arr.clone());
        text_only.insert(
            "embed_vision.embedding_projection.weight".into(),
            arr.clone(),
        );
        text_only.insert(
            "language_model.model.embed_tokens.weight".into(),
            arr.clone(),
        );

        Loader::sanitize(&mut text_only, &gemma4_unified_text_config(), false).unwrap();

        assert!(
            !text_only.contains_key("vision_embedder.patch_dense.weight"),
            "text-only Gemma4 unified load must drop vision_embedder.*"
        );
        assert!(
            !text_only.contains_key("vision_embedder.patch_ln1.weight"),
            "text-only Gemma4 unified load must drop all vision_embedder.*"
        );
        assert!(
            !text_only.contains_key("embed_vision.embedding_projection.weight"),
            "text-only Gemma4 unified load must drop embed_vision.*"
        );
        assert!(
            text_only.contains_key("model.embed_tokens.weight"),
            "language_model. text prefix should still be stripped"
        );

        let mut multimodal: HashMap<String, Array> = HashMap::new();
        multimodal.insert("vision_embedder.patch_dense.weight".into(), arr.clone());
        multimodal.insert("vision_embedder.patch_ln1.weight".into(), arr.clone());
        multimodal.insert(
            "embed_vision.embedding_projection.weight".into(),
            arr.clone(),
        );
        multimodal.insert("audio_tower.depthwise_conv1d.weight".into(), arr.clone());
        multimodal.insert(
            "embed_audio.embedding_projection.weight".into(),
            arr.clone(),
        );
        multimodal.insert(
            "language_model.model.embed_tokens.weight".into(),
            arr.clone(),
        );

        Loader::sanitize(&mut multimodal, &gemma4_unified_text_config(), true).unwrap();

        assert!(
            multimodal.contains_key("vision_embedder.patch_dense.weight"),
            "multimodal Gemma4 unified load must keep vision_embedder.*"
        );
        assert!(
            multimodal.contains_key("vision_embedder.patch_ln1.weight"),
            "multimodal Gemma4 unified load must keep all vision_embedder.*"
        );
        assert!(
            multimodal.contains_key("embed_vision.embedding_projection.weight"),
            "multimodal Gemma4 unified load must keep embed_vision.*"
        );
        assert!(
            !multimodal.contains_key("audio_tower.depthwise_conv1d.weight"),
            "audio tower keys remain unsupported and must be dropped"
        );
        assert!(
            !multimodal.contains_key("embed_audio.embedding_projection.weight"),
            "audio embedder keys remain unsupported and must be dropped"
        );
        assert!(
            multimodal.contains_key("model.embed_tokens.weight"),
            "language_model. text prefix should still be stripped"
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

        Loader::sanitize(&mut w, &empty_text_config(), false).unwrap();

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
