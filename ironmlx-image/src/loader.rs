//! Diffusers-style component loading and metadata preflight for image models.

use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Context};
use ironmlx_core::weights::{QuantMeta, QuantMode, WeightMap, WeightSource};
use mlx::Array;
use serde::Serialize;

use crate::Result;

const QWEN_IMAGE_21_MODEL_TYPE: &str = "qwen_image_2_1";

/// Metadata-only compatibility result used before weight transfer begins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageModelMetadataPreflight {
    pub model_type: String,
    pub artifact_role: &'static str,
    pub quantization: Option<ImageQuantizationMetadataPreflight>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageQuantizationMetadataPreflight {
    pub mode: &'static str,
    pub bits: i32,
    pub group_size: i32,
    pub override_count: usize,
}

/// Loader for one Diffusers-style component directory such as
/// `text_encoder/`, `transformer/`, or `vae/`.
///
/// The exact Qwen Image checkpoint contract permits either full-precision
/// tensors or one global affine-4/group-size-64 declaration. Per-prefix
/// overrides are rejected by metadata preflight rather than silently applied.
pub struct ComponentLoader {
    weights: WeightMap,
    config_raw: serde_json::Value,
    component_dir: PathBuf,
}

impl WeightSource for ComponentLoader {
    fn tensor(&self, key: &str) -> Result<&Array> {
        self.weights.tensor(key)
    }

    fn tensor_opt(&self, key: &str) -> Option<&Array> {
        self.weights.tensor_opt(key)
    }

    fn contains(&self, key: &str) -> bool {
        self.weights.contains(key)
    }

    fn quant_meta_for(&self, prefix: &str) -> Option<QuantMeta> {
        self.weights.quant_meta_for(prefix)
    }
}

impl ComponentLoader {
    pub fn open(model_dir: &Path, component: &str) -> Result<Self> {
        Self::open_filtered(model_dir, component, |_, _| true)
    }

    /// Open a component while retaining only tensors used by the execution
    /// graph. Filtering happens before eager evaluation, so unused towers do
    /// not become resident merely because they share a safetensors shard.
    pub fn open_filtered(
        model_dir: &Path,
        component: &str,
        mut keep: impl FnMut(&str, &Array) -> bool,
    ) -> Result<Self> {
        let mut parts = Path::new(component).components();
        if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
            return Err(anyhow!(
                "component name must be one normal path segment, got {component:?}"
            ));
        }

        let component_dir = model_dir.join(component);
        let config_path = component_dir.join("config.json");
        let config_raw: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("opening {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;
        let quant = component_quantization(&config_raw)?;
        let mut tensors = load_safetensors(&component_dir)?;
        tensors.retain(|key, value| keep(key, value));
        if tensors.is_empty() {
            return Err(anyhow!(
                "component filter retained no tensors from {}",
                component_dir.display()
            ));
        }
        validate_quantized_storage(&tensors, quant)?;

        let refs: Vec<&Array> = tensors.values().collect();
        mlx::transforms::eval(&refs)
            .with_context(|| format!("eager evaluation of {component} weights"))?;

        Ok(Self {
            weights: WeightMap::new(tensors, quant, HashMap::new()),
            config_raw,
            component_dir,
        })
    }

    pub fn tensor(&self, key: &str) -> Result<&Array> {
        self.weights
            .tensor_opt(key)
            .ok_or_else(|| anyhow!("ComponentLoader: missing tensor key `{key}`"))
    }

    pub fn tensor_opt(&self, key: &str) -> Option<&Array> {
        self.weights.tensor_opt(key)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.weights.contains(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.weights.tensors().keys().map(String::as_str)
    }

    pub fn quant_meta(&self) -> Option<QuantMeta> {
        self.weights.quant_meta()
    }

    pub fn quant_meta_for(&self, prefix: &str) -> Option<QuantMeta> {
        self.weights.quant_meta_for(prefix)
    }

    pub fn config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        Ok(serde_json::from_value(self.config_raw.clone())?)
    }

    pub fn config_raw_value(&self) -> &serde_json::Value {
        &self.config_raw
    }

    pub fn component_dir(&self) -> &Path {
        &self.component_dir
    }

    pub fn loaded_tensor_bytes(&self) -> usize {
        self.weights
            .tensors()
            .values()
            .fold(0usize, |total, tensor| {
                total.saturating_add(tensor.size().saturating_mul(tensor.dtype().byte_size()))
            })
    }
}

/// Validate the exact Qwen Image 2.1 component layout and quantization without
/// opening model weights.
pub fn preflight_model_metadata(model_dir: &Path) -> Result<ImageModelMetadataPreflight> {
    let model_index = required_json_file(model_dir, "model_index.json")?;
    exact_string(
        &model_index,
        "_class_name",
        "QwenImage21Pipeline",
        "pipeline",
    )?;

    let text_encoder = required_json_file(model_dir, "text_encoder/config.json")?;
    exact_string(&text_encoder, "model_type", "qwen3_vl", "text encoder")?;
    let text = text_encoder
        .get("text_config")
        .ok_or_else(|| anyhow!("Qwen Image 2.1 text_encoder/config.json missing text_config"))?;
    exact_string(text, "model_type", "qwen3_vl_text", "text encoder")?;
    exact_u64(text, "hidden_size", 4096, "text encoder")?;
    exact_u64(text, "intermediate_size", 12288, "text encoder")?;
    exact_u64(text, "num_hidden_layers", 36, "text encoder")?;
    exact_u64(text, "num_attention_heads", 32, "text encoder")?;
    exact_u64(text, "num_key_value_heads", 8, "text encoder")?;

    let transformer = required_json_file(model_dir, "transformer/config.json")?;
    exact_string(
        &transformer,
        "_class_name",
        "QwenImage21Transformer2DModel",
        "transformer",
    )?;
    exact_u64(&transformer, "in_channels", 64, "transformer")?;
    exact_u64(&transformer, "out_channels", 64, "transformer")?;
    exact_u64(&transformer, "num_layers", 32, "transformer")?;
    exact_u64(&transformer, "num_attention_heads", 32, "transformer")?;
    exact_u64(&transformer, "attention_head_dim", 128, "transformer")?;

    let vae = required_json_file(model_dir, "vae/config.json")?;
    exact_string(&vae, "_class_name", "AutoencoderKLQwenImage21", "VAE")?;
    exact_u64(&vae, "z_dim", 64, "VAE")?;
    exact_u64(&vae, "scale_factor_spatial", 16, "VAE")?;

    let scheduler = required_json_file(model_dir, "scheduler/scheduler_config.json")?;
    exact_string(
        &scheduler,
        "_class_name",
        "FlowMatchEulerDiscreteScheduler",
        "scheduler",
    )?;

    let required_quant = QuantMeta {
        group_size: 64,
        bits: 4,
        mode: QuantMode::Affine,
    };
    if component_quantization(&text_encoder)? != Some(required_quant)
        || component_quantization(&transformer)? != Some(required_quant)
    {
        return Err(anyhow!(
            "Qwen Image 2.1 requires affine 4-bit/group-size-64 text and transformer weights"
        ));
    }

    Ok(ImageModelMetadataPreflight {
        model_type: QWEN_IMAGE_21_MODEL_TYPE.to_owned(),
        artifact_role: "image_generation",
        quantization: Some(ImageQuantizationMetadataPreflight {
            mode: "affine",
            bits: 4,
            group_size: 64,
            override_count: 0,
        }),
    })
}

fn required_json_file(model_dir: &Path, relative: &str) -> Result<serde_json::Value> {
    let path = model_dir.join(relative);
    serde_json::from_reader(
        std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))
}

fn exact_u64(config: &serde_json::Value, key: &str, expected: u64, component: &str) -> Result<()> {
    let actual = config.get(key).and_then(serde_json::Value::as_u64);
    if actual != Some(expected) {
        return Err(anyhow!(
            "unsupported Qwen Image 2.1 {component} {key}: expected {expected}, got {actual:?}"
        ));
    }
    Ok(())
}

fn exact_string(
    config: &serde_json::Value,
    key: &str,
    expected: &str,
    component: &str,
) -> Result<()> {
    let actual = config.get(key).and_then(serde_json::Value::as_str);
    if actual != Some(expected) {
        return Err(anyhow!(
            "unsupported Qwen Image 2.1 {component} {key}: expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn component_quantization(config: &serde_json::Value) -> Result<Option<QuantMeta>> {
    let Some(value) = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
    else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("quantization must be a JSON object"))?;
    for (key, value) in object {
        if value.as_object().is_some()
            && (value.get("bits").is_some() || value.get("group_size").is_some())
        {
            return Err(anyhow!(
                "Qwen Image 2.1 does not support quantization override `{key}`"
            ));
        }
    }
    let mode = object
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("quantization.mode missing or non-string"))?;
    let bits = object
        .get("bits")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| anyhow!("quantization.bits missing or non-integer"))?;
    let group_size = object
        .get("group_size")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| anyhow!("quantization.group_size missing or non-integer"))?;
    if mode != "affine" || bits != 4 || group_size != 64 {
        return Err(anyhow!(
            "Qwen Image 2.1 requires affine 4-bit/group-size-64 weights, got mode={mode:?}, bits={bits}, group_size={group_size}"
        ));
    }
    Ok(Some(QuantMeta {
        group_size: 64,
        bits: 4,
        mode: QuantMode::Affine,
    }))
}

fn validate_quantized_storage(
    tensors: &HashMap<String, Array>,
    quant: Option<QuantMeta>,
) -> Result<()> {
    let mut scale_prefixes: Vec<String> = tensors
        .keys()
        .filter_map(|key| key.strip_suffix(".scales").map(str::to_owned))
        .collect();
    scale_prefixes.sort();
    for prefix in scale_prefixes {
        let weight_key = format!("{prefix}.weight");
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");
        let weight = tensors.get(&weight_key).ok_or_else(|| {
            anyhow!("{prefix}: quantized storage has `{scales_key}` but missing `{weight_key}`")
        })?;
        let scales = tensors
            .get(&scales_key)
            .expect("scale key was collected from tensors");
        let biases = tensors.get(&biases_key);
        quant
            .ok_or_else(|| {
                anyhow!("{prefix}: quantized storage exists without quantization metadata")
            })?
            .validate_storage(&prefix, weight, scales, biases)?;
    }
    for prefix in tensors.keys().filter_map(|key| key.strip_suffix(".biases")) {
        if !tensors.contains_key(&format!("{prefix}.scales")) {
            return Err(anyhow!(
                "{prefix}: quantized storage has `{prefix}.biases` but missing `{prefix}.scales`"
            ));
        }
    }
    Ok(())
}

fn load_safetensors(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let single = model_dir.join("model.safetensors");
    let sharded = model_dir.join("model.safetensors.index.json");
    if single.exists() {
        let (tensors, _metadata) = mlx::io::load_safetensors(
            single
                .to_str()
                .ok_or_else(|| anyhow!("non-UTF-8 path: {}", single.display()))?,
        )
        .map_err(|error| anyhow!("load_safetensors {}: {error}", single.display()))?;
        return Ok(tensors);
    }
    if sharded.exists() {
        let index: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&sharded)
                .with_context(|| format!("opening {}", sharded.display()))?,
        )
        .with_context(|| format!("parsing {}", sharded.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| anyhow!("safetensors index missing weight_map"))?;
        let shards: BTreeSet<&str> = weight_map
            .values()
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| anyhow!("safetensors index contains a non-string shard"))
            })
            .collect::<Result<_>>()?;
        let mut tensors = HashMap::new();
        for shard in shards {
            let path = model_dir.join(shard);
            let (loaded, _metadata) = mlx::io::load_safetensors(
                path.to_str()
                    .ok_or_else(|| anyhow!("non-UTF-8 path: {}", path.display()))?,
            )
            .map_err(|error| anyhow!("load_safetensors {}: {error}", path.display()))?;
            for (key, tensor) in loaded {
                if tensors.insert(key.clone(), tensor).is_some() {
                    return Err(anyhow!(
                        "safetensors shards contain duplicate tensor key `{key}`"
                    ));
                }
            }
        }
        return Ok(tensors);
    }
    Err(anyhow!(
        "no model.safetensors or model.safetensors.index.json in {}",
        model_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_qwen_image_21_metadata(dir: &Path) {
        for component in ["text_encoder", "transformer", "vae", "scheduler"] {
            std::fs::create_dir_all(dir.join(component)).expect("create component directory");
        }
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name":"QwenImage21Pipeline"}"#,
        )
        .expect("write model index");
        std::fs::write(
            dir.join("text_encoder/config.json"),
            r#"{
                "model_type":"qwen3_vl",
                "text_config":{
                    "model_type":"qwen3_vl_text",
                    "hidden_size":4096,
                    "intermediate_size":12288,
                    "num_hidden_layers":36,
                    "num_attention_heads":32,
                    "num_key_value_heads":8
                },
                "quantization":{"mode":"affine","bits":4,"group_size":64}
            }"#,
        )
        .expect("write text encoder config");
        std::fs::write(
            dir.join("transformer/config.json"),
            r#"{
                "_class_name":"QwenImage21Transformer2DModel",
                "in_channels":64,
                "out_channels":64,
                "num_layers":32,
                "num_attention_heads":32,
                "attention_head_dim":128,
                "quantization":{"mode":"affine","bits":4,"group_size":64}
            }"#,
        )
        .expect("write transformer config");
        std::fs::write(
            dir.join("vae/config.json"),
            r#"{
                "_class_name":"AutoencoderKLQwenImage21",
                "z_dim":64,
                "scale_factor_spatial":16
            }"#,
        )
        .expect("write VAE config");
        std::fs::write(
            dir.join("scheduler/scheduler_config.json"),
            r#"{"_class_name":"FlowMatchEulerDiscreteScheduler"}"#,
        )
        .expect("write scheduler config");
    }

    #[test]
    fn metadata_preflight_accepts_qwen_image_21_component_layout() {
        let dir = std::env::temp_dir().join(format!(
            "ironmlx-qwen-image-21-preflight-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create preflight dir");
        write_qwen_image_21_metadata(&dir);

        let result = preflight_model_metadata(&dir).expect("preflight Qwen Image 2.1 metadata");

        assert_eq!(result.model_type, "qwen_image_2_1");
        assert_eq!(result.artifact_role, "image_generation");
        assert_eq!(
            result.quantization,
            Some(ImageQuantizationMetadataPreflight {
                mode: "affine",
                bits: 4,
                group_size: 64,
                override_count: 0,
            })
        );
        std::fs::remove_dir_all(dir).expect("cleanup preflight dir");
    }

    #[test]
    fn metadata_preflight_rejects_mismatched_qwen_image_transformer_shape() {
        let dir = std::env::temp_dir().join(format!(
            "ironmlx-qwen-image-21-preflight-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create preflight dir");
        write_qwen_image_21_metadata(&dir);
        std::fs::write(
            dir.join("transformer/config.json"),
            r#"{
                "_class_name":"QwenImage21Transformer2DModel",
                "in_channels":32,
                "out_channels":64,
                "num_layers":32,
                "num_attention_heads":32,
                "attention_head_dim":128,
                "quantization":{"mode":"affine","bits":4,"group_size":64}
            }"#,
        )
        .expect("rewrite transformer config");

        let error = preflight_model_metadata(&dir).expect_err("reject incompatible transformer");

        assert!(error.to_string().contains("in_channels"));
        std::fs::remove_dir_all(dir).expect("cleanup preflight dir");
    }

    #[test]
    fn metadata_preflight_rejects_qwen_image_quantization_overrides() {
        let dir = std::env::temp_dir().join(format!(
            "ironmlx-qwen-image-21-preflight-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create preflight dir");
        write_qwen_image_21_metadata(&dir);
        std::fs::write(
            dir.join("text_encoder/config.json"),
            r#"{
                "model_type":"qwen3_vl",
                "text_config":{
                    "model_type":"qwen3_vl_text",
                    "hidden_size":4096,
                    "intermediate_size":12288,
                    "num_hidden_layers":36,
                    "num_attention_heads":32,
                    "num_key_value_heads":8
                },
                "quantization":{
                    "mode":"affine",
                    "bits":4,
                    "group_size":64,
                    "model.layers.0.self_attn.q_proj":{
                        "bits":8,
                        "group_size":64
                    }
                }
            }"#,
        )
        .expect("rewrite text encoder config");

        let error = preflight_model_metadata(&dir).expect_err("reject quantization override");

        assert!(error
            .to_string()
            .contains("does not support quantization override"));
        std::fs::remove_dir_all(dir).expect("cleanup preflight dir");
    }

    #[test]
    fn component_loader_rejects_nested_component_paths() {
        let error = match ComponentLoader::open(Path::new("/unused"), "../transformer") {
            Ok(_) => panic!("nested component path must be rejected"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("component name must be one normal path segment"));
    }
}
