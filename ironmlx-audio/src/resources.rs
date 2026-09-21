//! Strict, component-scoped safetensors access. No model/runtime reverse dependency.
pub mod derived;
use crate::{AudioError, ResourceIssue, ResourceReport, Result};
use ironmlx_core::weights::WeightMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub dtype: String,
    pub shape: Vec<usize>,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentSpec {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub tensors: BTreeMap<String, TensorSpec>,
}
#[derive(Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}
fn mismatch(component: &str, reason: impl Into<String>) -> AudioError {
    AudioError::ResourceMismatch {
        component: component.into(),
        reason: reason.into(),
    }
}
fn dtype_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "F16" | "BF16" => Some(2),
        "F32" => Some(4),
        "I64" => Some(8),
        _ => None,
    }
}

/// Verify the complete file, offsets and all tensor names/shapes/dtypes before allocation.
pub fn inspect_component(path: &Path, spec: &ComponentSpec) -> Result<u64> {
    let mut file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AudioError::ResourceMissing {
                component: spec.name.clone(),
            }
        } else {
            AudioError::Io(e)
        }
    })?;
    let size = file.metadata()?.len();
    if size != spec.bytes {
        return Err(mismatch(
            &spec.name,
            format!("file size {size} != {}", spec.bytes),
        ));
    }
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > 16 * 1024 * 1024 || header_len > size.saturating_sub(8) {
        return Err(mismatch(&spec.name, "invalid safetensors header length"));
    }
    let mut header = vec![0; header_len as usize];
    file.read_exact(&mut header)?;
    let mut raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|e| mismatch(&spec.name, e.to_string()))?;
    raw.remove("__metadata__");
    if raw.len() != spec.tensors.len() {
        return Err(mismatch(&spec.name, "tensor count mismatch"));
    }
    let mut ranges = Vec::new();
    for (name, want) in &spec.tensors {
        let actual: TensorHeader = serde_json::from_value(
            raw.remove(name)
                .ok_or_else(|| mismatch(&spec.name, format!("missing tensor {name}")))?,
        )
        .map_err(|e| mismatch(&spec.name, format!("{name}: {e}")))?;
        if actual.dtype != want.dtype || actual.shape != want.shape {
            return Err(mismatch(
                &spec.name,
                format!("{name}: shape/dtype mismatch"),
            ));
        }
        let bytes = actual
            .shape
            .iter()
            .try_fold(
                dtype_bytes(&actual.dtype)
                    .ok_or_else(|| mismatch(&spec.name, "unsupported dtype"))?,
                |n, d| n.checked_mul(*d as u64),
            )
            .ok_or_else(|| mismatch(&spec.name, "tensor size overflow"))?;
        if actual.data_offsets[1].checked_sub(actual.data_offsets[0]) != Some(bytes) {
            return Err(mismatch(&spec.name, format!("{name}: invalid offsets")));
        }
        ranges.push(actual.data_offsets);
    }
    ranges.sort_unstable();
    let mut tensor_bytes = 0;
    for [start, end] in ranges {
        if start != tensor_bytes {
            return Err(mismatch(&spec.name, "gapped/overlapping tensors"));
        }
        tensor_bytes = end;
    }
    if tensor_bytes.checked_add(header_len + 8) != Some(size) {
        return Err(mismatch(&spec.name, "tensor payload size mismatch"));
    }
    let mut hash = Sha256::new();
    hash.update(prefix);
    hash.update(header);
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    if format!("{:x}", hash.finalize()) != spec.sha256 {
        return Err(mismatch(&spec.name, "SHA-256 mismatch"));
    }
    Ok(tensor_bytes)
}

/// Load one verified component with the shared core's WeightSource implementation.
/// Tensors stay in the original checkpoint layout. Layer loaders make explicit permutations.
pub fn load_component(path: &Path, spec: &ComponentSpec) -> Result<WeightMap> {
    inspect_component(path, spec)?;
    let path = path
        .to_str()
        .ok_or_else(|| mismatch(&spec.name, "path is not UTF-8"))?;
    let (tensors, _) = mlx::io::load_safetensors(path)?;
    for (name, tensor) in &tensors {
        if tensor.dtype() != mlx::Dtype::Int64
            && !mlx::ops::all(&mlx::ops::isfinite(tensor)?, mlx::ops::All, false)?.item::<bool>()?
        {
            return Err(mismatch(&spec.name, format!("{name}: non-finite weight")));
        }
    }
    Ok(WeightMap::new(tensors, None, HashMap::new()))
}

/// Published MLX components, each with its own namespace and immutable schema.
#[derive(Clone, Copy, Debug)]
pub enum IndexTts25Component {
    Gpt,
    Codec,
    S2Mel,
    BigVgan,
    W2vBert,
}
impl IndexTts25Component {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Gpt => "gpt.safetensors",
            Self::Codec => "codec.safetensors",
            Self::S2Mel => "s2mel.safetensors",
            Self::BigVgan => "bigvgan.safetensors",
            Self::W2vBert => "model.safetensors",
        }
    }
    pub fn spec(self) -> Result<ComponentSpec> {
        let schema = match self {
            Self::Gpt => include_str!("../resources/indextts25/gpt.schema.json"),
            Self::Codec => include_str!("../resources/indextts25/codec.schema.json"),
            Self::S2Mel => include_str!("../resources/indextts25/s2mel.schema.json"),
            Self::BigVgan => include_str!("../resources/indextts25/bigvgan.schema.json"),
            Self::W2vBert => include_str!("../resources/indextts25/model.schema.json"),
        };
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../resources/indextts25/sources.json"))
                .map_err(|e| mismatch("profile", e.to_string()))?;
        let file = manifest["source"]["files"]
            .as_array()
            .and_then(|v| v.iter().find(|v| v["path"] == self.file_name()))
            .ok_or_else(|| mismatch("profile", "component missing"))?;
        Ok(ComponentSpec {
            name: self.file_name().into(),
            bytes: file["bytes"]
                .as_u64()
                .ok_or_else(|| mismatch("profile", "invalid size"))?,
            sha256: file["sha256"]
                .as_str()
                .ok_or_else(|| mismatch("profile", "missing digest"))?
                .into(),
            tensors: serde_json::from_str(schema)
                .map_err(|e| mismatch("profile", e.to_string()))?,
        })
    }
}
/// Inspect independent components and collect every failure instead of hiding later missing files.
/// This report covers the requested components only, not readiness of the whole TTS model.
pub fn inspect_components(components: &[(PathBuf, ComponentSpec)]) -> ResourceReport {
    let mut report = ResourceReport {
        complete: true,
        issues: Vec::new(),
        static_tensor_bytes: 0,
    };
    for (path, spec) in components {
        match inspect_component(path, spec) {
            Ok(bytes) => report.static_tensor_bytes += bytes,
            Err(error) => {
                report.complete = false;
                report.issues.push(ResourceIssue {
                    component: spec.name.clone(),
                    reason: error.to_string(),
                });
            }
        }
    }
    report
}
