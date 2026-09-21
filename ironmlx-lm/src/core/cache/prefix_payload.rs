//! Model-side prefix cache keys, tensor payloads and metadata. No store, worker or filesystem dependencies.

use crate::Result;
use anyhow::Context;
use mlx::{Array, Dtype};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixLayerKind {
    FullDense,
    FullPaged,
    FullTurboQuantPacked,
    Linear,
    Mla,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixEntryKind {
    WholePrefix,
    ImmutableBlock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixTensorSpec {
    pub dtype: Dtype,
    pub shape: Vec<i32>,
}

impl PrefixTensorSpec {
    pub fn from_array(array: &Array) -> Self {
        Self {
            dtype: array.dtype(),
            shape: array.shape().as_slice().to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixLayerSpec {
    pub kind: PrefixLayerKind,
    pub tensors: Vec<PrefixTensorSpec>,
}

impl PrefixLayerSpec {
    pub fn from_payload(payload: &PrefixLayerPayload) -> Self {
        match payload {
            PrefixLayerPayload::FullDense { k, v } => Self {
                kind: PrefixLayerKind::FullDense,
                tensors: vec![
                    PrefixTensorSpec::from_array(k),
                    PrefixTensorSpec::from_array(v),
                ],
            },
            PrefixLayerPayload::FullPaged { k_pages, v_pages } => Self {
                kind: PrefixLayerKind::FullPaged,
                tensors: vec![
                    PrefixTensorSpec::from_array(k_pages),
                    PrefixTensorSpec::from_array(v_pages),
                ],
            },
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            } => Self {
                kind: PrefixLayerKind::FullTurboQuantPacked,
                tensors: vec![
                    PrefixTensorSpec::from_array(k_packed),
                    PrefixTensorSpec::from_array(k_norms),
                    PrefixTensorSpec::from_array(v_packed),
                    PrefixTensorSpec::from_array(v_norms),
                ],
            },
            PrefixLayerPayload::Linear {
                conv_state,
                recurrent_state,
            } => Self {
                kind: PrefixLayerKind::Linear,
                tensors: vec![
                    PrefixTensorSpec::from_array(conv_state),
                    PrefixTensorSpec::from_array(recurrent_state),
                ],
            },
            PrefixLayerPayload::Mla { c_kv, k_pe } => Self {
                kind: PrefixLayerKind::Mla,
                tensors: vec![
                    PrefixTensorSpec::from_array(c_kv),
                    PrefixTensorSpec::from_array(k_pe),
                ],
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMtpLayerSpec {
    pub k: PrefixTensorSpec,
    pub v: PrefixTensorSpec,
}

impl PrefixMtpLayerSpec {
    pub fn from_payload(payload: &PrefixMtpLayerPayload) -> Self {
        Self {
            k: PrefixTensorSpec::from_array(&payload.k),
            v: PrefixTensorSpec::from_array(&payload.v),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedPrefixKeySpec {
    pub entry_kind: PrefixEntryKind,
    pub model_id: String,
    pub token_ids: Vec<i32>,
    pub cached_len: i32,
    pub fingerprint: Option<String>,
    pub block_size: i32,
    pub kv_cache_profile: Option<String>,
    pub main_layers: Vec<PrefixLayerSpec>,
    pub mtp_layers: Vec<PrefixMtpLayerSpec>,
    pub mtp_last_hidden: Option<PrefixTensorSpec>,
    pub gemma4_drafter_last_hidden: Option<PrefixTensorSpec>,
}

impl PagedPrefixKeySpec {
    pub fn payload_bytes(&self) -> usize {
        let main = self
            .main_layers
            .iter()
            .flat_map(|layer| layer.tensors.iter())
            .fold(0usize, |bytes, tensor| {
                bytes.saturating_add(tensor_spec_payload_bytes(tensor))
            });
        let mtp = self.mtp_layers.iter().fold(0usize, |bytes, layer| {
            bytes
                .saturating_add(tensor_spec_payload_bytes(&layer.k))
                .saturating_add(tensor_spec_payload_bytes(&layer.v))
        });
        [
            self.mtp_last_hidden.as_ref(),
            self.gemma4_drafter_last_hidden.as_ref(),
        ]
        .into_iter()
        .flatten()
        .fold(main.saturating_add(mtp), |bytes, tensor| {
            bytes.saturating_add(tensor_spec_payload_bytes(tensor))
        })
    }
}

#[derive(Debug, Clone)]
pub struct PagedPrefixLayer {
    pub k_pages: Array,
    pub v_pages: Array,
}

#[derive(Debug, Clone)]
pub enum PrefixLayerPayload {
    FullDense {
        k: Array,
        v: Array,
    },
    FullPaged {
        k_pages: Array,
        v_pages: Array,
    },
    FullTurboQuantPacked {
        k_packed: Array,
        k_norms: Array,
        v_packed: Array,
        v_norms: Array,
    },
    Linear {
        conv_state: Array,
        recurrent_state: Array,
    },
    Mla {
        c_kv: Array,
        k_pe: Array,
    },
}

#[derive(Debug, Clone)]
pub struct PrefixMtpLayerPayload {
    pub k: Array,
    pub v: Array,
}

#[derive(Debug, Clone, Default)]
pub struct PagedPrefixEntry {
    pub main_layers: Vec<PrefixLayerPayload>,
    pub mtp_layers: Vec<PrefixMtpLayerPayload>,
    pub mtp_last_hidden: Option<Array>,
    pub gemma4_drafter_last_hidden: Option<Array>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedPrefixEntryStats {
    pub cached_len: i32,
    pub main_layers: usize,
    pub full_dense_layers: usize,
    pub full_paged_layers: usize,
    pub linear_layers: usize,
    pub mla_layers: usize,
    pub mtp_layers: usize,
    pub full_paged_pages: usize,
    pub tensor_count: usize,
    pub payload_bytes: usize,
}

impl PagedPrefixEntry {
    pub fn eval(&self) -> Result<()> {
        let mut arrays = Vec::new();
        for layer in &self.main_layers {
            match layer {
                PrefixLayerPayload::FullDense { k, v }
                | PrefixLayerPayload::FullPaged {
                    k_pages: k,
                    v_pages: v,
                } => arrays.extend([k, v]),
                PrefixLayerPayload::FullTurboQuantPacked {
                    k_packed,
                    k_norms,
                    v_packed,
                    v_norms,
                } => arrays.extend([k_packed, k_norms, v_packed, v_norms]),
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                } => arrays.extend([conv_state, recurrent_state]),
                PrefixLayerPayload::Mla { c_kv, k_pe } => arrays.extend([c_kv, k_pe]),
            }
        }
        for layer in &self.mtp_layers {
            arrays.extend([&layer.k, &layer.v]);
        }
        if let Some(hidden) = self.mtp_last_hidden.as_ref() {
            arrays.push(hidden);
        }
        if let Some(hidden) = self.gemma4_drafter_last_hidden.as_ref() {
            arrays.push(hidden);
        }
        mlx::transforms::eval(&arrays).context("evaluate async prefix store payload")?;
        Ok(())
    }

    pub fn main_layer_specs(&self) -> Vec<PrefixLayerSpec> {
        self.main_layers
            .iter()
            .map(PrefixLayerSpec::from_payload)
            .collect()
    }

    pub fn mtp_layer_specs(&self) -> Vec<PrefixMtpLayerSpec> {
        self.mtp_layers
            .iter()
            .map(PrefixMtpLayerSpec::from_payload)
            .collect()
    }

    pub fn mtp_last_hidden_spec(&self) -> Option<PrefixTensorSpec> {
        self.mtp_last_hidden
            .as_ref()
            .map(PrefixTensorSpec::from_array)
    }

    pub fn gemma4_drafter_last_hidden_spec(&self) -> Option<PrefixTensorSpec> {
        self.gemma4_drafter_last_hidden
            .as_ref()
            .map(PrefixTensorSpec::from_array)
    }

    pub fn observability_stats(&self, cached_len: i32) -> PagedPrefixEntryStats {
        let mut stats = PagedPrefixEntryStats {
            cached_len,
            main_layers: self.main_layers.len(),
            mtp_layers: self.mtp_layers.len(),
            ..PagedPrefixEntryStats::default()
        };
        for layer in &self.main_layers {
            match layer {
                PrefixLayerPayload::FullDense { k, v } => {
                    stats.full_dense_layers += 1;
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(k))
                        .saturating_add(tensor_payload_bytes(v));
                }
                PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                    stats.full_paged_layers += 1;
                    stats.full_paged_pages += first_dim_usize(k_pages);
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(k_pages))
                        .saturating_add(tensor_payload_bytes(v_pages));
                }
                PrefixLayerPayload::FullTurboQuantPacked {
                    k_packed,
                    k_norms,
                    v_packed,
                    v_norms,
                } => {
                    stats.tensor_count += 4;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(k_packed))
                        .saturating_add(tensor_payload_bytes(k_norms))
                        .saturating_add(tensor_payload_bytes(v_packed))
                        .saturating_add(tensor_payload_bytes(v_norms));
                }
                PrefixLayerPayload::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    stats.linear_layers += 1;
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(conv_state))
                        .saturating_add(tensor_payload_bytes(recurrent_state));
                }
                PrefixLayerPayload::Mla { c_kv, k_pe } => {
                    stats.mla_layers += 1;
                    stats.tensor_count += 2;
                    stats.payload_bytes = stats
                        .payload_bytes
                        .saturating_add(tensor_payload_bytes(c_kv))
                        .saturating_add(tensor_payload_bytes(k_pe));
                }
            }
        }
        for layer in &self.mtp_layers {
            stats.tensor_count += 2;
            stats.payload_bytes = stats
                .payload_bytes
                .saturating_add(tensor_payload_bytes(&layer.k))
                .saturating_add(tensor_payload_bytes(&layer.v));
        }
        if let Some(last_hidden) = &self.mtp_last_hidden {
            stats.tensor_count += 1;
            stats.payload_bytes = stats
                .payload_bytes
                .saturating_add(tensor_payload_bytes(last_hidden));
        }
        if let Some(last_hidden) = &self.gemma4_drafter_last_hidden {
            stats.tensor_count += 1;
            stats.payload_bytes = stats
                .payload_bytes
                .saturating_add(tensor_payload_bytes(last_hidden));
        }
        stats
    }
}

fn tensor_payload_bytes(array: &Array) -> usize {
    tensor_element_count(array).saturating_mul(dtype_size_bytes(array.dtype()))
}

fn tensor_spec_payload_bytes(spec: &PrefixTensorSpec) -> usize {
    spec.shape
        .iter()
        .try_fold(1usize, |elements, &dim| {
            usize::try_from(dim)
                .ok()
                .map(|dim| elements.saturating_mul(dim))
        })
        .unwrap_or(0)
        .saturating_mul(dtype_size_bytes(spec.dtype))
}

fn tensor_element_count(array: &Array) -> usize {
    let mut elements = 1_usize;
    for &dim in array.shape().as_slice() {
        let Ok(dim) = usize::try_from(dim) else {
            return 0;
        };
        elements = elements.saturating_mul(dim);
    }
    elements
}

fn dtype_size_bytes(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
        _ => 0,
    }
}

fn first_dim_usize(array: &Array) -> usize {
    array
        .shape()
        .as_slice()
        .first()
        .copied()
        .and_then(|dim| usize::try_from(dim).ok())
        .unwrap_or(0)
}
