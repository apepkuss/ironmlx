//! Borrowed model cache chunks used by offload adapters; independent of storage and residency policy.

use super::prefix_payload::{PagedPrefixEntry, PrefixLayerPayload, PrefixMtpLayerPayload};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveKvLayerChunkKind {
    FullDense,
    FullPaged,
    FullTurboQuantPacked,
    Mla,
    GatedDeltaLinear,
    MtpSpeculativeSideCache,
}

impl ActiveKvLayerChunkKind {
    pub fn is_supported_for_active_offload(self) -> bool {
        matches!(
            self,
            Self::FullDense
                | Self::FullPaged
                | Self::FullTurboQuantPacked
                | Self::Mla
                | Self::GatedDeltaLinear
                | Self::MtpSpeculativeSideCache
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ActiveKvLayerChunkPayload<'a> {
    FullDense {
        k: &'a mlx::Array,
        v: &'a mlx::Array,
    },
    FullPaged {
        k_pages: &'a mlx::Array,
        v_pages: &'a mlx::Array,
    },
    FullTurboQuantPacked {
        k_packed: &'a mlx::Array,
        k_norms: &'a mlx::Array,
        v_packed: &'a mlx::Array,
        v_norms: &'a mlx::Array,
    },
    GatedDeltaLinear {
        conv_state: &'a mlx::Array,
        recurrent_state: &'a mlx::Array,
    },
    Mla {
        c_kv: &'a mlx::Array,
        k_pe: &'a mlx::Array,
    },
    MtpSpeculativeSideCache {
        k: &'a mlx::Array,
        v: &'a mlx::Array,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ActiveKvLayerChunk<'a> {
    pub layer_index: usize,
    pub is_main_layer: bool,
    pub kind: ActiveKvLayerChunkKind,
    pub payload: ActiveKvLayerChunkPayload<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct ActiveKvEntryChunkReader<'a> {
    entry: &'a PagedPrefixEntry,
}

impl<'a> ActiveKvEntryChunkReader<'a> {
    pub fn new(entry: &'a PagedPrefixEntry) -> Self {
        Self { entry }
    }

    pub fn chunks(&self) -> impl Iterator<Item = ActiveKvLayerChunk<'a>> + '_ {
        self.entry
            .main_layers
            .iter()
            .enumerate()
            .map(main_layer_chunk)
            .chain(
                self.entry
                    .mtp_layers
                    .iter()
                    .enumerate()
                    .map(mtp_layer_chunk),
            )
    }
}

fn main_layer_chunk<'a>(
    (layer_index, layer): (usize, &'a PrefixLayerPayload),
) -> ActiveKvLayerChunk<'a> {
    match layer {
        PrefixLayerPayload::FullDense { k, v } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::FullDense,
            payload: ActiveKvLayerChunkPayload::FullDense { k, v },
        },
        PrefixLayerPayload::FullPaged { k_pages, v_pages } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::FullPaged,
            payload: ActiveKvLayerChunkPayload::FullPaged { k_pages, v_pages },
        },
        PrefixLayerPayload::FullTurboQuantPacked {
            k_packed,
            k_norms,
            v_packed,
            v_norms,
        } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::FullTurboQuantPacked,
            payload: ActiveKvLayerChunkPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            },
        },
        PrefixLayerPayload::Linear {
            conv_state,
            recurrent_state,
        } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::GatedDeltaLinear,
            payload: ActiveKvLayerChunkPayload::GatedDeltaLinear {
                conv_state,
                recurrent_state,
            },
        },
        PrefixLayerPayload::Mla { c_kv, k_pe } => ActiveKvLayerChunk {
            layer_index,
            is_main_layer: true,
            kind: ActiveKvLayerChunkKind::Mla,
            payload: ActiveKvLayerChunkPayload::Mla { c_kv, k_pe },
        },
    }
}

fn mtp_layer_chunk<'a>(
    (layer_index, layer): (usize, &'a PrefixMtpLayerPayload),
) -> ActiveKvLayerChunk<'a> {
    ActiveKvLayerChunk {
        layer_index,
        is_main_layer: false,
        kind: ActiveKvLayerChunkKind::MtpSpeculativeSideCache,
        payload: ActiveKvLayerChunkPayload::MtpSpeculativeSideCache {
            k: &layer.k,
            v: &layer.v,
        },
    }
}
