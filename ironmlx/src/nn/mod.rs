//! Neural-network primitives shared across model architectures.
//!
//! Each layer exposes a `from_loader(&Loader, prefix)` static constructor
//! that reads its weights directly. Forward methods are inherent (per-layer);
//! there is no `Module` trait — see P1 spec § 3 for rationale.

pub(crate) mod activations;
pub(crate) use activations::gelu_tanh;
pub mod attention;
pub(crate) mod batch_stable_qmm;
pub mod conv;
pub mod decoder_layer;
pub mod embedding;
pub mod gated_attention;
pub mod gated_delta_net;
pub(crate) mod gemma4_verify_attention;
pub mod linear;
pub mod mlp;
pub mod mrope;
pub mod mtp;
pub mod norm;
pub(crate) mod position_stable_qmm;
pub(crate) mod product_stable_qmm;
pub(crate) mod sorted_moe_weighted_sum;
pub(crate) mod verify_qmm;

pub use attention::{Attention, AttentionConfig};
pub use conv::{Conv1d, Conv1dConfig};
pub use decoder_layer::{
    enable_paged_hot_cold_tiering_caches, enable_paged_kv_caches, enable_turboquant_kv_caches,
    paged_prefix_key_spec_for_full_caches, paged_prefix_layers_for_row, prefix_entry_for_row,
    prefix_key_spec_for_caches, restore_paged_prefix_layers_for_row, restore_prefix_entry_for_row,
    restore_prefix_entry_for_rows, AttnKind, AttnPath, DecoderLayer, DecoderLayerConfig,
    LayerCache, LayerCacheSnapshot,
};
pub use embedding::Embedding;
pub use gated_attention::{GatedAttention, GatedAttentionConfig};
pub use gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
pub use linear::Linear;
pub use mlp::Mlp;
pub use mrope::Mrope;
pub use mtp::{Mtp, MtpConfig, MtpStepOutput};
pub use norm::{LayerNorm, RmsNorm, RmsNormGated};

/// Enter the same scoped verify route used by the production scheduler.
///
/// This is exposed for real-model qualification tests that must exercise the
/// exact Q>1 execution path rather than an ordinary multi-token prefill.
#[doc(hidden)]
pub fn verify_qmm_scope() -> impl Drop {
    verify_qmm::armed_scope()
}
