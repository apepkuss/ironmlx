//! Neural-network primitives shared across model architectures.
//!
//! Each layer exposes a `from_loader(&Loader, prefix)` static constructor
//! that reads its weights directly. Forward methods are inherent (per-layer);
//! there is no `Module` trait — see P1 spec § 3 for rationale.

pub mod attention;
pub mod embedding;
pub mod gated_attention; // NEW — P3b2
pub mod linear;
pub mod mlp;
pub mod mrope;
pub mod norm;

pub use attention::{Attention, AttentionConfig};
pub use embedding::Embedding;
pub use gated_attention::{GatedAttention, GatedAttentionConfig}; // NEW — P3b2
pub use linear::Linear;
pub use mlp::Mlp;
pub use mrope::Mrope;
pub use norm::{LayerNorm, RmsNorm};
