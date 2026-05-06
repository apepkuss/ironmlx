//! Neural-network primitives shared across model architectures.
//!
//! Each layer exposes a `from_loader(&Loader, prefix)` static constructor
//! that reads its weights directly. Forward methods are inherent (per-layer);
//! there is no `Module` trait — see P1 spec § 3 for rationale.

pub mod embedding;
pub mod linear;

pub use embedding::Embedding;
pub use linear::Linear;

// Added in later P1 tasks:
// pub mod norm;
// pub mod mlp;
// pub mod mrope;
// pub mod attention;
