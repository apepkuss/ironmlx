//! Neural-network primitives shared across model architectures.
//!
//! Each model in [`crate::models`] composes these building blocks. Anything
//! specific to a single architecture (e.g. Qwen3.5's gated delta network)
//! lives in that model's directory, not here.
//!
//! ## `Module` trait
//!
//! [`Module`] only manages parameter loading / iteration. `forward` is
//! intentionally *not* in the trait — model layers have heterogeneous
//! signatures (attention takes (q, k, v, mask, cache), MLP takes a single
//! `&Array`, etc.) and forcing a uniform `forward` would require boxing
//! arguments into enums. Each layer struct exposes its own forward inherent
//! method.

mod module;

pub use module::Module;

// P1 primitive layers — added incrementally.
// pub mod linear;
// pub mod embedding;
// pub mod norm;
// pub mod mrope;
// pub mod attention;
// pub mod mlp;
