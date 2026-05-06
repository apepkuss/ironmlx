//! Generation infrastructure that's model-agnostic.

pub mod loader;

pub use loader::{EosTokenId, Loader, QuantMeta, QuantMode, TokenizerConfig};

// Added in later P1 tasks:
// pub mod tokenizer;
// pub mod chat_template;
// pub mod sampler;
