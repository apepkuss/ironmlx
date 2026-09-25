//! Fixed IndexTTS 2.5 reference encoders. Model internals stay inside the crate.
mod campplus;
mod cfm;
mod codec;
mod conditioning;
mod gpt;
mod layers;
mod reference;
mod vocoder;
mod w2v;
pub use reference::{IndexTts25ReferenceEncoder, ReferenceConditioning};

mod model;
pub use model::{
    IndexTts25, IndexTts25Loader, INDEXTTS25_MAX_OUTPUT_FRAMES, INDEXTTS25_OUTPUT_SAMPLE_RATE_HZ,
};
