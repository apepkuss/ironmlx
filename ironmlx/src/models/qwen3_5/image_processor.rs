//! Pure-Rust port of HF `Qwen2VLImageProcessorFast`. See spec §4.1.
//!
//! Pipeline: decode → smart_resize → normalize → patchify.
