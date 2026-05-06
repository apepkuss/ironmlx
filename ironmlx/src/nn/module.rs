//! [`Module`] — parameter-management trait shared by all layer / model
//! structs.
//!
//! Forward methods are *not* in this trait (see `nn/mod.rs` doc comment).
//! `Module` only standardizes how a struct exposes its tensor parameters
//! for safetensors loading and (eventually) inspection / quantization.

use std::collections::HashMap;

use mlx::Array;

use crate::Result;

/// Trait implemented by every layer / model struct that owns tensor
/// parameters loadable from safetensors.
///
/// Implementations describe their parameter tree by writing `(name, &Array)`
/// pairs into a flat name-keyed map. Names use dot-separated paths (e.g.
/// `"layers.0.self_attn.q_proj.weight"`), matching the safetensors convention.
pub trait Module {
    /// Visit every owned parameter, prefixing its name with `prefix` and a
    /// dot if `prefix` is non-empty. Sub-modules forward to their own
    /// `parameters` with the extended prefix.
    fn parameters<'a>(&'a self, prefix: &str, out: &mut HashMap<String, &'a Array>);

    /// Replace owned parameters from a name → Array map, consuming entries
    /// the module recognizes. After loading, callers can inspect leftover
    /// keys to detect unused weights (not all checkpoints map 1:1).
    fn load(&mut self, prefix: &str, src: &mut HashMap<String, Array>) -> Result<()>;
}

/// Helper for the common pattern `parent_prefix + "." + child` (or just
/// `child` when parent is empty). Used by [`Module`] implementations to
/// build dotted parameter paths.
#[allow(dead_code)] // First consumer lands in P1 (Linear / Embedding).
pub fn join(prefix: &str, child: &str) -> String {
    if prefix.is_empty() {
        child.to_owned()
    } else {
        format!("{prefix}.{child}")
    }
}
