//! Compatibility exports for shared sampling computation.
pub(crate) use ironmlx_core::sampler::{draw_uniforms, sample_target_tokens_with_uniforms_batch};
pub use ironmlx_core::sampler::{sample_batch, Sampler};
#[cfg(test)]
pub(crate) use test_support::{SamplingDistribution, SamplingTestExt};

#[cfg(test)]
mod test_support;
