//! Compatibility exports for shared sampling computation.
pub(crate) use ironmlx_core::sampler::{
    draw_uniforms, prepare_target_tokens_with_uniforms_batch, prepare_uniforms,
    sample_target_tokens_with_uniforms_batch, PreparedTargetTokenSampling,
};
pub use ironmlx_core::sampler::{sample_batch, Sampler};
#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::{SamplingDistribution, SamplingTestExt};
