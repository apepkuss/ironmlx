//! Model-side reference algebra for speculative verification tests.
use crate::Result;
use ironmlx_core::sampler::{test_support::Distribution, Sampler};
use mlx::Array;

#[derive(Debug, Clone)]
pub(crate) struct SamplingDistribution(Distribution);
impl SamplingDistribution {
    pub(crate) fn new(probs: Vec<f32>) -> Result<Self> {
        Distribution::new(probs).map(Self)
    }
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
    pub(crate) fn probability(&self, token: u32) -> f32 {
        self.0.probability(token)
    }
    pub(crate) fn probabilities(&self) -> Vec<f32> {
        self.0.probabilities()
    }
    pub(crate) fn sample_with_uniform(&self, uniform: f32) -> Result<u32> {
        self.0.sample_with_uniform(uniform)
    }
    pub(crate) fn acceptance_probability(&self, draft: &Self, token: u32) -> Result<f32> {
        anyhow::ensure!(
            self.len() == draft.len(),
            "target vocabulary {} != draft vocabulary {}",
            self.len(),
            draft.len()
        );
        let p = self.probability(token);
        let q = draft.probability(token);
        anyhow::ensure!(
            q > 0.0,
            "sampled draft token {token} has zero proposal probability"
        );
        Ok((p / q).clamp(0.0, 1.0))
    }
    pub(crate) fn residual(&self, draft: &Self) -> Result<Self> {
        anyhow::ensure!(
            self.len() == draft.len(),
            "target vocabulary {} != draft vocabulary {}",
            self.len(),
            draft.len()
        );
        Self::new(
            self.probabilities()
                .iter()
                .enumerate()
                .map(|(token, &p)| (p - draft.probability(token as u32)).max(0.0))
                .collect(),
        )
    }
    pub(crate) fn residual_point_mass(&self, token: u32) -> Result<Self> {
        anyhow::ensure!(
            (token as usize) < self.len(),
            "point-mass token {token} is outside sampling vocabulary {}",
            self.len()
        );
        Self::new(
            self.probabilities()
                .iter()
                .enumerate()
                .map(|(index, &p)| {
                    if index == token as usize {
                        (p - 1.0).max(0.0)
                    } else {
                        p
                    }
                })
                .collect(),
        )
    }
}
pub(crate) trait SamplingTestExt {
    fn distributions(
        &self,
        logits: &Array,
        histories: &[&[u32]],
    ) -> Result<Vec<SamplingDistribution>>;
}
impl SamplingTestExt for Sampler {
    fn distributions(
        &self,
        logits: &Array,
        histories: &[&[u32]],
    ) -> Result<Vec<SamplingDistribution>> {
        ironmlx_core::sampler::test_support::distributions(self, logits, histories)
            .map(|rows| rows.into_iter().map(SamplingDistribution).collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_sampling_residual_recovers_target_marginal() {
        let target = SamplingDistribution::new(vec![0.6, 0.3, 0.1]).unwrap();
        let draft = SamplingDistribution::new(vec![0.2, 0.7, 0.1]).unwrap();
        let residual = target.residual(&draft).unwrap();
        let rejection_mass = (0..target.len())
            .map(|token| {
                let token = token as u32;
                draft.probability(token)
                    * (1.0 - target.acceptance_probability(&draft, token).unwrap())
            })
            .sum::<f32>();
        let mut output = vec![0.0_f32; target.len()];
        for (token, output_prob) in output.iter_mut().enumerate() {
            let token = token as u32;
            *output_prob = draft.probability(token)
                * target.acceptance_probability(&draft, token).unwrap()
                + rejection_mass * residual.probability(token);
        }

        for (&actual, expected) in output.iter().zip(target.probabilities()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn exact_sampling_rejects_impossible_sampled_draft_token() {
        let target = SamplingDistribution::new(vec![0.5, 0.5]).unwrap();
        let draft = SamplingDistribution::new(vec![1.0, 0.0]).unwrap();
        let err = target
            .acceptance_probability(&draft, 1)
            .expect_err("zero-q sampled token must fail");
        assert!(err.to_string().contains("zero proposal probability"));
    }

    #[test]
    fn exact_sampling_rejects_empty_residual() {
        let target = SamplingDistribution::new(vec![0.5, 0.5]).unwrap();
        let draft = SamplingDistribution::new(vec![0.5, 0.5]).unwrap();
        let err = target
            .residual(&draft)
            .expect_err("identical distributions have no rejection residual");
        assert!(err.to_string().contains("no finite positive mass"));
    }
}
