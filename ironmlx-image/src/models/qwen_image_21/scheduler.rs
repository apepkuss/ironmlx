use anyhow::anyhow;

use crate::Result;

#[derive(Debug, Clone)]
pub struct FlowMatchSchedule {
    pub timesteps: Vec<f32>,
    /// One more entry than `timesteps`; the final value is zero.
    pub sigmas: Vec<f32>,
}

impl FlowMatchSchedule {
    pub fn qwen_image_21(image_sequence_length: usize, inference_steps: usize) -> Result<Self> {
        if image_sequence_length == 0 {
            return Err(anyhow!("image sequence length must be positive"));
        }
        if !(2..=200).contains(&inference_steps) {
            return Err(anyhow!(
                "Qwen Image 2.1 inference_steps must be in 2..=200, got {inference_steps}"
            ));
        }

        // scheduler_config.json: dynamic exponential shifting from base
        // (256, 0.5) to max (8192, 0.9), stretched to terminal sigma 0.02.
        let sequence = image_sequence_length as f32;
        let slope = (0.9_f32 - 0.5_f32) / (8192.0_f32 - 256.0_f32);
        let mu = slope * sequence + (0.5_f32 - slope * 256.0_f32);
        let exp_mu = mu.exp();
        let mut sigmas = Vec::with_capacity(inference_steps + 1);
        for index in 0..inference_steps {
            let fraction = index as f32 / (inference_steps - 1) as f32;
            let linear = 1.0 - fraction * (1.0 - 1.0 / inference_steps as f32);
            sigmas.push(exp_mu / (exp_mu + (1.0 / linear - 1.0)));
        }
        let terminal = 0.02_f32;
        let scale = (1.0 - sigmas[inference_steps - 1]) / (1.0 - terminal);
        for sigma in &mut sigmas {
            *sigma = 1.0 - (1.0 - *sigma) / scale;
        }
        let timesteps = sigmas.iter().map(|sigma| sigma * 1000.0).collect();
        sigmas.push(0.0);
        Ok(Self { timesteps, sigmas })
    }

    pub fn delta(&self, step: usize) -> Result<f32> {
        let current = *self
            .sigmas
            .get(step)
            .ok_or_else(|| anyhow!("scheduler step {step} is out of range"))?;
        let next = *self
            .sigmas
            .get(step + 1)
            .ok_or_else(|| anyhow!("scheduler step {step} has no next sigma"))?;
        Ok(next - current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_image_schedule_is_monotonic_and_ends_at_zero() {
        let schedule = FlowMatchSchedule::qwen_image_21(1024, 50).unwrap();
        assert_eq!(schedule.timesteps.len(), 50);
        assert_eq!(schedule.sigmas.len(), 51);
        assert_eq!(schedule.sigmas[0], 1.0);
        assert!((schedule.sigmas[49] - 0.02).abs() < 1e-6);
        assert_eq!(schedule.sigmas[50], 0.0);
        assert!(schedule.sigmas.windows(2).all(|pair| pair[1] < pair[0]));
        assert!(schedule.delta(0).unwrap() < 0.0);
    }

    #[test]
    fn rejects_degenerate_one_step_schedule() {
        assert!(FlowMatchSchedule::qwen_image_21(1024, 1).is_err());
    }
}
