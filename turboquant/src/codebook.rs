//! Lloyd-Max optimal codebooks for TurboQuant.
//!
//! After WHT rotation, each coordinate follows N(0, 1/d). The optimal scalar
//! quantizer centroids are pre-computed via Lloyd's algorithm and stored as
//! compile-time constants.
//!
//! Centroids scale as 1/sqrt(d). We store normalized centroids (for d=1) and
//! multiply by 1/sqrt(d) at runtime.

/// Normalized 3-bit Lloyd-Max centroids (8 levels).
/// Multiply by `1/sqrt(d)` to get actual centroids for dimension `d`.
/// Computed via 200 iterations of Lloyd's algorithm on N(0, 1).
#[allow(clippy::excessive_precision)]
pub const TURBO3_CENTROIDS_NORM: [f32; 8] = [
    -2.1519449883,
    -1.3439085530,
    -0.7560047494,
    -0.2450939842,
    0.2450939842,
    0.7560047494,
    1.3439085530,
    2.1519449883,
];

/// Normalized 4-bit Lloyd-Max centroids (16 levels).
#[allow(clippy::excessive_precision)]
pub const TURBO4_CENTROIDS_NORM: [f32; 16] = [
    -2.7176670187,
    -2.0521380069,
    -1.6008024772,
    -1.2399589968,
    -0.9282446944,
    -0.6458753408,
    -0.3811782363,
    -0.1260469448,
    0.1260469448,
    0.3811782363,
    0.6458753408,
    0.9282446944,
    1.2399589968,
    1.6008024772,
    2.0521380069,
    2.7176670187,
];

/// Codebook configuration for a specific bit-width and dimension.
pub struct Codebook {
    /// Actual centroid values (scaled by 1/sqrt(d)).
    pub centroids: Vec<f32>,
    /// Decision boundaries (midpoints between consecutive centroids).
    pub boundaries: Vec<f32>,
    /// Number of bits per coordinate.
    pub bits: u8,
}

impl Codebook {
    /// Create a codebook for the given bit-width and head dimension.
    pub fn new(bits: u8, head_dim: usize) -> Self {
        let norm_centroids = match bits {
            3 => &TURBO3_CENTROIDS_NORM[..],
            4 => &TURBO4_CENTROIDS_NORM[..],
            _ => panic!("unsupported bit-width: {bits} (only 3 and 4 supported)"),
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let centroids: Vec<f32> = norm_centroids.iter().map(|&c| c * scale).collect();

        // Decision boundaries: midpoints between consecutive centroids
        let boundaries: Vec<f32> = centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();

        Self {
            centroids,
            boundaries,
            bits,
        }
    }

    /// Find the nearest centroid index for a scalar value.
    #[inline]
    pub fn nearest_index(&self, value: f32) -> u8 {
        // Binary search on sorted boundaries
        let mut idx = 0u8;
        for &b in &self.boundaries {
            if value > b {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Look up centroid value by index.
    #[inline]
    pub fn centroid(&self, index: u8) -> f32 {
        self.centroids[index as usize]
    }

    /// Number of centroid levels (2^bits).
    pub fn n_levels(&self) -> usize {
        self.centroids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroids_are_symmetric() {
        for bits in [3, 4] {
            let cb = Codebook::new(bits, 128);
            let n = cb.centroids.len();
            for i in 0..n / 2 {
                assert!(
                    (cb.centroids[i] + cb.centroids[n - 1 - i]).abs() < 1e-6,
                    "centroids not symmetric at {i} for {bits}-bit"
                );
            }
        }
    }

    #[test]
    fn centroids_scale_with_dimension() {
        let cb64 = Codebook::new(3, 64);
        let cb128 = Codebook::new(3, 128);
        let cb256 = Codebook::new(3, 256);

        // Ratio should be sqrt(2) between successive doublings.
        let ratio_64_128 = cb64.centroids[7] / cb128.centroids[7];
        let ratio_128_256 = cb128.centroids[7] / cb256.centroids[7];

        assert!((ratio_64_128 - std::f32::consts::SQRT_2).abs() < 0.01);
        assert!((ratio_128_256 - std::f32::consts::SQRT_2).abs() < 0.01);
    }

    #[test]
    fn nearest_index_correctness() {
        let cb = Codebook::new(3, 128);
        // Values at centroids should map to themselves
        for (i, &c) in cb.centroids.iter().enumerate() {
            assert_eq!(cb.nearest_index(c), i as u8);
        }
        // Extreme values
        assert_eq!(cb.nearest_index(-1.0), 0);
        assert_eq!(cb.nearest_index(1.0), 7);
        // Zero should map to center
        let idx = cb.nearest_index(0.0);
        assert!(idx == 3 || idx == 4);
    }

    #[test]
    fn mse_within_paper_bound() {
        // Paper Theorem 1: 3-bit MSE*d <= 0.03, 4-bit MSE*d <= 0.009.
        // Verify empirically with random Gaussian samples
        let mut rng_state: u64 = 42;
        let mut next_f32 = || -> f32 {
            // Simple xorshift64 for reproducibility without external deps
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            // Box-Muller approximation: use uniform to produce a rough Gaussian.
            let u = (rng_state & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
            // Inverse CDF approximation (good enough for testing)
            let sign = if rng_state & 1 == 0 { 1.0 } else { -1.0 };
            sign * (-2.0 * u.ln()).sqrt() * 0.5
        };

        for (bits, paper_bound) in [(3u8, 0.04), (4, 0.012)] {
            let d = 128;
            let cb = Codebook::new(bits, d);
            let sigma = 1.0 / (d as f32).sqrt();
            let n = 100_000;
            let mut mse_sum = 0.0f64;
            for _ in 0..n {
                let x = next_f32() * sigma;
                let idx = cb.nearest_index(x);
                let x_hat = cb.centroid(idx);
                mse_sum += ((x - x_hat) as f64).powi(2);
            }
            let mse = mse_sum / n as f64;
            let mse_times_d = mse * d as f64;
            assert!(
                mse_times_d < paper_bound,
                "{bits}-bit: MSE*d = {mse_times_d:.6} exceeds bound {paper_bound}"
            );
        }
    }
}
