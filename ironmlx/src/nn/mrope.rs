//! Multimodal RoPE — Qwen3.5-style with `mrope_section`, partial rotary
//! factor, and interleaved layout.
//!
//! At P1 we only verify **construction + parameter wiring**: `inv_freq` is
//! pre-computed via `arange` + scalar ops so the shape/dtype math is exercised
//! end-to-end. The runtime methods [`Mrope::cos_sin`] and [`Mrope::apply`]
//! deliberately return `Err` — they need concrete `position_ids` shapes that
//! only Qwen3.5 model assembly (P3) provides. P3 wires real position streams
//! into attention and asserts numerical agreement against a reference.

use mlx::ops::constructors;
use mlx::{Array, Dtype};
use smallvec::SmallVec;

use crate::Result;

/// Multimodal Rotary Positional Embedding state.
///
/// Stores the precomputed `inv_freq` table and per-section rotation lengths
/// (one section per modality stream — temporal/height/width for Qwen3.5
/// vision; for text-only prompts the three streams collapse to the same
/// position id).
pub struct Mrope {
    /// Pre-computed inverse-frequency table of shape `[rot_dim/2]`.
    /// Stored once per layer; `cos`/`sin` are derived per forward.
    inv_freq: Array,
    /// Per-section rotation lengths, e.g. `[11, 11, 10]`.
    /// Sum equals `rot_dim/2`.
    sections: SmallVec<[i32; 4]>,
    /// Whether dims are interleaved (Qwen3.5: `true`) vs split-half (LLaMA: `false`).
    interleaved: bool,
    /// Number of dims actually rotated (= `head_dim * partial_rotary_factor`,
    /// rounded down to the nearest even integer).
    rot_dim: i32,
    /// Full per-head dim — the trailing `head_dim - rot_dim` channels pass
    /// through unchanged.
    head_dim: i32,
}

impl Mrope {
    /// Build an `Mrope` from the model config knobs.
    ///
    /// - `head_dim`: per-head channel count.
    /// - `theta`: RoPE base (Qwen3.5 uses `1e7`).
    /// - `partial`: fraction of `head_dim` rotated (Qwen3.5: `0.25`).
    /// - `sections`: per-stream lengths summing to `rot_dim/2`.
    /// - `interleaved`: whether even/odd channels alternate (Qwen3.5: `true`).
    pub fn new(
        head_dim: i32,
        theta: f32,
        partial: f32,
        sections: &[i32],
        interleaved: bool,
    ) -> Result<Self> {
        let rot_dim = (head_dim as f32 * partial) as i32 & !1; // even
        let half = rot_dim / 2;

        // inv_freq[i] = 1 / theta^(2i / rot_dim) for i in [0, half).
        // Compute via exp(-(2i / rot_dim) * ln(theta)) so we stay in fp32.
        let exps = constructors::arange(0.0, half as f64, 1.0, Dtype::Float32)?;
        let scale = 2.0_f32 / rot_dim as f32;
        // `&Array * T` is the panic-on-err scalar overload — fine for
        // construction-time arithmetic where any failure indicates a bug.
        let exps_scaled = &exps * scale;
        let log_theta = theta.ln();
        let x_log = &exps_scaled * log_theta;
        let theta_pow = x_log.exp()?;
        let one = constructors::ones((1,), Dtype::Float32)?;
        let inv_freq = &one / &theta_pow;

        debug_assert!(
            sections.iter().sum::<i32>() == half,
            "sections sum {} must equal half rot_dim {}",
            sections.iter().sum::<i32>(),
            half
        );

        Ok(Self {
            inv_freq,
            sections: SmallVec::from_slice(sections),
            interleaved,
            rot_dim,
            head_dim,
        })
    }

    /// Number of channels actually rotated (`head_dim * partial_rotary_factor`,
    /// rounded down to even).
    pub fn rot_dim(&self) -> i32 {
        self.rot_dim
    }

    /// Full per-head dim.
    pub fn head_dim(&self) -> i32 {
        self.head_dim
    }

    /// Per-section rotation lengths.
    pub fn sections(&self) -> &[i32] {
        self.sections.as_slice()
    }

    /// Whether the rotated channels are interleaved (`true`) or split-half
    /// (`false`).
    pub fn interleaved(&self) -> bool {
        self.interleaved
    }

    /// Pre-computed `inv_freq` table — exposed for tests / debugging.
    #[doc(hidden)]
    pub fn inv_freq(&self) -> &Array {
        &self.inv_freq
    }

    /// Compute `(cos, sin)` tables from `position_ids`.
    ///
    /// `position_ids` has shape `[3, batch, seq]`: the three streams
    /// (temporal, height, width) feed the three `mrope_section` slices.
    /// Text-only prompts pass three identical streams.
    ///
    /// Returns `(cos, sin)` each of shape `[batch, seq, rot_dim/2]`,
    /// broadcastable against Q/K halves of shape `[batch, heads, seq, rot_dim/2]`.
    ///
    /// **Stubbed at P1.** Returns `Err` — full implementation lands in P3
    /// where real position-id shapes from Qwen3.5 model assembly drive the
    /// per-section gather + interleave layout.
    pub fn cos_sin(&self, position_ids: &Array) -> Result<(Array, Array)> {
        let _ = position_ids;
        Err(anyhow::anyhow!(
            "Mrope::cos_sin not implemented at P1 — exercised in P3 model assembly"
        ))
    }

    /// Apply pre-computed cos/sin rotation to `x` (Q or K), leaving the
    /// trailing `head_dim - rot_dim` channels unchanged.
    ///
    /// **Stubbed at P1.** Returns `Err` — full implementation lands in P3.
    pub fn apply(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let _ = (x, cos, sin);
        Err(anyhow::anyhow!(
            "Mrope::apply not implemented at P1 — exercised in P3 model assembly"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrope_construction_with_qwen35_params() {
        // head_dim 256, partial 0.25 -> rot_dim 64, half 32.
        // sections [11, 11, 10] sum = 32, matches half.
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        assert_eq!(mrope.rot_dim(), 64);
        assert_eq!(mrope.head_dim(), 256);
        assert_eq!(mrope.sections(), &[11, 11, 10]);
        assert!(mrope.interleaved());
    }

    #[test]
    fn mrope_inv_freq_shape() {
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        assert_eq!(mrope.inv_freq().shape().as_slice(), &[32]);
    }
}
