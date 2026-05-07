//! Multimodal RoPE — Qwen3.5-style with `mrope_section`, partial rotary
//! factor, and interleaved layout.
//!
//! At P1 we only verify **construction + parameter wiring**: `inv_freq` is
//! pre-computed via `arange` + scalar ops so the shape/dtype math is exercised
//! end-to-end. The runtime methods [`Mrope::cos_sin`] and [`Mrope::apply`]
//! deliberately return `Err` — they need concrete `position_ids` shapes that
//! only Qwen3.5 model assembly (P3) provides. P3 wires real position streams
//! into attention and asserts numerical agreement against a reference.

use std::sync::OnceLock;

use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::ops::cast::astype;
use mlx::ops::constructors;
use mlx::ops::indexing::slice;
use mlx::ops::shape::{concatenate, expand_dims, reshape, squeeze};
use mlx::{Array, Dtype, MetalKernel};
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
    /// Lazily-built `mlx::compile`d cos/sin pipeline. Built once per
    /// instance on first `cos_sin()` call; replayed on every subsequent call.
    cos_sin_compiled: OnceLock<CompiledFn>,
    /// Lazily-built `MetalKernel` for the fused (q, k, cos, sin) -> (q', k')
    /// apply path (filled in T2).
    #[allow(dead_code)]
    apply_kernel: OnceLock<MetalKernel>,
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
            cos_sin_compiled: OnceLock::new(),
            apply_kernel: OnceLock::new(),
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

    /// Compute `(cos, sin)` rotation tables from `position_ids`.
    ///
    /// `position_ids: [n_streams, B, S]` — one stream per `mrope_section`
    /// (Qwen3.5: 3 streams = temporal/height/width; text-only prompts pass
    /// 3 identical streams).
    ///
    /// Returns `(cos: [B, S, rot_dim/2], sin: [B, S, rot_dim/2])` in fp32;
    /// caller is responsible for `astype` to the working compute dtype.
    ///
    /// First call lazily compiles the pipeline via `mlx::compile`; subsequent
    /// calls replay the optimized graph.
    pub fn cos_sin(&self, position_ids: &Array) -> Result<(Array, Array)> {
        let f = self.cos_sin_compiled.get_or_init(|| {
            self.build_cos_sin_pipeline()
                .expect("build_cos_sin_pipeline cannot fail at first call")
        });
        let mut outs = f.invoke(&[position_ids, &self.inv_freq])?;
        // CompiledFn::invoke returns a Vec<Array> in declared order.
        let sin = outs.remove(1);
        let cos = outs.remove(0);
        Ok((cos, sin))
    }

    fn build_cos_sin_pipeline(&self) -> Result<CompiledFn> {
        // Cumulative section offsets; e.g. sections=[11,11,10] -> offsets=[0,11,22,32].
        let n_streams = self.sections.len() as i32;
        let half: i32 = self.sections.iter().sum();
        let mut offsets: Vec<i32> = Vec::with_capacity(self.sections.len() + 1);
        offsets.push(0);
        let mut acc = 0_i32;
        for n in self.sections.iter() {
            acc += *n;
            offsets.push(acc);
        }

        // `move` closure captures `offsets` and `n_streams` (Copy/Vec — Send + 'static OK).
        // `inputs[0]` = position_ids [n_streams, B, S] i32
        // `inputs[1]` = inv_freq      [half] fp32
        //
        // The closure must return `mlx::Result` (not `crate::Result` / anyhow)
        // because that is the bound required by `mlx::compile::compile`.
        let pipeline = move |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let pos = inputs[0];
            let inv_freq = inputs[1];

            // 1. broadcast multiply: pos[s,b,t] * inv_freq[d]
            //    pos_f32 -> [n_streams, B, S]; expand to [n_streams, B, S, 1]
            //    inv_freq -> [half]; reshape to [1, 1, 1, half]
            let pos_f32 = astype(pos, Dtype::Float32)?;
            let pos_unsq = expand_dims(&pos_f32, &[3_i32][..])?; // [n_streams, B, S, 1]
            let inv_freq_unsq = reshape(inv_freq, &[1_i32, 1, 1, half][..])?; // [1, 1, 1, half]
                                                                              // Array * Array uses the panic-on-err operator overload (same as in new()).
            let freqs: Array = &pos_unsq * &inv_freq_unsq; // [n_streams, B, S, half]

            // 2. cos / sin (fp32)
            let cos_per_stream = freqs.cos()?;
            let sin_per_stream = freqs.sin()?;

            // 3. C-A: per-section slice + concat along last dim.
            //    For each stream s (0..n_streams), take
            //        cos_per_stream[s:s+1, :, :, offsets[s]..offsets[s+1]]
            //    then squeeze the leading stream-axis -> [B, S, sect_len].
            //    Finally concat all segments along axis -1 -> [B, S, half].
            let mut cos_segs: Vec<Array> = Vec::with_capacity(n_streams as usize);
            let mut sin_segs: Vec<Array> = Vec::with_capacity(n_streams as usize);
            for s in 0..n_streams {
                let lo = offsets[s as usize];
                let hi = offsets[s as usize + 1];

                // start = [s, 0, 0, lo], stop = [s+1, B, S, hi]
                // Use i32::MAX for the B and S dims: MLX clamps slice stops to
                // the actual dimension size, so this is equivalent to "take all"
                // without capturing a concrete runtime shape — required for
                // ShapeMode::Shapeless compatibility across variable seq lengths.
                let start = vec![s, 0_i32, 0, lo];
                let stop = vec![s + 1, i32::MAX, i32::MAX, hi];

                let cos_seg = slice(&cos_per_stream, start.as_slice(), stop.as_slice())?;
                let sin_seg = slice(&sin_per_stream, start.as_slice(), stop.as_slice())?;
                // Squeeze leading stream axis (size 1).
                let cos_seg = squeeze(&cos_seg, &[0_i32][..])?;
                let sin_seg = squeeze(&sin_seg, &[0_i32][..])?;

                cos_segs.push(cos_seg);
                sin_segs.push(sin_seg);
            }
            let cos_segs_refs: Vec<&Array> = cos_segs.iter().collect();
            let sin_segs_refs: Vec<&Array> = sin_segs.iter().collect();
            let cos = concatenate(&cos_segs_refs, -1)?;
            let sin = concatenate(&sin_segs_refs, -1)?;

            Ok(vec![cos, sin])
        };

        // ShapeMode::Fixed: re-traces when input shapes change (e.g. prefill S>>1
        // vs decode S=1). `Slice` does not implement `output_shapes` and therefore
        // cannot participate in a Shapeless graph; Fixed is the correct policy here.
        // Each distinct (B, S) pair traces once; the compiled graph is then replayed
        // for all subsequent calls with the same shape, so amortised cost is low.
        compile(pipeline, ShapeMode::Fixed).map_err(anyhow::Error::from)
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

    #[test]
    fn cos_sin_shape_and_dtype() {
        // Qwen3.5: head_dim=256, partial=0.25 -> rot_dim=64, half=32
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

        // position_ids [3, B=1, S=8] i32, three identical streams (text-only)
        let pos: Array = (
            &[
                0_i32, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
            ][..],
            (3_i32, 1, 8),
        )
            .try_into()
            .unwrap();

        let (cos, sin) = mrope.cos_sin(&pos).expect("cos_sin");

        assert_eq!(cos.shape().as_slice(), &[1, 8, 32]);
        assert_eq!(sin.shape().as_slice(), &[1, 8, 32]);
        assert_eq!(cos.dtype(), Dtype::Float32);
        assert_eq!(sin.dtype(), Dtype::Float32);
    }

    #[test]
    fn cos_sin_seq_eq_one_decode() {
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        // Decode step: position 42 across all 3 streams.
        let pos: Array = (&[42_i32, 42, 42][..], (3_i32, 1, 1)).try_into().unwrap();
        let (cos, sin) = mrope.cos_sin(&pos).expect("cos_sin seq=1");
        assert_eq!(cos.shape().as_slice(), &[1, 1, 32]);
        assert_eq!(sin.shape().as_slice(), &[1, 1, 32]);
    }
}
