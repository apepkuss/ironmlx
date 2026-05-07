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
    /// apply path.
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
        // CompiledFn::invoke returns a Vec<Array> in the order the closure
        // pushed them: [cos, sin]. Pop from the back to avoid index shifts.
        let sin = outs.pop().expect("pipeline returned cos+sin");
        let cos = outs.pop().expect("pipeline returned cos+sin");
        Ok((cos, sin))
    }

    /// Build the `mlx::compile`d cos/sin pipeline. Captures the cumulative
    /// section offsets at compile time (model constants) into a `move`
    /// closure, then traces:
    ///
    ///   `pos × inv_freq → cos / sin → per-stream slice + squeeze + concat`
    ///
    /// Uses `ShapeMode::Fixed` (re-traces per distinct `(B, S)`) because
    /// MLX's `Slice` primitive lacks `output_shapes` inference, blocking
    /// `Shapeless`. Called once per `Mrope` instance from `cos_sin`'s
    /// `OnceLock::get_or_init`.
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
                // Use i32::MAX for B and S stops: MLX clamps slice stops to the
                // actual dim size at runtime, so this is equivalent to "take all"
                // without baking a concrete (B, S) into the Fixed-mode trace —
                // lets the trace stay reusable across distinct shapes within the
                // same Fixed-shape cache entry.
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
            let cos = concatenate(&cos_segs.iter().collect::<Vec<&Array>>(), -1)?;
            let sin = concatenate(&sin_segs.iter().collect::<Vec<&Array>>(), -1)?;

            Ok(vec![cos, sin])
        };

        // ShapeMode::Fixed: re-traces when input shapes change (e.g. prefill S>>1
        // vs decode S=1). `Slice` does not implement `output_shapes` and therefore
        // cannot participate in a Shapeless graph; Fixed is the correct policy here.
        // Each distinct (B, S) pair traces once; the compiled graph is then replayed
        // for all subsequent calls with the same shape, so amortised cost is low.
        compile(pipeline, ShapeMode::Fixed).map_err(anyhow::Error::from)
    }

    /// Apply rotary rotation to Q and K in a single fused dispatch.
    ///
    /// `q: [B, Hq, S, HEAD_DIM]`, `k: [B, Hkv, S, HEAD_DIM]`,
    /// `cos: [B, S, ROTARY_DIM/2]` (fp32), `sin: [B, S, ROTARY_DIM/2]` (fp32).
    ///
    /// Returns `(q_rot, k_rot)` with the same shape and dtype as their inputs.
    /// The trailing `HEAD_DIM - ROTARY_DIM` channels pass through unchanged.
    pub fn apply(&self, q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
        // Apply kernel currently supports only interleaved layout (Qwen3.5).
        // If a future caller constructs Mrope with `interleaved=false`
        // (LLaMA-style split-half), this guard fires loudly rather than
        // silently producing wrong results — the shader formula is hardcoded
        // interleaved.
        if !self.interleaved {
            return Err(anyhow::anyhow!(
                "Mrope::apply: split-half (interleaved=false) layout is not implemented; only interleaved=true is supported (Qwen3.5)"
            ));
        }

        // Sanity (cheap; full validation is at MLX dispatch boundaries).
        let q_shape = q.shape();
        let k_shape = k.shape();
        let q_dims = q_shape.as_slice();
        let k_dims = k_shape.as_slice();
        if q_dims.len() != 4 || k_dims.len() != 4 {
            return Err(anyhow::anyhow!(
                "Mrope::apply expects rank-4 q/k; got q.ndim={}, k.ndim={}",
                q_dims.len(),
                k_dims.len()
            ));
        }
        if q_dims[3] != self.head_dim || k_dims[3] != self.head_dim {
            return Err(anyhow::anyhow!(
                "Mrope::apply: q.head_dim={} k.head_dim={} != configured {}",
                q_dims[3],
                k_dims[3],
                self.head_dim
            ));
        }

        // The Metal shader reads B and S from q_shape only and applies them
        // to k's stride math. Consistency check: if Q and K disagree on B or
        // S, the K addressing is silently wrong.
        if q_dims[0] != k_dims[0] {
            return Err(anyhow::anyhow!(
                "Mrope::apply: q.batch={} != k.batch={}",
                q_dims[0],
                k_dims[0]
            ));
        }
        if q_dims[2] != k_dims[2] {
            return Err(anyhow::anyhow!(
                "Mrope::apply: q.seq={} != k.seq={}",
                q_dims[2],
                k_dims[2]
            ));
        }

        let b = q_dims[0];
        let hq = q_dims[1];
        let hkv = k_dims[1];
        let s = q_dims[2];

        let kernel = self.apply_kernel.get_or_init(|| {
            self.build_apply_kernel()
                .expect("build_apply_kernel cannot fail at first call")
        });

        // Grid: cover (B*(Hq+Hkv)) × S × HEAD_DIM elements; one thread per element.
        let grid_x = b * (hq + hkv);
        let grid_y = s;
        let grid_z = self.head_dim;
        // Threadgroup: 1 thread on the (qk_head, t) axes; HEAD_DIM threads on the d axis.
        // HEAD_DIM=256 fits within Metal's 1024-thread threadgroup limit.
        let tg_x = 1;
        let tg_y = 1;
        let tg_z = self.head_dim;

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[q, k, cos, sin])
            .output_shapes(&[q.shape().clone(), k.shape().clone()])
            .output_dtypes(&[q.dtype(), k.dtype()])
            .grid(grid_x, grid_y, grid_z)
            .threadgroup(tg_x, tg_y, tg_z)
            .template_int("HEAD_DIM", self.head_dim)
            .template_int("ROTARY_DIM", self.rot_dim)
            .dispatch()?;

        let q_rot = outputs.take_at(0)?;
        let k_rot = outputs.take_at(0)?; // erase-and-shift: K shifts to slot 0
        Ok((q_rot, k_rot))
    }

    /// Lazily build the fused Q+K rotary `MetalKernel`. Templated on
    /// `HEAD_DIM` and `ROTARY_DIM` so Metal's compiler unrolls the rotate
    /// loop and folds index arithmetic. MLX auto-injects `q_shape` / `k_shape`
    /// buffers when the source references them (see
    /// `/Volumes/Dev/mlx/mlx/backend/metal/custom_kernel.cpp:93-105,190-192`).
    fn build_apply_kernel(&self) -> Result<mlx::MetalKernel> {
        // Metal shader. Templates: HEAD_DIM, ROTARY_DIM. ROT_PAIRS = ROTARY_DIM/2.
        //
        // Each thread handles one element of (Q or K) at indices (b, head, t, d).
        // The first grid dim (qk_head) ranges over B*(Hq+Hkv): the lower B*Hq
        // values address Q; the upper B*Hkv address K. Hq, Hkv, B, S are pulled
        // from the input shape buffers (auto-injected by MLX when the source
        // references `<name>_shape`).
        let src = r#"
        constexpr uint ROT_PAIRS = ROTARY_DIM / 2;

        uint qk_head = thread_position_in_grid.x;
        uint t       = thread_position_in_grid.y;
        uint d       = thread_position_in_grid.z;

        uint B   = (uint)q_shape[0];
        uint Hq  = (uint)q_shape[1];
        uint S   = (uint)q_shape[2];
        uint Hkv = (uint)k_shape[1];

        // Decode (b, head, is_q)
        bool is_q;
        uint b;
        uint h;
        if (qk_head < B * Hq) {
            is_q = true;
            b = qk_head / Hq;
            h = qk_head % Hq;
        } else {
            is_q = false;
            uint k_head_flat = qk_head - B * Hq;
            b = k_head_flat / Hkv;
            h = k_head_flat % Hkv;
        }

        uint H = is_q ? Hq : Hkv;
        // Row-major (B, H, S, HEAD_DIM):
        uint base = ((b * H + h) * S + t) * HEAD_DIM;

        // cos/sin: row-major (B, S, ROT_PAIRS), broadcast across heads.
        uint cs_idx = (b * S + t) * ROT_PAIRS;

        if (d < ROTARY_DIM) {
            // Interleaved: pair (2p, 2p+1) shares cos[p], sin[p].
            uint p = d >> 1;
            bool is_even = (d & 1u) == 0u;

            float c = cos[cs_idx + p];
            float si = sin[cs_idx + p];

            if (is_q) {
                float x_self = float(q[base + d]);
                float x_pair = float(q[base + (is_even ? d + 1 : d - 1)]);
                float rotated = is_even
                    ? (x_self * c - x_pair * si)
                    : (x_pair * si + x_self * c);
                q_out[base + d] = static_cast<__typeof__(*q)>(rotated);
            } else {
                float x_self = float(k[base + d]);
                float x_pair = float(k[base + (is_even ? d + 1 : d - 1)]);
                float rotated = is_even
                    ? (x_self * c - x_pair * si)
                    : (x_pair * si + x_self * c);
                k_out[base + d] = static_cast<__typeof__(*k)>(rotated);
            }
        } else {
            // Pass-through tail (HEAD_DIM - ROTARY_DIM channels).
            if (is_q) {
                q_out[base + d] = q[base + d];
            } else {
                k_out[base + d] = k[base + d];
            }
        }
        "#;

        Ok(mlx::MetalKernel::builder("ironmlx_mrope_apply_qk")
            .inputs(&["q", "k", "cos", "sin"])
            .outputs(&["q_out", "k_out"])
            .source(src)
            .ensure_row_contiguous(true)
            .atomic_outputs(false)
            .build()?)
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

    #[test]
    fn apply_shape_and_dtype_fp32() {
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

        // Q [B=1, Hq=64, S=4, head_dim=256], K [B=1, Hkv=8, S=4, head_dim=256]
        // Use small S=4 to keep the test fast.
        let q = Array::zeros((1_i32, 64, 4, 256), Dtype::Float32).unwrap();
        let k = Array::zeros((1_i32, 8, 4, 256), Dtype::Float32).unwrap();
        let cos = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");

        assert_eq!(q_rot.shape().as_slice(), &[1, 64, 4, 256]);
        assert_eq!(k_rot.shape().as_slice(), &[1, 8, 4, 256]);
        assert_eq!(q_rot.dtype(), Dtype::Float32);
        assert_eq!(k_rot.dtype(), Dtype::Float32);
    }

    #[test]
    fn apply_shape_and_dtype_bf16() {
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        let q = Array::zeros((1_i32, 64, 4, 256), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 8, 4, 256), Dtype::Bfloat16).unwrap();
        // cos/sin always fp32 (per spec § 3.1).
        let cos = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply bf16");

        assert_eq!(q_rot.dtype(), Dtype::Bfloat16);
        assert_eq!(k_rot.dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn apply_partial_rotary_tail_unchanged() {
        // head_dim=256, partial=0.25 -> rot_dim=64. Tail [64..256) must be unchanged.
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();

        // Distinct integer values per element so we can spot any unintended mutation.
        // Q shape [1, 1, 1, 256] with values 0..256 (fp32).
        let q_data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let q: Array = (q_data.as_slice(), (1_i32, 1, 1, 256)).try_into().unwrap();
        let k: Array = (q_data.as_slice(), (1_i32, 1, 1, 256)).try_into().unwrap();

        // cos = ones, sin = zeros: rotation is identity on rotated dims;
        // tail dims must also stay unchanged.
        let cos = constructors::ones((1_i32, 1, 32), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 1, 32), Dtype::Float32).unwrap();

        let (q_rot, _k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");
        let rot_data: Vec<f32> = q_rot.to_vec().unwrap();

        // Tail must be byte-identical to input.
        for d in 64..256 {
            assert_eq!(rot_data[d], q_data[d], "tail channel {d} mutated");
        }
        // Rotated dims with cos=1 sin=0: identity for both even and odd halves.
        // even idx: x_even * 1 - x_odd * 0 = x_even
        // odd idx:  x_even * 0 + x_odd * 1 = x_odd
        for d in 0..64 {
            assert_eq!(
                rot_data[d], q_data[d],
                "rotated channel {d} not identity under cos=1,sin=0"
            );
        }
    }

    #[test]
    fn apply_interleaved_pair_known_rotation() {
        // Build a tiny mrope where head_dim=4, rot_dim=4, sections=[1,1,0]
        // (or any sections summing to half=2). Manual values let us check
        // the rotation formula bit-exactly.
        let mrope = Mrope::new(4, 10000.0, 1.0, &[1, 1, 0], true).unwrap();

        // Q data: [1, 2, 3, 4] shaped [1, 1, 1, 4] (B=1, H=1, S=1, head_dim=4)
        let q: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1, 1, 4))
            .try_into()
            .unwrap();
        let k: Array = (&[10.0_f32, 20.0, 30.0, 40.0][..], (1_i32, 1, 1, 4))
            .try_into()
            .unwrap();

        // cos = [c0, c1], sin = [s0, s1] for the 2 pairs.
        // Use cos = [0, 1], sin = [1, 0]:
        //   pair 0 (channels 0,1): cos=0, sin=1
        //     y[0] = x[0]*0 - x[1]*1 = -x[1] = -2
        //     y[1] = x[0]*1 + x[1]*0 = x[0]  =  1
        //   pair 1 (channels 2,3): cos=1, sin=0  (identity)
        //     y[2] = x[2] = 3
        //     y[3] = x[3] = 4
        let cos: Array = (&[0.0_f32, 1.0][..], (1_i32, 1, 2)).try_into().unwrap();
        let sin: Array = (&[1.0_f32, 0.0][..], (1_i32, 1, 2)).try_into().unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply");
        let q_out: Vec<f32> = q_rot.to_vec().unwrap();
        let k_out: Vec<f32> = k_rot.to_vec().unwrap();

        assert_eq!(q_out, vec![-2.0, 1.0, 3.0, 4.0]);
        assert_eq!(k_out, vec![-20.0, 10.0, 30.0, 40.0]);
    }

    #[test]
    fn apply_gqa_different_q_kv_heads() {
        // Qwen3.5-style GQA: Hq=64, Hkv=8.
        let mrope = Mrope::new(256, 1e7, 0.25, &[11, 11, 10], true).unwrap();
        let q = Array::zeros((1_i32, 64, 2, 256), Dtype::Float32).unwrap();
        let k = Array::zeros((1_i32, 8, 2, 256), Dtype::Float32).unwrap();
        let cos = Array::zeros((1_i32, 2, 32), Dtype::Float32).unwrap();
        let sin = Array::zeros((1_i32, 2, 32), Dtype::Float32).unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &cos, &sin).expect("apply gqa");
        // Both must produce the right shape under GQA.
        assert_eq!(q_rot.shape().as_slice(), &[1, 64, 2, 256]);
        assert_eq!(k_rot.shape().as_slice(), &[1, 8, 2, 256]);
    }
}
