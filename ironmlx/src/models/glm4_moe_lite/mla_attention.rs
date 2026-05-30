//! Absorbed Multi-head Latent Attention (MLA) for GLM-4.7-Flash
//! (`glm4_moe_lite`).
//!
//! The checkpoint stores MLA in **absorbed (matrix-folded)** form (DeepSeek-V2
//! §2.1.3): there is no `kv_b_proj`; instead `embed_q` (`W^UK^T`, 192→512 per
//! head) and `unembed_out` (`W^UV`, 512→256 per head) are stored as per-head
//! stacked 4-bit quantized weights `[H, out, in/8]`. This module provides:
//!
//! - [`PerHeadQuantLinear`] — per-head stacked quantized matmul (mirrors omlx
//!   `QuantizedMultiLinear`); the single kv-head latent broadcasts across all
//!   `H` query heads automatically via `quantized_matmul`'s batch broadcast.
//! - [`MlaAttention::project_qkv`] — the shared prefix (q/kv down+up
//!   projections, latent split + norm, decoupled RoPE) common to both the
//!   decode (`L==1`, query absorbed to latent) and prefill (`L>1`, latent
//!   un-folded per head) regimes. The two regimes are added in Task 4b.
//!
//! Mirrors `mlx_lm/models/glm4_moe_lite.py:124-148` + `mla.py`
//! `QuantizedMultiLinear` (OBS reference, not spec).

use anyhow::Result;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Linear, RmsNorm};

use super::config::Glm4MoeLiteConfig;
use super::rope::Glm4Rope;

/// Per-head stacked quantized linear (mirrors omlx `QuantizedMultiLinear`).
///
/// Weight is `[H, out, in/8]` (4-bit packed) with per-group `scales` and
/// optional affine `biases` (zero-points). The forward [`apply`](Self::apply)
/// is a single `quantized_matmul`:
///
/// - `transpose=true`  → `in → out` (decode: fold query into the latent space).
/// - `transpose=false` → `out → in` (prefill: un-fold latent per head).
///
/// A single-kv-head input `x[B,1,L,*]` broadcasts across the weight's `H`
/// heads automatically (MLX `quantized_matmul` batch broadcast — verified by
/// PROBE), so no manual head replication is needed.
pub struct PerHeadQuantLinear {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
}

impl PerHeadQuantLinear {
    /// Load `{prefix}.weight` (`[H, out, in/8]`, required), `{prefix}.scales`
    /// (required), and `{prefix}.biases` (optional affine zero-points).
    /// `group_size`/`bits` come from the loader's quantization metadata.
    ///
    /// The GLM-4.7-Flash mlx-community checkpoint quantizes `embed_q` and
    /// `unembed_out` (4-bit affine, group_size=64) — verified by PROBE against
    /// the real snapshot — so this constructor requires quantization metadata
    /// and has no fp fallback.
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let qm = loader.quant_meta_for(prefix).ok_or_else(|| {
            anyhow::anyhow!("{prefix}: expected quantized per-head weight (no quantization meta)")
        })?;
        Ok(Self {
            weight: loader.tensor(&format!("{prefix}.weight"))?.clone(),
            scales: loader.tensor(&format!("{prefix}.scales"))?.clone(),
            biases: loader.tensor_opt(&format!("{prefix}.biases")).cloned(),
            group_size: qm.group_size,
            bits: qm.bits,
        })
    }

    /// Build directly from pre-quantized arrays. Test/composition seam used by
    /// [`MlaAttention::from_parts`] and unit/integration tests.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can
    /// construct instances without a safetensors fixture; hidden from rustdoc.
    /// Mirrors the [`Linear::new_quant`] seam convention.
    #[doc(hidden)]
    pub fn from_parts(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Per-head quantized matmul. `transpose=true` maps `in→out` (decode),
    /// `transpose=false` maps `out→in` (prefill un-fold). A single-kv-head `x`
    /// broadcasts across the weight's `H` heads.
    pub fn apply(
        &self,
        x: &Array,
        transpose: bool,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Ok(mlx::quantization::quantized_matmul_on(
            x,
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            transpose,
            Some(self.group_size),
            Some(self.bits),
            "affine",
            target,
        )?)
    }
}

/// Absorbed-MLA attention for one GLM-4.7-Flash decoder layer.
///
/// Holds the down/up query projections (`q_a_proj`/`q_b_proj`), the joint
/// KV+rope down projection (`kv_a_proj_with_mqa`), the two latent layernorms,
/// the absorbed per-head projections (`embed_q`/`unembed_out`), the output
/// projection (`o_proj`), the decoupled RoPE wrapper, and the per-head dims.
pub struct MlaAttention {
    q_a_proj: Linear,
    q_b_proj: Linear,
    kv_a_proj_with_mqa: Linear,
    o_proj: Linear,
    q_a_layernorm: RmsNorm,
    kv_a_layernorm: RmsNorm,
    embed_q: PerHeadQuantLinear,
    unembed_out: PerHeadQuantLinear,
    rope: Glm4Rope,
    n_heads: i32,
    qk_nope: i32,
    qk_rope: i32,
    kv_lora: i32,
    v_head: i32,
    scale: f32,
}

impl MlaAttention {
    /// Load all submodules at `{prefix}.{name}` and derive dims + softmax scale
    /// from `cfg`.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: &Glm4MoeLiteConfig) -> Result<Self> {
        Ok(Self {
            q_a_proj: Linear::from_loader(loader, &format!("{prefix}.q_a_proj"))?,
            q_b_proj: Linear::from_loader(loader, &format!("{prefix}.q_b_proj"))?,
            kv_a_proj_with_mqa: Linear::from_loader(
                loader,
                &format!("{prefix}.kv_a_proj_with_mqa"),
            )?,
            o_proj: Linear::from_loader(loader, &format!("{prefix}.o_proj"))?,
            q_a_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.q_a_layernorm"),
                cfg.rms_norm_eps,
            )?,
            kv_a_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.kv_a_layernorm"),
                cfg.rms_norm_eps,
            )?,
            embed_q: PerHeadQuantLinear::from_loader(loader, &format!("{prefix}.embed_q"))?,
            unembed_out: PerHeadQuantLinear::from_loader(loader, &format!("{prefix}.unembed_out"))?,
            rope: Glm4Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta),
            n_heads: cfg.num_attention_heads,
            qk_nope: cfg.qk_nope_head_dim,
            qk_rope: cfg.qk_rope_head_dim,
            kv_lora: cfg.kv_lora_rank,
            v_head: cfg.v_head_dim,
            scale: cfg.softmax_scale(),
        })
    }

    /// Per-head v-projection (`unembed_out`). Consumed by Task 4b's two-regime
    /// SDPA dispatch (decode: `output @ unembed_out`; prefill: `v = unembed_out(latent)`).
    pub fn unembed_out(&self) -> &PerHeadQuantLinear {
        &self.unembed_out
    }

    /// Absorbed-MLA query projection (`embed_q`). Consumed by Task 4b
    /// (decode: `q_nope = embed_q(q_nope)`; prefill: `k = embed_q(latent, transpose=false)`).
    pub fn embed_q(&self) -> &PerHeadQuantLinear {
        &self.embed_q
    }

    /// Output projection (`o_proj`). Consumed by Task 4b after head merge.
    pub fn o_proj(&self) -> &Linear {
        &self.o_proj
    }

    /// Attention softmax scale `1/sqrt(q_head_dim)`. Consumed by Task 4b SDPA.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Per-head value dim (`v_head_dim`). Consumed by Task 4b head merge.
    pub fn v_head(&self) -> i32 {
        self.v_head
    }

    /// Shared MLA prefix common to both regimes (mirrors
    /// `glm4_moe_lite.py:132-148`).
    ///
    /// Returns `(q_nope, q_pe, c_kv_n, k_pe)` with shapes:
    /// - `q_nope`  `[B, H, S, qk_nope]`        — NoPE query (per head).
    /// - `q_pe`    `[B, H, S, qk_rope]`        — RoPE'd query (per head).
    /// - `c_kv_n`  `[B, 1, S, kv_lora]`        — normalized latent KV (single head).
    /// - `k_pe`    `[B, 1, S, qk_rope]`        — RoPE'd key (single head).
    ///
    /// `offset` is the per-row `[B]` i32 RoPE start position (cache length).
    pub fn project_qkv(
        &self,
        x: &Array,
        offset: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array, Array, Array)> {
        let target = target.into();
        let dims = x.shape();
        let s = dims.as_slice();
        let b = s[0];
        let seq = s[1];
        let q_head_dim = self.qk_nope + self.qk_rope;

        // --- Query path: q = q_b_proj(q_a_layernorm(q_a_proj(x))) ---
        let q = self.q_a_proj.forward_on(x, target)?;
        let q = self.q_a_layernorm.forward_on(&q, target)?;
        let q = self.q_b_proj.forward_on(&q, target)?;
        // [B,S,H*q_head_dim] -> [B,S,H,q_head_dim] -> [B,H,S,q_head_dim]
        let q = q
            .reshape_on((b, seq, self.n_heads, q_head_dim), target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;
        // Split last axis into nope[..:qk_nope] + pe[qk_nope:].
        let q_parts = mlx::ops::shape::split_at_on(&q, &[self.qk_nope], -1, target)?;
        let q_nope = q_parts[0].clone();
        let q_pe = q_parts[1].clone();

        // --- KV path: kv = kv_a_proj_with_mqa(x); split latent + k_pe ---
        let kv = self.kv_a_proj_with_mqa.forward_on(x, target)?;
        let kv_parts = mlx::ops::shape::split_at_on(&kv, &[self.kv_lora], -1, target)?;
        let c_kv = kv_parts[0].clone();
        let k_pe = kv_parts[1].clone();
        let c_kv_n = self.kv_a_layernorm.forward_on(&c_kv, target)?;
        // Latent: [B,S,kv_lora] -> [B,1,S,kv_lora] (single kv head).
        let c_kv_n = c_kv_n.reshape_on((b, 1_i32, seq, self.kv_lora), target)?;
        // k_pe: [B,S,qk_rope] -> [B,S,1,qk_rope] -> [B,1,S,qk_rope]
        // (reshape to [B,S,1,rope] then transpose, per omlx:141).
        let k_pe = k_pe
            .reshape_on((b, seq, 1_i32, self.qk_rope), target)?
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?;

        // --- Decoupled RoPE on the rope channels of q and k ---
        let q_pe = self.rope.apply(&q_pe, offset, target)?;
        let k_pe = self.rope.apply(&k_pe, offset, target)?;

        Ok((q_nope, q_pe, c_kv_n, k_pe))
    }

    /// Tiny test constructor from explicit arrays (dims `H=2, qk_nope=4,
    /// qk_rope=2, kv_lora=6, v_head=4`). Used by Task 4b's regime-parity tests.
    ///
    /// `pub` (not `pub(crate)`) + `#[doc(hidden)]` so integration tests can
    /// build a synthetic `MlaAttention`; mirrors the [`Linear::new_fp`] seam.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        q_a_proj: Linear,
        q_b_proj: Linear,
        kv_a_proj_with_mqa: Linear,
        o_proj: Linear,
        q_a_layernorm: RmsNorm,
        kv_a_layernorm: RmsNorm,
        embed_q: PerHeadQuantLinear,
        unembed_out: PerHeadQuantLinear,
        rope: Glm4Rope,
        n_heads: i32,
        qk_nope: i32,
        qk_rope: i32,
        kv_lora: i32,
        v_head: i32,
        scale: f32,
    ) -> Self {
        Self {
            q_a_proj,
            q_b_proj,
            kv_a_proj_with_mqa,
            o_proj,
            q_a_layernorm,
            kv_a_layernorm,
            embed_q,
            unembed_out,
            rope,
            n_heads,
            qk_nope,
            qk_rope,
            kv_lora,
            v_head,
            scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(d: &[f32], s: &[i32]) -> Array {
        (d, s).try_into().unwrap()
    }

    fn off(d: &[i32], s: &[i32]) -> Array {
        (d, s).try_into().unwrap()
    }

    /// Quantize an fp `[H, out, in]` weight (group_size = in, must be a
    /// supported size: 32/64/128) and wrap as a `PerHeadQuantLinear`.
    fn quant_per_head(w_fp: &Array, in_dim: i32) -> PerHeadQuantLinear {
        let q = mlx::quantization::quantize_on(w_fp, Some(in_dim), Some(4), "affine", None, ())
            .unwrap();
        PerHeadQuantLinear::from_parts(q[0].clone(), q[1].clone(), q.get(2).cloned(), in_dim, 4)
    }

    /// Dequantize-then-matmul reference for `transpose=true` (in→out), per head.
    fn dequant_ref_transpose_true(
        phl: &PerHeadQuantLinear,
        x: &Array,
        h: i32,
        out: i32,
        in_dim: i32,
    ) -> Vec<f32> {
        let w = mlx::quantization::dequantize(
            &phl.weight,
            &phl.scales,
            phl.biases.as_ref(),
            Some(in_dim),
            Some(4),
            "affine",
            None,
            None,
        )
        .unwrap();
        let wv = w.to_vec::<f32>().unwrap(); // [H, out, in]
        let xv = x.to_vec::<f32>().unwrap(); // [1, H, 1, in]
        let mut y = vec![0.0f32; (h * out) as usize];
        for head in 0..h as usize {
            for o in 0..out as usize {
                let mut acc = 0.0f32;
                for i in 0..in_dim as usize {
                    let wi = wv[head * out as usize * in_dim as usize + o * in_dim as usize + i];
                    let xi = xv[head * in_dim as usize + i];
                    acc += wi * xi;
                }
                y[head * out as usize + o] = acc;
            }
        }
        y
    }

    #[test]
    fn per_head_quant_transpose_true_and_false() {
        let h = 2_i32;
        let out = 3_i32;
        let in_dim = 32_i32; // group_size 32 (smallest supported)

        // Distinct per-head weights: head0 identity-ish (scale 1), head1 scale 2.
        let mut wdata = vec![0.0f32; (h * out * in_dim) as usize];
        let row = in_dim as usize;
        for (head, scale) in [(0usize, 1.0f32), (1usize, 2.0f32)] {
            for r in 0..out as usize {
                wdata[head * out as usize * row + r * row + r] = scale;
            }
        }
        let w_fp = arr(&wdata, &[h, out, in_dim]);
        let phl = quant_per_head(&w_fp, in_dim);

        // transpose=true: x[1,2,1,32] -> [1,2,1,3]
        let mut xd = vec![0.0f32; (h * in_dim) as usize];
        for c in 0..out as usize {
            xd[c] = (c + 1) as f32; // head0 channels [1,2,3,...]
            xd[in_dim as usize + c] = (c + 1) as f32; // head1
        }
        let x = arr(&xd, &[1, h, 1, in_dim]);
        let y = phl.apply(&x, true, ()).unwrap();
        assert_eq!(y.shape().as_slice(), &[1, 2, 1, 3]);
        let got = y.to_vec::<f32>().unwrap();
        let want = dequant_ref_transpose_true(&phl, &x, h, out, in_dim);
        for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-2,
                "transpose=true ch {i}: got {g}, want {e}, diff {}",
                (g - e).abs()
            );
        }

        // transpose=false un-fold: x[1,2,1,3] -> [1,2,1,32]
        let xt = arr(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[1, h, 1, out]);
        let yt = phl.apply(&xt, false, ()).unwrap();
        assert_eq!(yt.shape().as_slice(), &[1, 2, 1, in_dim]);
    }

    #[test]
    fn per_head_quant_broadcasts_single_head() {
        // Single-input-head x[1,1,1,32] with H=2 weight -> [1,2,1,3], with each
        // weight head applied (head0 scale 1, head1 scale 2).
        let h = 2_i32;
        let out = 3_i32;
        let in_dim = 32_i32;
        let mut wdata = vec![0.0f32; (h * out * in_dim) as usize];
        let row = in_dim as usize;
        for (head, scale) in [(0usize, 1.0f32), (1usize, 2.0f32)] {
            for r in 0..out as usize {
                wdata[head * out as usize * row + r * row + r] = scale;
            }
        }
        let w_fp = arr(&wdata, &[h, out, in_dim]);
        let phl = quant_per_head(&w_fp, in_dim);

        let mut xd = vec![0.0f32; in_dim as usize];
        for c in 0..out as usize {
            xd[c] = (c + 1) as f32;
        }
        let xb = arr(&xd, &[1, 1, 1, in_dim]);
        let yb = phl.apply(&xb, true, ()).unwrap();
        assert_eq!(yb.shape().as_slice(), &[1, 2, 1, 3]);
        let got = yb.to_vec::<f32>().unwrap();
        // head0 = [1,2,3], head1 = [2,4,6] (×2 weight).
        let expected = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-2,
                "broadcast ch {i}: got {g}, want {e}, diff {}",
                (g - e).abs()
            );
        }
    }

    /// Build a tiny `MlaAttention` (H=2, qk_nope=4, qk_rope=2, kv_lora=6,
    /// v_head=4) from synthetic fp/quantized parts for shape tests.
    fn tiny_mla() -> MlaAttention {
        let h = 2_i32;
        let qk_nope = 4_i32;
        let qk_rope = 2_i32;
        let kv_lora = 6_i32;
        let v_head = 4_i32;
        let q_head_dim = qk_nope + qk_rope; // 6
        let q_lora = 8_i32;
        let hidden = 10_i32;

        // q_a_proj: hidden(10) -> q_lora(8). FP identity-ish (zeros ok for shape).
        let q_a_w = arr(&vec![0.1f32; (q_lora * hidden) as usize], &[q_lora, hidden]);
        let q_a_proj = Linear::new_fp(q_a_w, None);
        // q_b_proj: q_lora(8) -> H*q_head_dim (2*6=12).
        let q_b_out = h * q_head_dim;
        let q_b_w = arr(
            &vec![0.1f32; (q_b_out * q_lora) as usize],
            &[q_b_out, q_lora],
        );
        let q_b_proj = Linear::new_fp(q_b_w, None);
        // kv_a_proj_with_mqa: hidden(10) -> kv_lora+qk_rope (6+2=8).
        let kv_out = kv_lora + qk_rope;
        let kv_w = arr(&vec![0.1f32; (kv_out * hidden) as usize], &[kv_out, hidden]);
        let kv_a_proj_with_mqa = Linear::new_fp(kv_w, None);
        // o_proj: H*v_head (2*4=8) -> hidden(10).
        let o_w = arr(
            &vec![0.1f32; (hidden * h * v_head) as usize],
            &[hidden, h * v_head],
        );
        let o_proj = Linear::new_fp(o_w, None);

        // Layernorms (plain weight, all ones).
        let q_a_layernorm = RmsNorm::new(arr(&vec![1.0f32; q_lora as usize], &[q_lora]), 1e-5);
        let kv_a_layernorm = RmsNorm::new(arr(&vec![1.0f32; kv_lora as usize], &[kv_lora]), 1e-5);

        // embed_q: per head qk_nope(4) -> kv_lora(6); quantized, in=qk_nope must
        // be a supported group_size. qk_nope=4 is NOT supported, so pad the test
        // weight to in=32 by quantizing a [H, kv_lora, 32] weight and noting the
        // shape test only exercises project_qkv (which does not call embed_q).
        // We still need a constructible PerHeadQuantLinear; build one whose
        // shapes are irrelevant to project_qkv.
        let dummy_in = 32_i32;
        let eq_w = arr(
            &vec![0.05f32; (h * kv_lora * dummy_in) as usize],
            &[h, kv_lora, dummy_in],
        );
        let embed_q = quant_per_head(&eq_w, dummy_in);
        let uo_w = arr(
            &vec![0.05f32; (h * v_head * dummy_in) as usize],
            &[h, v_head, dummy_in],
        );
        let unembed_out = quant_per_head(&uo_w, dummy_in);

        let rope = Glm4Rope::new(qk_rope, 10000.0);
        let scale = 1.0 / (q_head_dim as f32).sqrt();

        MlaAttention::from_parts(
            q_a_proj,
            q_b_proj,
            kv_a_proj_with_mqa,
            o_proj,
            q_a_layernorm,
            kv_a_layernorm,
            embed_q,
            unembed_out,
            rope,
            h,
            qk_nope,
            qk_rope,
            kv_lora,
            v_head,
            scale,
        )
    }

    #[test]
    fn shared_prefix_shapes() {
        let mla = tiny_mla();
        let hidden = 10_i32;
        let seq = 3_i32;
        let x = arr(&vec![0.2f32; (seq * hidden) as usize], &[1, seq, hidden]);
        let offset = off(&[0], &[1]);
        let (q_nope, q_pe, c_kv_n, k_pe) = mla.project_qkv(&x, &offset, ()).unwrap();
        assert_eq!(q_nope.shape().as_slice(), &[1, 2, 3, 4], "q_nope");
        assert_eq!(q_pe.shape().as_slice(), &[1, 2, 3, 2], "q_pe");
        assert_eq!(c_kv_n.shape().as_slice(), &[1, 1, 3, 6], "c_kv_n");
        assert_eq!(k_pe.shape().as_slice(), &[1, 1, 3, 2], "k_pe");
    }
}
