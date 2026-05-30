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
use super::mla_cache::MlaLatentCache;
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

    /// Dequantize the stored weight back to full precision `[H, out, in]`.
    ///
    /// `#[doc(hidden)] pub` test/diagnostic seam: lets the Task 4b equivalence
    /// gate compute an fp-exact reference for the fold/un-fold algebra that
    /// bypasses the `quantized_matmul` transpose=true vs transpose=false kernel
    /// asymmetry (the two kernels reduce in different float orders).
    #[doc(hidden)]
    pub fn dequant_weight(&self) -> Result<Array> {
        Ok(mlx::quantization::dequantize(
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            Some(self.group_size),
            Some(self.bits),
            "affine",
            None,
            None,
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

    /// Two-regime absorbed-MLA forward (mirrors `glm4_moe_lite.py:124-174`).
    ///
    /// Dispatches on the query sequence length `L = x.shape()[1]`:
    /// - **Decode (`L == 1`)**: the query is folded into the latent space
    ///   (`q_lat = embed_q(q_nope)`), attention runs against the cached latent
    ///   (`K = V = kv_latent`), and the per-head value un-fold (`unembed_out`)
    ///   is applied AFTER softmax. This is the absorbed form.
    /// - **Prefill (`L > 1`)**: the cached latent is un-folded per head into
    ///   keys (`embed_q(latent, transpose=false)`) and values
    ///   (`unembed_out(latent)`) BEFORE attention. This is the standard form.
    ///
    /// The two regimes are algebraically identical attention; only the order of
    /// the linear maps relative to softmax differs (Task 4b equivalence gate).
    ///
    /// `pe_scores = (q_pe * scale) @ k_pe_allᵀ` carries the decoupled-RoPE
    /// contribution to the logits AND the engine mask. SDPA is invoked with
    /// `mask_mode = "array"` (`mask_arr = Some(&pe_scores)`); the `"causal"`
    /// mode is intentionally unused — all masking is folded into `pe_scores`.
    ///
    /// **Mask convention (ADDITIVE float, ironmlx engine):** the engine emits a
    /// `[B, 1, L, Lc]`-broadcastable additive mask (`0` for valid, `-inf` for
    /// blocked), matching `nn::Attention`/`GatedAttention` (`attention.rs:107`).
    /// We therefore fold it as `pe_scores = pe_scores + mask`, NOT via the
    /// boolean `mx.where(...)` of omlx:154-159 (omlx feeds a boolean mask; the
    /// ironmlx scheduler does not). For decode the engine typically passes
    /// `mask = None` (the single query sees the whole valid cache).
    ///
    /// `offset` is the per-row `[B]` i32 RoPE start position; `per_row_lens` is
    /// the per-row number of new tokens written this step.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        offset: &Array,
        cache: &mut MlaLatentCache,
        per_row_lens: &[i32],
        mask: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let dims = x.shape();
        let s = dims.as_slice();
        let b = s[0];
        let l = s[1];

        // Shared prefix: q/kv down+up projections, latent split + norm, RoPE.
        let (q_nope, q_pe, c_kv_n, k_pe) = self.project_qkv(x, offset, target)?;

        // Append the new latent + k_pe and fetch the full history.
        // kv_latent: [B,1,Lc,kv_lora]; k_pe_all: [B,1,Lc,qk_rope].
        let (kv_latent, k_pe_all) =
            cache.update_and_fetch_on(&c_kv_n, &k_pe, per_row_lens, target)?;

        // Decoupled-RoPE logit term: (q_pe * scale) @ k_pe_allᵀ.
        // q_pe [B,H,L,qk_rope]; swapaxes(-1,-2) on the single-kv-head k_pe_all
        // [B,1,Lc,qk_rope] -> [B,1,qk_rope,Lc]; matmul broadcasts the kv head
        // across all H query heads -> pe_scores [B,H,L,Lc].
        let q_pe_scaled = mlx::ops::binary::multiply_on(&q_pe, &self.scale_array()?, target)?;
        let k_pe_t = k_pe_all.transpose_axes_on(&[0, 1, 3, 2][..], target)?;
        let mut pe_scores = q_pe_scaled.matmul_on(&k_pe_t, target)?;

        // Fold the engine's ADDITIVE float mask into pe_scores (see doc above).
        if let Some(m) = mask {
            pe_scores = mlx::ops::binary::add_on(&pe_scores, m, target)?;
        }

        // Regime-dispatched SDPA + value un-fold (decode when L==1).
        let out = self.attend_regime(&q_nope, &kv_latent, &pe_scores, l == 1, target)?;

        // Merge heads: [B,H,L,v_head] -> [B,L,H,v_head] -> [B,L,H*v_head].
        let out = out
            .transpose_axes_on(&[0, 2, 1, 3][..], target)?
            .reshape_on((b, l, self.n_heads * self.v_head), target)?;
        self.o_proj.forward_on(&out, target)
    }

    /// Two-regime SDPA core: given the per-head NoPE query `q_nope`
    /// `[B,H,L,qk_nope]`, the cached latent `kv_latent` `[B,1,Lc,kv_lora]`, and
    /// the precomputed logit mask `pe_scores` `[B,H,L,Lc]` (decoupled-RoPE term
    /// + engine mask), run the absorbed (`decode`) or un-folded (prefill) SDPA
    /// and return the per-head attention output `[B,H,L,v_head]`.
    ///
    /// SDPA computes `softmax(scale·q@kᵀ + pe_scores) @ v`; the same softmax
    /// scale is passed in both regimes, and `pe_scores` already carries the
    /// `*scale` RoPE term + mask (so `"causal"` mode is intentionally unused —
    /// all masking is folded into `pe_scores`, `mask_mode = "array"`).
    ///
    /// The two regimes are algebraically identical:
    /// - **Decode** folds the query into the latent (`q_lat = embed_q(q_nope)`),
    ///   attends with `K = V = kv_latent`, then un-folds the output
    ///   (`unembed_out` AFTER softmax).
    /// - **Prefill** un-folds the latent into per-head `K = embed_q(latent,
    ///   transpose=false)` and `V = unembed_out(latent)` BEFORE attention.
    ///
    /// Because `embed_q` transpose=true / transpose=false are the same weight
    /// transposed and `unembed_out` is linear, both regimes yield the same
    /// attention given identical inputs (Task 4b equivalence gate). `#[doc(hidden)]
    /// pub` so the equivalence test can drive BOTH regimes on shared inputs.
    #[doc(hidden)]
    pub fn attend_regime(
        &self,
        q_nope: &Array,
        kv_latent: &Array,
        pe_scores: &Array,
        decode: bool,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        // SDPA requires the `mask_arr` dtype to promote to the attention output
        // dtype (q/k/v promoted type). `pe_scores` accumulates in float32 (the
        // scaled-RoPE matmul promotes via the f32 `scale_array`, and the engine
        // additive mask carries `-inf` in f32), while q/k/v are the bf16
        // activation dtype — f32 does NOT promote to bf16. Demote `pe_scores`
        // to the latent (SDPA input) dtype so the mask matches. `kv_latent` is
        // an SDPA input in both regimes, so its dtype is the safe target.
        let mask_dtype = kv_latent.dtype();
        let pe_scores = if pe_scores.dtype() == mask_dtype {
            pe_scores.clone()
        } else {
            mlx::ops::cast::astype_on(pe_scores, mask_dtype, target)?
        };
        if decode {
            // DECODE: fold the query into latent space, attend against the
            // cached latent (K = V = kv_latent), then un-fold the output.
            let q_lat = self.embed_q.apply(q_nope, true, target)?; // [B,H,L,kv_lora]
            let o = mlx::fast::scaled_dot_product_attention_on(
                &q_lat,
                kv_latent,
                kv_latent,
                self.scale,
                "array",
                Some(&pe_scores),
                None,
                target,
            )?; // [B,H,L,kv_lora]
            self.unembed_out.apply(&o, true, target) // [B,H,L,v_head]
        } else {
            // PREFILL: un-fold the cached latent into per-head K (qk_nope) and
            // V (v_head). SDPA tolerates V last-dim != Q/K last-dim (MLX only
            // enforces q==k last dim + k==v head-count).
            let k = self.embed_q.apply(kv_latent, false, target)?; // [B,H,Lc,qk_nope]
            let v = self.unembed_out.apply(kv_latent, true, target)?; // [B,H,Lc,v_head]
            Ok(mlx::fast::scaled_dot_product_attention_on(
                q_nope,
                &k,
                &v,
                self.scale,
                "array",
                Some(&pe_scores),
                None,
                target,
            )?) // [B,H,L,v_head]
        }
    }

    /// Build the softmax scale as a 1-element `Array` for `multiply_on`
    /// (the overloaded `&Array * f32` panics on error; this propagates it).
    fn scale_array(&self) -> Result<Array> {
        Ok((&[self.scale][..], ()).try_into()?)
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

    // ===================== Task 4b: two-regime forward =====================

    // `MlaLatentCache`, `Array`, `Linear`, `RmsNorm`, `Glm4Rope` come via
    // `use super::*`; `Dtype` is only needed by the Task 4b cache ctors.
    use mlx::Dtype;

    /// Task 4b cache dims (must match `tiny_mla_fwd`): kv_lora=32, qk_rope=2.
    const KV_LORA: i32 = 32;
    const QK_ROPE: i32 = 2;

    /// Deterministic pseudo-random fp data in `[-0.5, 0.5]`.
    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// 8-bit per-head quant of an fp `[H, out, in]` weight (group_size = in).
    fn quant_per_head_8b(w_fp: &Array, in_dim: i32) -> PerHeadQuantLinear {
        let q = mlx::quantization::quantize_on(w_fp, Some(in_dim), Some(8), "affine", None, ())
            .unwrap();
        PerHeadQuantLinear::from_parts(q[0].clone(), q[1].clone(), q.get(2).cloned(), in_dim, 8)
    }

    /// Tiny `MlaAttention` for the two-regime forward tests.
    ///
    /// Dims H=2, qk_nope=32, qk_rope=2, kv_lora=32, v_head=4, hidden=8. Unlike
    /// `tiny_mla` (used by the project_qkv shape test), `embed_q`/`unembed_out`
    /// here have REAL, non-trivial per-head weights with quant inner dims that
    /// are valid group sizes: `embed_q` in=qk_nope=32, `unembed_out` in=kv_lora=32
    /// (MLX `quantized_matmul` requires the inner dim to be a multiple of a
    /// supported group_size 32/64/128). 8-bit keeps the empirical decode-vs-
    /// prefill float floor under 1e-3; the algebra is proved separately in f64.
    fn tiny_mla_fwd() -> MlaAttention {
        let h = 2_i32;
        let qk_nope = 32_i32;
        let qk_rope = 2_i32;
        let kv_lora = 32_i32;
        let v_head = 4_i32;
        let q_head_dim = qk_nope + qk_rope; // 34
        let q_lora = 8_i32;
        let hidden = 8_i32;

        let q_a_proj = Linear::new_fp(
            arr(&rnd((q_lora * hidden) as usize, 1), &[q_lora, hidden]),
            None,
        );
        let q_b_out = h * q_head_dim; // 68
        let q_b_proj = Linear::new_fp(
            arr(&rnd((q_b_out * q_lora) as usize, 2), &[q_b_out, q_lora]),
            None,
        );
        let kv_out = kv_lora + qk_rope; // 34
        let kv_a_proj_with_mqa = Linear::new_fp(
            arr(&rnd((kv_out * hidden) as usize, 3), &[kv_out, hidden]),
            None,
        );
        let o_proj = Linear::new_fp(
            arr(
                &rnd((hidden * h * v_head) as usize, 4),
                &[hidden, h * v_head],
            ),
            None,
        );

        let q_a_layernorm = RmsNorm::new(arr(&vec![1.0f32; q_lora as usize], &[q_lora]), 1e-5);
        let kv_a_layernorm = RmsNorm::new(arr(&vec![1.0f32; kv_lora as usize], &[kv_lora]), 1e-5);

        // embed_q: per head qk_nope(32) -> kv_lora(32); weight [H, kv_lora, qk_nope].
        let embed_q = quant_per_head_8b(
            &arr(
                &rnd((h * kv_lora * qk_nope) as usize, 5),
                &[h, kv_lora, qk_nope],
            ),
            qk_nope,
        );
        // unembed_out: per head kv_lora(32) -> v_head(4); weight [H, v_head, kv_lora].
        let unembed_out = quant_per_head_8b(
            &arr(
                &rnd((h * v_head * kv_lora) as usize, 6),
                &[h, v_head, kv_lora],
            ),
            kv_lora,
        );

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

    /// Lower-triangular additive causal mask `[1,1,l,l]` (0 valid, -inf future).
    fn causal_mask(l: i32) -> Array {
        let neg = f32::NEG_INFINITY;
        let mut m = vec![0.0f32; (l * l) as usize];
        for i in 0..l as usize {
            for j in 0..l as usize {
                if j > i {
                    m[i * l as usize + j] = neg;
                }
            }
        }
        arr(&m, &[1, 1, l, l])
    }

    /// f64 reference attention for ONE regime, computed in pure Rust so the two
    /// orderings sum in a deterministic, high-precision order (B=1, single
    /// query). Both orderings are mathematically identical, so they agree to
    /// ~f64 epsilon — the strict algebra guarantee.
    ///
    /// - decode: `q_lat = W_eq·q`; `scores_j = scale·(q_lat·lat_j)+pe_j`;
    ///   `ctx = Σ_j softmax(scores)_j·lat_j`; `out = W_uo·ctx`.
    /// - prefill: `k_j = W_eqᵀ·lat_j`; `v_j = W_uo·lat_j`;
    ///   `scores_j = scale·(q·k_j)+pe_j`; `out = Σ_j softmax(scores)_j·v_j`.
    ///
    /// `w_eq` `[H,kv_lora,qk_nope]`, `w_uo` `[H,v_head,kv_lora]`.
    #[allow(clippy::too_many_arguments)]
    fn f64_attend(
        q_nope: &[f32],
        latent: &[f32],
        pe: &[f32],
        w_eq: &[f32],
        w_uo: &[f32],
        h: usize,
        qk_nope: usize,
        kv_lora: usize,
        v_head: usize,
        lc: usize,
        scale: f64,
        decode: bool,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; h * v_head];
        for head in 0..h {
            let qh = &q_nope[head * qk_nope..(head + 1) * qk_nope];
            let weq = &w_eq[head * kv_lora * qk_nope..(head + 1) * kv_lora * qk_nope];
            let wuo = &w_uo[head * v_head * kv_lora..(head + 1) * v_head * kv_lora];

            let mut scores = vec![0.0f64; lc];
            if decode {
                let mut q_lat = vec![0.0f64; kv_lora];
                for (o, ql) in q_lat.iter_mut().enumerate() {
                    let mut acc = 0.0f64;
                    for (i, &qi) in qh.iter().enumerate() {
                        acc += weq[o * qk_nope + i] as f64 * qi as f64;
                    }
                    *ql = acc;
                }
                for (j, sc) in scores.iter_mut().enumerate() {
                    let lat_j = &latent[j * kv_lora..(j + 1) * kv_lora];
                    let mut dot = 0.0f64;
                    for (o, &ql) in q_lat.iter().enumerate() {
                        dot += ql * lat_j[o] as f64;
                    }
                    *sc = scale * dot + pe[head * lc + j] as f64;
                }
            } else {
                for (j, sc) in scores.iter_mut().enumerate() {
                    let lat_j = &latent[j * kv_lora..(j + 1) * kv_lora];
                    let mut dot = 0.0f64;
                    for (i, &qi) in qh.iter().enumerate() {
                        let mut kij = 0.0f64;
                        for (o, &lo) in lat_j.iter().enumerate() {
                            kij += weq[o * qk_nope + i] as f64 * lo as f64;
                        }
                        dot += qi as f64 * kij;
                    }
                    *sc = scale * dot + pe[head * lc + j] as f64;
                }
            }

            let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
            let den: f64 = exps.iter().sum();
            let attn: Vec<f64> = exps.iter().map(|e| e / den).collect();

            for (o, ov) in out[head * v_head..(head + 1) * v_head]
                .iter_mut()
                .enumerate()
            {
                let mut acc = 0.0f64;
                if decode {
                    for c in 0..kv_lora {
                        let mut ctx_c = 0.0f64;
                        for (j, &a) in attn.iter().enumerate() {
                            ctx_c += a * latent[j * kv_lora + c] as f64;
                        }
                        acc += wuo[o * kv_lora + c] as f64 * ctx_c;
                    }
                } else {
                    for (j, &a) in attn.iter().enumerate() {
                        let lat_j = &latent[j * kv_lora..(j + 1) * kv_lora];
                        let mut vjo = 0.0f64;
                        for (c, &lc_v) in lat_j.iter().enumerate() {
                            vjo += wuo[o * kv_lora + c] as f64 * lc_v as f64;
                        }
                        acc += a * vjo;
                    }
                }
                *ov = acc;
            }
        }
        out
    }

    /// THE CORRECTNESS GATE: decode (absorbed) and prefill (un-folded) are the
    /// SAME attention.
    ///
    /// Apples-to-apples: BOTH regimes are fed the EXACT same post-projection
    /// tensors (`q_nope`, `kv_latent`, `pe_scores`) via [`MlaAttention::attend_regime`],
    /// toggling only the `decode` flag. (We deliberately do NOT compare a
    /// 4-token prefill against a decode-of-the-4th-token: the SAME token
    /// projected in a 4-row batch vs a 1-row batch already differs by ~1e-3 due
    /// to shape-dependent MLX matmul/RoPE float rounding — measured directly —
    /// which would confound the regime comparison.)
    ///
    /// Two layers of assurance:
    /// 1. ALGEBRA (f64-exact, < 1e-9): a pure-Rust f64 reference computes both
    ///    orderings in deterministic high precision; they agree to f64 epsilon —
    ///    the rigorous mathematical proof that decode == prefill.
    /// 2. IMPLEMENTATION (< 1e-3): the real quantized `attend_regime` regimes
    ///    agree with EACH OTHER and with the f64 reference. They cannot agree
    ///    tighter than ~3e-4 for two compounding, IRREDUCIBLE reasons (both
    ///    verified): (a) f32 summation associativity — decode contracts the
    ///    qk_nope dim early (into `q_lat`) while prefill contracts it last (into
    ///    `k`), so even a pure-f32 reference differs at ~1.5e-4; (b) MLX
    ///    dispatches `quantized_matmul` transpose=true (qmv/qmm_t) and
    ///    transpose=false (qvm/qmm_n) to DIFFERENT kernels with different
    ///    reduction trees (bit-width- and snap-to-grid-independent — verified).
    ///    A 1e-4 gate on the QUANTIZED outputs is physically unattainable when
    ///    both quant kernels run; assertion (1) carries the strict algebra
    ///    guarantee, (2) confirms the implementation realizes it within the
    ///    float floor. This also value-checks the prefill `transpose=false`
    ///    un-fold (closing the Task 4a review Minor).
    #[test]
    fn decode_and_prefill_regimes_agree() {
        let mla = tiny_mla_fwd();
        let h = 2_usize;
        let qk_nope = 32_usize;
        let kv_lora = KV_LORA as usize;
        let v_head = 4_usize;
        let lc = 3_usize; // cached context length

        // Shared inputs, built ONCE and fed to both regimes.
        let q_data = rnd(h * qk_nope, 100);
        let lat_data = rnd(lc * kv_lora, 101);
        let pe_data = rnd(h * lc, 102);
        let q_nope = arr(&q_data, &[1, h as i32, 1, qk_nope as i32]);
        let kv_latent = arr(&lat_data, &[1, 1, lc as i32, kv_lora as i32]);
        let pe_scores = arr(&pe_data, &[1, h as i32, 1, lc as i32]);

        let out_dec = mla
            .attend_regime(&q_nope, &kv_latent, &pe_scores, true, ())
            .unwrap();
        let out_pre = mla
            .attend_regime(&q_nope, &kv_latent, &pe_scores, false, ())
            .unwrap();
        assert_eq!(
            out_dec.shape().as_slice(),
            &[1, h as i32, 1, v_head as i32],
            "decode shape"
        );
        assert_eq!(
            out_pre.shape().as_slice(),
            &[1, h as i32, 1, v_head as i32],
            "prefill shape"
        );

        // (1) f64-exact algebra proof.
        let w_eq = mla
            .embed_q()
            .dequant_weight()
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let w_uo = mla
            .unembed_out()
            .dequant_weight()
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let scale = mla.scale() as f64;
        let ref_dec = f64_attend(
            &q_data, &lat_data, &pe_data, &w_eq, &w_uo, h, qk_nope, kv_lora, v_head, lc, scale,
            true,
        );
        let ref_pre = f64_attend(
            &q_data, &lat_data, &pe_data, &w_eq, &w_uo, h, qk_nope, kv_lora, v_head, lc, scale,
            false,
        );
        let algebra = ref_dec
            .iter()
            .zip(ref_pre.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            algebra < 1e-9,
            "f64-exact decode vs prefill (THE algebra) max|diff| = {algebra} (want < 1e-9)"
        );

        // (2) Real quantized regimes agree with each other + the f64 reference.
        let dv = out_dec.to_vec::<f32>().unwrap();
        let pv = out_pre.to_vec::<f32>().unwrap();
        let regimes = dv
            .iter()
            .zip(pv.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let dec_vs_ref = dv
            .iter()
            .zip(ref_dec.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - *b as f32).abs()));
        let pre_vs_ref = pv
            .iter()
            .zip(ref_pre.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - *b as f32).abs()));
        assert!(
            regimes < 1e-3 && dec_vs_ref < 1e-3 && pre_vs_ref < 1e-3,
            "quantized regimes vs each other ({regimes}) / vs f64 ref (dec {dec_vs_ref}, pre {pre_vs_ref}); want < 1e-3 float+quant-kernel floor"
        );
    }

    /// The additive mask is actually applied: blocking a real cached position
    /// must change the decode output (no future/masked leakage).
    #[test]
    fn decode_respects_causality() {
        let mla = tiny_mla_fwd();
        let hidden = 8_i32;
        let x_all_vec = rnd((4 * hidden) as usize, 7);

        let build_cache = || {
            let mut c = MlaLatentCache::new(1, KV_LORA, QK_ROPE, Dtype::Float32, 16).with_step(16);
            let mask_seed = causal_mask(3);
            mla.forward_on(
                &arr(&x_all_vec[..(3 * hidden) as usize], &[1, 3, hidden]),
                &off(&[0], &[1]),
                &mut c,
                &[3],
                Some(&mask_seed),
                (),
            )
            .unwrap();
            c
        };

        let x4 = arr(
            &x_all_vec[(3 * hidden) as usize..(4 * hidden) as usize],
            &[1, 1, hidden],
        );

        // Decode seeing all 4 positions (after the write, Lc = 4): mask=None.
        let mut c_full = build_cache();
        let out_full = mla
            .forward_on(&x4, &off(&[3], &[1]), &mut c_full, &[1], None, ())
            .unwrap();

        // Decode with cached position 2 blocked via an additive mask -> output
        // must change (the masked position carried signal).
        let mut c_mask = build_cache();
        let neg = f32::NEG_INFINITY;
        let dmask = arr(&[0.0, 0.0, neg, 0.0], &[1, 1, 1, 4]);
        let out_mask = mla
            .forward_on(&x4, &off(&[3], &[1]), &mut c_mask, &[1], Some(&dmask), ())
            .unwrap();

        let vf = out_full.to_vec::<f32>().unwrap();
        let vm = out_mask.to_vec::<f32>().unwrap();
        let max_diff = vf
            .iter()
            .zip(vm.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_diff > 1e-5,
            "masking cache position 2 changed nothing (max|diff|={max_diff}) — mask not applied"
        );
    }

    /// `forward_on` produces `[B,L,hidden]` in prefill and `[B,1,hidden]` in
    /// decode, dispatching on the query seq len.
    #[test]
    fn regime_output_shapes() {
        let mla = tiny_mla_fwd();
        let hidden = 8_i32;

        // Prefill L=3 -> [1,3,hidden].
        let mut c1 = MlaLatentCache::new(1, KV_LORA, QK_ROPE, Dtype::Float32, 16).with_step(16);
        let x3 = arr(&rnd((3 * hidden) as usize, 11), &[1, 3, hidden]);
        let mask3 = causal_mask(3);
        let o3 = mla
            .forward_on(&x3, &off(&[0], &[1]), &mut c1, &[3], Some(&mask3), ())
            .unwrap();
        assert_eq!(o3.shape().as_slice(), &[1, 3, hidden]);

        // Decode L=1 against the now-3-token cache -> [1,1,hidden].
        let x1 = arr(&rnd(hidden as usize, 12), &[1, 1, hidden]);
        let o1 = mla
            .forward_on(&x1, &off(&[3], &[1]), &mut c1, &[1], None, ())
            .unwrap();
        assert_eq!(o1.shape().as_slice(), &[1, 1, hidden]);
    }
}
