//! GLM-4.7-Flash MoE block: noaux_tc sigmoid router + ungated shared expert.
//!
//! Mirrors mlx_lm `glm4_moe_lite.py`:
//!   - `group_expert_select` (`:197-228`) — the noaux_tc router (n_group=1
//!     path; ironmlx rejects grouped routing at config validation, so the
//!     group-mask branch is intentionally absent here).
//!   - `Glm4MoeLiteMoE.__call__` (`:277-290`):
//!
//! ```text
//! inds, scores = gate(x)
//! y = switch_mlp(x, inds)
//! y = (y * scores[..., None]).sum(-2)
//! y = y + shared_experts(x)     # shared is UNGATED (no shared_expert_gate)
//! ```
//!
//! Two silent-bug risks (spec § 6) are pinned by the router unit test:
//!   1. selection scores use `sigmoid` (NOT softmax — that is the Qwen path).
//!   2. routing weights are taken from the RAW sigmoid scores, NOT the
//!      bias-corrected `choice` scores (bias only steers SELECTION).

use anyhow::{Context, Result};
use mlx::ops::indexing::{slice_strided_on, take_along_axis_on};
use mlx::ops::shape::reshape_on;
use mlx::ops::sort::argpartition_on;
use mlx::ops::sum_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::Loader;
use crate::models::glm4_moe_lite::config::Glm4MoeLiteConfig;
use crate::models::qwen3_5_moe::sparse_moe::RoutedExperts;
use crate::nn::{Linear, Mlp};

#[cfg(feature = "p5h-profile")]
fn p5h_layer_fields(layer_idx: i32) -> crate::core::p5h::SpanFields {
    crate::core::p5h::SpanFields {
        layer_idx: Some(layer_idx),
        ..Default::default()
    }
}

#[cfg(feature = "p5h-profile")]
fn p5h_eval(arrays: &[&Array]) -> Result<()> {
    if crate::core::p5h::is_measurement_eval_probes_active() {
        mlx::transforms::eval(arrays)?;
    }
    Ok(())
}

/// noaux_tc router: sigmoid scores + additive selection bias → top-k experts.
///
/// Mirrors mlx_lm `group_expert_select` for the `n_group == 1` path (no group
/// masking). All score math is float32 (per omlx `:207`).
///
/// Arguments:
///   - `logits`: `[BS, E]` router logits (plain float `gate(x)`).
///   - `bias`: `[E]` `e_score_correction_bias`, broadcast over BS for selection.
///   - `k`: experts per token (`num_experts_per_tok`).
///   - `norm_topk_prob`: renormalize the top-k weights to sum to 1.
///   - `scale`: `routed_scaling_factor` applied to the final weights.
///
/// Returns `(inds, weights)`:
///   - `inds`: `[BS, k]` Uint32 expert ids (from `argpartition`).
///   - `weights`: `[BS, k]` Float32 routing weights, from the RAW sigmoid
///     scores (NOT the bias-corrected scores), optionally normalized, scaled.
///
/// `#[doc(hidden)] pub` (codebase convention, cf. `Linear::new_quant`) so the
/// integration test harness can call it directly.
#[doc(hidden)]
pub fn noaux_tc_route(
    logits: &Array,
    bias: &Array,
    k: i32,
    norm_topk_prob: bool,
    scale: f32,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    // Step 1: float32 sigmoid scores (omlx :207).
    let scores = logits
        .astype_on(Dtype::Float32, target)
        .context("noaux_tc_route: cast logits to f32")?
        .sigmoid_on(target)
        .context("noaux_tc_route: sigmoid")?; // [BS, E]

    // Step 2/3: selection scores = raw sigmoid + bias (omlx :209). The RAW
    // `scores` are kept for the weights below; only selection uses `choice`.
    let bias_f32 = bias
        .astype_on(Dtype::Float32, target)
        .context("noaux_tc_route: cast bias to f32")?;
    let choice = scores
        .try_add_on(&bias_f32, target)
        .context("noaux_tc_route: scores + bias")?; // [BS, E] (bias [E] broadcasts)

    let e = {
        let shape = choice.shape();
        let dims = shape.as_slice();
        if dims.len() != 2 {
            return Err(anyhow::anyhow!(
                "noaux_tc_route: logits must be rank-2 [BS,E], got rank {}",
                dims.len()
            ));
        }
        dims[dims.len() - 1]
    };
    if k <= 0 || k > e {
        return Err(anyhow::anyhow!(
            "noaux_tc_route: k must be in 1..={e}, got {k}"
        ));
    }

    // Step 5: top-k selection. Mirror omlx's negate form (omlx :221):
    //   inds = argpartition(-choice, kth=k-1, axis=-1)[..., :k]
    // Negating turns "largest choice" into "smallest negated", so the top-k
    // experts land in the FIRST k positions after partitioning at kth=k-1.
    let neg_choice = choice
        .try_neg_on(target)
        .context("noaux_tc_route: negate choice")?;
    let part = argpartition_on(&neg_choice, k - 1, -1, target)
        .context("noaux_tc_route: argpartition top-k")?; // Uint32 [BS, E]
    let bs = {
        let shape = part.shape();
        shape.as_slice()[0]
    };
    let inds = slice_strided_on(&part, [0_i32, 0], [bs, k], [1_i32, 1], target)
        .context("noaux_tc_route: slice top-k indices [.., :k]")?; // [BS, k]

    // Step 6: weights from the RAW sigmoid scores (omlx :222) — NOT `choice`.
    let mut weights = take_along_axis_on(&scores, &inds, -1, target)
        .context("noaux_tc_route: gather raw scores for top-k")?; // [BS, k]

    // Step 7: optional top-k renormalization (omlx :223-225, +1e-20 included).
    if k > 1 && norm_topk_prob {
        let denom = sum_on(&weights, -1_i32, /* keepdims */ true, target)
            .context("noaux_tc_route: sum top-k weights")?;
        let eps: Array = (&[1e-20_f32][..], ())
            .try_into()
            .map_err(|e| anyhow::anyhow!("noaux_tc_route: build eps scalar: {e}"))?;
        let denom = denom
            .try_add_on(&eps, target)
            .context("noaux_tc_route: denom + 1e-20")?;
        weights = weights
            .try_div_on(&denom, target)
            .context("noaux_tc_route: normalize weights")?;
    }

    // Step 8: routed_scaling_factor (omlx :226).
    let scale_arr: Array = (&[scale][..], ())
        .try_into()
        .map_err(|e| anyhow::anyhow!("noaux_tc_route: build scale scalar: {e}"))?;
    let weights = weights
        .try_mul_on(&scale_arr, target)
        .context("noaux_tc_route: scale weights")?;

    Ok((inds, weights))
}

/// GLM-4.7-Flash sparse MoE block.
///
/// Routing: noaux_tc sigmoid router (`gate` + `e_score_correction_bias`).
/// Routed path: `RoutedExperts::apply_experts` (shared SwitchGLU combine).
/// Shared path: a standard SwiGLU `Mlp`, added UNGATED (GLM has no
/// `shared_expert_gate`, unlike Qwen).
pub struct Glm4MoeBlock {
    /// Router gate: Linear(hidden → n_routed_experts). Plain float (the
    /// `gate.weight` checkpoint tensor has no `.scales`, so `Linear` loads it
    /// as `Fp`).
    gate: Linear,
    /// `e_score_correction_bias` `[E]` — additive selection bias.
    bias: Array,
    /// Stacked routed expert weights (`switch_mlp`).
    experts: RoutedExperts,
    /// Shared expert (`shared_experts`), added ungated.
    shared: Mlp,
    /// Experts selected per token (`num_experts_per_tok`).
    k: i32,
    /// Renormalize top-k weights (`norm_topk_prob`).
    norm: bool,
    /// `routed_scaling_factor` (from config — never hardcoded).
    scale: f32,
}

/// Diagnostic MoE execution mode for full-forward attribution benches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmMoeBlockMode {
    Full,
    RoutedOnly,
    RoutedFixedOnly,
    SharedOnly,
}

impl GlmMoeBlockMode {
    fn include_routed(self) -> bool {
        matches!(self, Self::Full | Self::RoutedOnly | Self::RoutedFixedOnly)
    }

    fn include_shared(self) -> bool {
        matches!(self, Self::Full | Self::SharedOnly)
    }
}

impl Glm4MoeBlock {
    /// Construct from `{prefix}` (typically `"model.layers.{i}.mlp"`).
    ///
    /// Sub-paths:
    ///   `{prefix}.gate`                          — router Linear (plain float)
    ///   `{prefix}.gate.e_score_correction_bias`  — selection bias `[E]`
    ///   `{prefix}.switch_mlp`                    — routed stacked experts
    ///   `{prefix}.shared_experts`                — shared SwiGLU Mlp (ungated)
    ///
    /// Note: GLM has NO `shared_expert_gate` tensor (shared expert is ungated).
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: &Glm4MoeLiteConfig) -> Result<Self> {
        let gate = Linear::from_loader(loader, &format!("{prefix}.gate"))
            .context("Glm4MoeBlock: loading router gate")?;
        let bias = loader
            .tensor(&format!("{prefix}.gate.e_score_correction_bias"))
            .context("Glm4MoeBlock: loading e_score_correction_bias")?
            .clone();
        let experts = RoutedExperts::from_loader(loader, &format!("{prefix}.switch_mlp"))
            .context("Glm4MoeBlock: loading routed experts")?;
        let shared = Mlp::from_loader(loader, &format!("{prefix}.shared_experts"))
            .context("Glm4MoeBlock: loading shared_experts")?;

        Ok(Self {
            gate,
            bias,
            experts,
            shared,
            k: cfg.num_experts_per_tok,
            norm: cfg.norm_topk_prob,
            scale: cfg.routed_scaling_factor,
        })
    }

    /// Forward pass: `[B, S, H]` → `[B, S, H]`.
    ///
    /// `layer_idx` is accepted to mirror the Qwen `SparseMoeBlock::forward_on`
    /// signature shape (consumed by p5h spans there); inert here.
    pub fn forward_on(&self, x: &Array, target: StreamOrDevice, layer_idx: i32) -> Result<Array> {
        self.forward_on_with_mode(x, target, layer_idx, GlmMoeBlockMode::Full)
    }

    /// Diagnostic variant of [`Self::forward_on`] that can isolate routed or shared experts.
    pub fn forward_on_with_mode(
        &self,
        x: &Array,
        target: StreamOrDevice,
        layer_idx: i32,
        mode: GlmMoeBlockMode,
    ) -> Result<Array> {
        let _ = layer_idx;
        let dims = x.shape();
        let dvec = dims.as_slice();
        if dvec.len() != 3 {
            return Err(anyhow::anyhow!(
                "Glm4MoeBlock::forward_on: x must be rank-3 [B,S,H], got rank {}",
                dvec.len()
            ));
        }
        let (b, s, h) = (dvec[0], dvec[1], dvec[2]);
        let bs = b * s;

        let flat =
            reshape_on(x, [bs, h], target).context("Glm4MoeBlock: reshape [B,S,H] → [BS,H]")?;

        #[cfg(feature = "p5h-profile")]
        {
            let routed = if mode.include_routed() {
                let (inds, weights) = crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "glm_moe_router_noaux_topk",
                    || p5h_layer_fields(layer_idx),
                    || -> Result<(Array, Array)> {
                        let (inds, weights) = if mode == GlmMoeBlockMode::RoutedFixedOnly {
                            fixed_route(bs, self.k, self.scale)?
                        } else {
                            let logits = self
                                .gate
                                .forward_on(&flat, target)
                                .context("Glm4MoeBlock: router gate forward")?; // [BS, E]
                            noaux_tc_route(
                                &logits, &self.bias, self.k, self.norm, self.scale, target,
                            )?
                        };
                        p5h_eval(&[&inds, &weights])?;
                        Ok((inds, weights))
                    },
                )?;

                Some(crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "glm_moe_routed_experts",
                    || p5h_layer_fields(layer_idx),
                    || -> Result<Array> {
                        let routed = self
                            .experts
                            .apply_experts_cast_output(&flat, &inds, &weights, target, layer_idx)
                            .context("Glm4MoeBlock: routed experts")?;
                        p5h_eval(&[&routed])?;
                        Ok(routed)
                    },
                )?) // [BS, H]
            } else {
                None
            };

            let shared = if mode.include_shared() {
                Some(crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "glm_moe_shared_expert",
                    || p5h_layer_fields(layer_idx),
                    || -> Result<Array> {
                        let shared = self
                            .shared
                            .forward_on(&flat, target)
                            .context("Glm4MoeBlock: shared expert forward")?;
                        p5h_eval(&[&shared])?;
                        Ok(shared)
                    },
                )?) // [BS, H] (UNGATED)
            } else {
                None
            };

            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "glm_moe_output_sum",
                || p5h_layer_fields(layer_idx),
                || -> Result<Array> {
                    let out_flat = combine_moe_outputs(routed, shared, target)
                        .context("Glm4MoeBlock: combine outputs")?; // [BS, H]
                    let out = reshape_on(&out_flat, [b, s, h], target)
                        .context("Glm4MoeBlock: reshape [BS,H] → [B,S,H]")?;
                    p5h_eval(&[&out])?;
                    Ok(out)
                },
            )
        }

        #[cfg(not(feature = "p5h-profile"))]
        {
            let routed = if mode.include_routed() {
                let (inds, weights) = if mode == GlmMoeBlockMode::RoutedFixedOnly {
                    fixed_route(bs, self.k, self.scale)?
                } else {
                    // Router: plain-float Linear logits → noaux_tc sigmoid selection.
                    let logits = self
                        .gate
                        .forward_on(&flat, target)
                        .context("Glm4MoeBlock: router gate forward")?; // [BS, E]
                    noaux_tc_route(&logits, &self.bias, self.k, self.norm, self.scale, target)?
                };

                Some(
                    self.experts
                        .apply_experts_cast_output(&flat, &inds, &weights, target, layer_idx)
                        .context("Glm4MoeBlock: routed experts")?,
                ) // [BS, H]
            } else {
                None
            };
            let shared = if mode.include_shared() {
                Some(
                    self.shared
                        .forward_on(&flat, target)
                        .context("Glm4MoeBlock: shared expert forward")?,
                ) // [BS, H] (UNGATED)
            } else {
                None
            };

            let out_flat = combine_moe_outputs(routed, shared, target)
                .context("Glm4MoeBlock: combine outputs")?; // [BS, H]
            reshape_on(&out_flat, [b, s, h], target)
                .context("Glm4MoeBlock: reshape [BS,H] → [B,S,H]")
        }
    }
}

fn fixed_route(bs: i32, k: i32, scale: f32) -> Result<(Array, Array)> {
    let mut ids = Vec::with_capacity((bs * k) as usize);
    let mut weights = Vec::with_capacity((bs * k) as usize);
    let weight = scale / k as f32;
    for _ in 0..bs {
        for expert in 0..k {
            ids.push(expert as u32);
            weights.push(weight);
        }
    }
    let inds: Array = (&ids[..], &[bs, k][..]).try_into()?;
    let weights: Array = (&weights[..], &[bs, k][..]).try_into()?;
    Ok((inds, weights))
}

fn combine_moe_outputs(
    routed: Option<Array>,
    shared: Option<Array>,
    target: StreamOrDevice,
) -> Result<Array> {
    match (routed, shared) {
        (Some(routed), Some(shared)) => Ok(routed
            .try_add_on(&shared, target)
            .context("Glm4MoeBlock: routed + shared")?),
        (Some(routed), None) => Ok(routed),
        (None, Some(shared)) => Ok(shared),
        (None, None) => Err(anyhow::anyhow!(
            "Glm4MoeBlock: diagnostic mode disabled both routed and shared outputs"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(d: &[f32], s: &[i32]) -> Array {
        (d, s).try_into().unwrap()
    }

    /// Real-weight full-block check (env-gated; skips without weights).
    /// Loads layer 1's MoE `mlp` and runs a tiny forward, asserting the
    /// output is correctly shaped, finite, and non-trivial — exercising the
    /// router + `apply_experts` + ungated shared path end-to-end.
    #[test]
    fn glm_moe_block_forward_is_finite() -> Result<()> {
        let Ok(snap) = std::env::var("GLM47_MODEL_DIR") else {
            eprintln!("skip: GLM47_MODEL_DIR unset");
            return Ok(());
        };
        let loader = Loader::open(std::path::Path::new(&snap))?;
        let cfg = Glm4MoeLiteConfig::from_loader(&loader)?;
        let block = Glm4MoeBlock::from_loader(&loader, "model.layers.1.mlp", &cfg)?;

        let h = cfg.hidden_size as usize;
        let data: Vec<f32> = (0..(2 * h))
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect();
        let x: Array = (data.as_slice(), &[1, 2, cfg.hidden_size][..])
            .try_into()
            .map_err(|e| anyhow::anyhow!("build x: {e}"))?;

        let out = block.forward_on(&x, StreamOrDevice::default(), 1)?;
        assert_eq!(out.shape().as_slice(), &[1, 2, cfg.hidden_size]);
        let ov = out.astype(Dtype::Float32)?.to_vec::<f32>()?;
        assert!(
            ov.iter().all(|v| v.is_finite()),
            "MoE output must be finite"
        );
        assert!(
            ov.iter().any(|v| *v != 0.0),
            "MoE output must be non-trivial"
        );
        Ok(())
    }

    /// SILENT-BUG SENTINEL for the two `noaux_tc_route` risks (spec § 6):
    ///   - selection scores use sigmoid (NOT softmax — the Qwen path),
    ///   - routing weights come from the RAW sigmoid scores (NOT the
    ///     bias-corrected `choice` scores).
    ///
    /// NON-UNIFORM logits are mandatory: with the old `[0,0,0,0]` inputs the
    /// correct path, the softmax-bug path, and the weights-from-choice-bug path
    /// ALL collapse to `[0.9, 0.9]` after `norm_topk_prob`, making the test a
    /// tautology. Here logits = [0,1,2,0] → sigmoid s = [0.5, 0.7310586,
    /// 0.8807971, 0.5]; choice = s + bias (bias = [-9,9,9,-9]) selects experts
    /// {1,2}; RAW weights {0.7310586, 0.8807971} → normalized (sum 1.6118557)
    /// {0.453534, 0.546466} → ×1.8 = {0.816361, 0.983639}.
    ///
    /// This now FAILS under either bug:
    ///   - softmax bug → normalized ≈[0.269, 0.731] → ×1.8 ≈[0.484, 1.316];
    ///   - weights-from-choice bug → normalized ≈[0.496, 0.504] → ×1.8
    ///     ≈[0.893, 0.907].
    /// Only correct sigmoid-from-raw yields {0.816, 0.984}. argpartition order
    /// is unspecified, so the two weights are asserted order-independently.
    #[test]
    fn router_selects_with_bias_weights_from_raw_sigmoid() -> Result<()> {
        let logits = arr(&[0.0, 1.0, 2.0, 0.0], &[1, 4]);
        let bias = arr(&[-9.0, 9.0, 9.0, -9.0], &[4]);
        let (inds, weights) =
            noaux_tc_route(&logits, &bias, 2, true, 1.8, StreamOrDevice::default())?;

        let mut iv: Vec<u32> = inds.to_vec::<u32>()?;
        iv.sort_unstable();
        assert_eq!(
            iv,
            vec![1, 2],
            "selection must pick the bias-boosted experts"
        );

        let mut wv: Vec<f32> = weights.to_vec::<f32>()?;
        assert_eq!(wv.len(), 2);
        wv.sort_by(|a, b| a.partial_cmp(b).expect("weights are finite"));
        let want = [0.816361_f32, 0.983639];
        for (w, e) in wv.iter().zip(want.iter()) {
            assert!(
                (w - e).abs() < 1e-4,
                "weight must be raw-sigmoid → norm → ×1.8 = {e}, got {w} (all {wv:?})"
            );
        }
        Ok(())
    }

    /// Without normalization the raw sigmoid scores flow through unchanged
    /// (then scaled). bias steers selection only.
    #[test]
    fn router_no_norm_keeps_raw_scaled_weights() -> Result<()> {
        // logits row → distinct sigmoids; bias forces selection of {1,2}.
        let logits = arr(&[0.0, 1.0, 2.0, 0.0], &[1, 4]);
        let bias = arr(&[-9.0, 9.0, 9.0, -9.0], &[4]);
        let (inds, weights) =
            noaux_tc_route(&logits, &bias, 2, false, 2.0, StreamOrDevice::default())?;

        let ids: Vec<u32> = inds.to_vec::<u32>()?;
        let wv: Vec<f32> = weights.to_vec::<f32>()?;
        // Map each selected id to its expected raw sigmoid * scale.
        let logit_by_id = [0.0_f32, 1.0, 2.0, 0.0];
        for (id, w) in ids.iter().zip(wv.iter()) {
            let raw = 1.0 / (1.0 + (-logit_by_id[*id as usize]).exp());
            let want = raw * 2.0;
            assert!((w - want).abs() < 1e-5, "id={id} got {w} want {want}");
        }
        Ok(())
    }
}
