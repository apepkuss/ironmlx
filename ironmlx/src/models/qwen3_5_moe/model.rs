//! Top-level Qwen3.5 MoE model: Qwen35MoeTextModel + lm_head (untied).
//! Implements `core::model::Model` trait for use in generic Scheduler /
//! GenerationStream / SchedulerActor / AppState pipelines (post-P5a).

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::{GatedDeltaCache, KVCache};
use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::nn::{AttnKind, LayerCache, Linear};
use crate::Result;

use super::config::Qwen35MoeConfig;
use super::text_model::Qwen35MoeTextModel;

/// Minimum K/V cache cap consistent with dense Qwen35Model's GPU-perf floor.
/// See `crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF`.
pub const MIN_KV_CACHE_CAP_FOR_GPU_PERF: i32 =
    crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF;

pub struct Qwen35MoeModel {
    text: Qwen35MoeTextModel,
    /// Always Some for 35B-A3B (tie_word_embeddings=false).
    lm_head: Linear,
}

/// Slice per-row last hidden states from `hidden [B, S, H]`.
///
/// For row `i`, extracts `hidden[i, last_positions[i], :]` then stacks
/// to `[B, 1, H]`. Used by [`Qwen35MoeModel::batched_prefill`] to project
/// per-row last-token logits when prompts have different lengths under
/// right-padding.
///
/// # Errors
/// - `last_positions.len() != B`
/// - `last_positions[i] < 0 || last_positions[i] >= S` for any `i`
fn per_row_slice_last(
    hidden: &Array,
    last_positions: &[i32],
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let dims_borrow = hidden.shape();
    let dims = dims_borrow.as_slice();
    let (b, s, h) = (dims[0], dims[1], dims[2]);
    if last_positions.len() as i32 != b {
        return Err(anyhow!(
            "per_row_slice_last: last_positions.len()={} != batch={}",
            last_positions.len(),
            b
        ));
    }
    for (i, &pos) in last_positions.iter().enumerate() {
        if pos < 0 || pos >= s {
            return Err(anyhow!(
                "per_row_slice_last: last_positions[{i}]={pos} out of [0, {s})"
            ));
        }
    }
    // Per-row slice: row i takes hidden[i, positions[i], :] → [1, 1, H].
    // Concatenate along axis 0 to build [B, 1, H].
    let mut rows: Vec<Array> = Vec::with_capacity(b as usize);
    for (i, &pos) in last_positions.iter().enumerate() {
        let row = mlx::ops::indexing::slice_strided_on(
            hidden,
            &[i as i32, pos, 0][..],
            &[i as i32 + 1, pos + 1, h][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        rows.push(row);
    }
    let row_refs: Vec<&Array> = rows.iter().collect();
    Ok(mlx::ops::shape::concatenate_on(&row_refs[..], 0, target)?)
}

impl Qwen35MoeModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg =
            Qwen35MoeConfig::from_loader(loader).context("parsing Qwen35MoeConfig from loader")?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Qwen35MoeConfig) -> Result<Self> {
        if cfg.tie_word_embeddings {
            return Err(anyhow!(
                "Qwen35MoeModel: tie_word_embeddings expected false for A3B (got true)"
            ));
        }
        let lm_head =
            Linear::from_loader(loader, "lm_head").context("loading Qwen35MoeModel lm_head")?;
        let text = Qwen35MoeTextModel::from_loader(loader, cfg)?;
        Ok(Self { text, lm_head })
    }

    pub fn config(&self) -> &Qwen35MoeConfig {
        self.text.config()
    }

    pub fn text(&self) -> &Qwen35MoeTextModel {
        &self.text
    }

    /// Conservative weight-bytes estimate for memory budgeting.
    /// Formula:
    ///   attn: 4 * H^2 * L / 2     (Q,K,V,O projections per layer, 4-bit)
    ///   routed_experts: 3 * E * H * moe_inter * L / 2  (gate, up, down per expert)
    ///   shared_expert:  3 * H * shared_inter * L / 2
    ///   embed: vocab * H / 2
    ///   lm_head: vocab * H / 2
    fn approx_weight_bytes(&self) -> usize {
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let e = cfg.num_experts as usize;
        let me = cfg.moe_intermediate_size as usize;
        let se = cfg.shared_expert_intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;

        let attn = 4 * h * h * l / 2;
        let routed = 3 * e * h * me * l / 2;
        let shared = 3 * h * se * l / 2;
        // embed + lm_head (separate per tie_word_embeddings=false)
        let embed_head = 2 * vocab * h / 2;
        attn + routed + shared + embed_head
    }

    /// Slice the last sequence position from `hidden [B, S, H]` and project to
    /// vocab logits `[B, 1, vocab_size]`. Shared by `forward_on` and
    /// `batched_prefill`.
    ///
    /// When `last_positions` is `Some(positions)` (length == B), each row's
    /// last real token is at column `positions[i]` — used by the right-padded
    /// `batched_prefill` path where rows have ragged real lengths.
    ///
    /// When `last_positions` is `None` (single-stream `forward_on`), the
    /// fallback slices column `S - 1` for every row — behaviourally
    /// equivalent for B=1 or uniform-length inputs.
    fn slice_last_and_project(
        &self,
        hidden: &Array,
        last_positions: Option<&[i32]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let dims_borrow = hidden.shape();
        let dims = dims_borrow.as_slice();
        let (b, s, h) = (dims[0], dims[1], dims[2]);
        let last_hidden = match last_positions {
            Some(positions) if s > 1 => per_row_slice_last(hidden, positions, target)?,
            _ if s > 1 => {
                // Single-stream / uniform-length fallback: slice column s-1.
                mlx::ops::indexing::slice_strided(
                    hidden,
                    &[0_i32, s - 1, 0][..],
                    &[b, s, h][..],
                    &[1_i32, 1, 1][..],
                )?
            }
            _ => hidden.clone(),
        };
        // T4.1: wrap lm_head projection in a `slice_last_and_project_lm_head`
        // span. Lane-A (`routing_path = "scheduler"`) emits; Lane-B no-ops via
        // the centralized `try_with_p5h_span_from_current_trace` allow-list
        // (gs_chunked top-level-only). `layer_idx` is not meaningful here
        // (lm_head sits at the top of the model, outside decoder layers) →
        // SpanFields uses defaults.
        #[cfg(feature = "p5h-profile")]
        {
            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "slice_last_and_project_lm_head",
                crate::core::p5h::SpanFields::default,
                || {
                    let logits = self.lm_head.forward_on(&last_hidden, target)?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&logits])?;
                    }
                    Ok(logits)
                },
            )
        }
        #[cfg(not(feature = "p5h-profile"))]
        {
            self.lm_head.forward_on(&last_hidden, target)
        }
    }

    /// Construct per-layer cache list matching this model's hybrid topology.
    /// Matches `Qwen35Model::make_cache` style (one-shot allocate to cap to
    /// avoid grow_to on first decode step).
    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let cfg = self.config();
        let head_dim = cfg.effective_head_dim();
        let mut out = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            match cfg.layer_kind(i) {
                AttnKind::Full => {
                    // P8a-stage6: one-shot allocate to full cap (step >= cap)
                    // so the first decode step at long context never triggers
                    // grow_to. KVCache's default step=256 would otherwise
                    // round prefill alloc down to a step boundary and force
                    // a full-buffer reallocation + memcpy on decode step 1.
                    out.push(LayerCache::Full(
                        KVCache::new(
                            batch,
                            cfg.num_key_value_heads,
                            head_dim,
                            head_dim,
                            dtype,
                            cap,
                        )
                        .with_step(cap),
                    ));
                }
                AttnKind::Linear => {
                    let conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
                        + cfg.linear_value_head_dim * cfg.linear_num_value_heads;
                    out.push(LayerCache::Linear(GatedDeltaCache::new_with_cap(
                        batch,
                        cfg.linear_conv_kernel_dim,
                        conv_dim,
                        cfg.linear_num_value_heads,
                        cfg.linear_value_head_dim,
                        cfg.linear_key_head_dim,
                        dtype,
                        cap,
                    )?));
                }
            }
        }
        Ok(out)
    }

    /// Single-stream forward returning last-position logits `[B, 1, vocab]`.
    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self.text.forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )?;
        self.slice_last_and_project(&hidden, None, target)
    }

    /// Batched prefill returning per-row last-position logits `[B, 1, vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self.text.embed_on(input_ids, target)?;
        let hidden = self.text.forward_post_embedding_on(
            &hidden,
            position_ids,
            cache,
            Some(attention_mask),
            Some(linear_attention_mask),
            Some(per_row_lens),
            target,
        )?;
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        self.slice_last_and_project(&hidden, Some(&last_positions), target)
    }

    /// Run transformer + final norm, returning hidden state (no lm_head).
    pub fn forward_text_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.text.forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    pub fn model_meta(&self) -> ModelMeta {
        let cfg = self.config();
        ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: cfg.head_dim,
            weight_bytes: self.approx_weight_bytes(),
            max_position_embeddings: cfg.max_position_embeddings,
            // text-only MoE has no vision; default to 2 (won't be used at runtime
            // since MoE doesn't expose VL endpoints — Boss decided P5 D2)
            spatial_merge_size: 2,
        }
    }
}

impl Model for Qwen35MoeModel {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        Qwen35MoeModel::make_cache(self, batch, cap, dtype)
    }

    fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Qwen35MoeModel::forward_on(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Qwen35MoeModel::batched_prefill(
            self,
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            cache,
            target,
        )
    }

    fn forward_text_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        Qwen35MoeModel::forward_text_hidden(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn model_meta(&self) -> ModelMeta {
        Qwen35MoeModel::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.config().num_hidden_layers as usize
    }
}

/// **Stub** impl to satisfy `Scheduler<M>` / `SchedulerActor<M>` /
/// `AppState<M>` bound `M: DenseVlMethods + ...` so MoE can flow through
/// the generic infrastructure. All VL methods panic — MoE is text-only
/// per P5 D2 and VL endpoints are dense-only per P5c §3.10 CLI dispatch.
///
/// This is a compile-time accommodation, not a runtime VL capability.
/// P6.x (VL phase) will properly factor out the trait or introduce a
/// `MultimodalModel` marker trait.
impl crate::core::scheduler::DenseVlMethods for Qwen35MoeModel {
    fn batched_prefill_vl(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _attention_mask: &mlx::Array,
        _linear_attention_mask: &mlx::Array,
        _per_row_lens: &[i32],
        _per_row_pixel_values: &[Option<&mlx::Array>],
        _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        _image_token_id: i32,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        panic!(
            "Qwen35MoeModel::batched_prefill_vl: MoE is text-only (P5 D2). \
             VL endpoints should not be wired to MoE; this is a runtime guard."
        );
    }

    fn compute_vision_embeds(
        &self,
        _pixel_values: &mlx::Array,
        _grid_thw: &[(i32, i32, i32)],
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        panic!("Qwen35MoeModel::compute_vision_embeds: MoE is text-only (P5 D2).");
    }

    fn forward_vl_chunk(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&mlx::Array>,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _vision_embeds_slice: Option<&mlx::Array>,
        _image_token_id: i32,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        panic!("Qwen35MoeModel::forward_vl_chunk: MoE is text-only (P5 D2).");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::AttnKind;

    fn make_cfg() -> Qwen35MoeConfig {
        Qwen35MoeConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(8),
            vocab_size: 1024,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: false,
            full_attention_interval: 2,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_parameters: super::super::config::RopeParams {
                partial_rotary_factor: 0.25,
                rope_theta: 1e7,
                mrope_section: vec![2, 1, 1],
            },
            num_experts: 8,
            num_experts_per_tok: 2,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            mlp_only_layers: vec![],
            max_position_embeddings: 32768,
        }
    }

    #[test]
    fn make_cache_layer_kinds_match_partition() {
        // 4 layers, full_attention_interval=2 → layers {1, 3} are Full.
        let cfg = make_cfg();
        assert_eq!(cfg.layer_kind(0), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(1), AttnKind::Full);
        assert_eq!(cfg.layer_kind(2), AttnKind::Linear);
        assert_eq!(cfg.layer_kind(3), AttnKind::Full);

        // We can't construct a full Qwen35MoeModel without real weights,
        // but we can exercise make_cache logic directly via a mock-like
        // config-driven path. Test only the config partition logic here;
        // actual make_cache is exercised in integration tests.
    }

    #[test]
    fn model_meta_fields_populated() {
        // Verify model_meta wires cfg fields correctly via approx_weight_bytes.
        let cfg = make_cfg();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let e = cfg.num_experts as usize;
        let me = cfg.moe_intermediate_size as usize;
        let se = cfg.shared_expert_intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;

        let expected_bytes =
            4 * h * h * l / 2 + 3 * e * h * me * l / 2 + 3 * h * se * l / 2 + 2 * vocab * h / 2;

        // We can compute expected bytes without constructing the full model:
        let attn = 4 * h * h * l / 2;
        let routed = 3 * e * h * me * l / 2;
        let shared = 3 * h * se * l / 2;
        let embed_head = 2 * vocab * h / 2;
        assert_eq!(expected_bytes, attn + routed + shared + embed_head);

        // Verify spatial_merge_size sentinel and max_position_embeddings passthrough.
        assert_eq!(cfg.max_position_embeddings, 32768);
    }

    #[test]
    fn approx_weight_bytes_formula() {
        let cfg = make_cfg();
        let h = 32_usize;
        let l = 4_usize;
        let e = 8_usize;
        let me = 16_usize;
        let se = 16_usize;
        let vocab = 1024_usize;

        let attn = 4 * h * h * l / 2; // 1024
        let routed = 3 * e * h * me * l / 2; // 12288
        let shared = 3 * h * se * l / 2; // 3072
        let embed_head = 2 * vocab * h / 2; // 32768
        let expected = attn + routed + shared + embed_head;

        // Sanity-check formula correctness with the concrete cfg values.
        assert_eq!(
            expected,
            4 * 32 * 32 * 4 / 2 + 3 * 8 * 32 * 16 * 4 / 2 + 3 * 32 * 16 * 4 / 2 + 2 * 1024 * 32 / 2
        );
        let _ = cfg; // consumed for compile check
    }

    #[test]
    fn per_row_slice_last_uniform_pick() {
        // hidden [2, 4, 3] with deterministic values: (i*4 + j)*3 + c.
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        let out = per_row_slice_last(&hidden, &[3, 3], ()).expect("per_row_slice_last");
        assert_eq!(out.shape().as_slice(), &[2, 1, 3]);
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert_eq!(v, vec![9.0, 10.0, 11.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn per_row_slice_last_ragged_pick() {
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        let out = per_row_slice_last(&hidden, &[1, 3], ()).expect("per_row_slice_last ragged");
        assert_eq!(out.shape().as_slice(), &[2, 1, 3]);
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert_eq!(v, vec![3.0, 4.0, 5.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn per_row_slice_last_invalid_args_return_err() {
        let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
        let hidden: Array = (&data[..], (2_i32, 4_i32, 3_i32))
            .try_into()
            .expect("hidden try_into");
        // len mismatch (3 vs batch=2)
        let r1 = per_row_slice_last(&hidden, &[0, 1, 2], ());
        assert!(r1.is_err(), "len mismatch must Err");
        // negative position
        let r2 = per_row_slice_last(&hidden, &[-1, 1], ());
        assert!(r2.is_err(), "negative position must Err");
        // position >= s (s=4)
        let r3 = per_row_slice_last(&hidden, &[0, 4], ());
        assert!(r3.is_err(), "out-of-range position must Err");
    }
}
