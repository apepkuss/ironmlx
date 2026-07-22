//! GLM-4.7-Flash (`glm4_moe_lite`) top-level model.
//!
//! `embed_tokens → [Glm4DecoderLayer; num_hidden_layers] → norm → lm_head`
//! (mirrors `glm4_moe_lite.py:320-379`; lm_head is separate —
//! `tie_word_embeddings = false`).
//!
//! # Engine contract notes
//! - **`position_ids` is ignored.** GLM derives the RoPE offset from the
//!   per-row pre-update cache length ([`MlaLatentCache::offsets`]), so
//!   [`Model::requires_position_ids`] returns `false` and the scheduler passes
//!   a placeholder.
//! - **`linear_attention_mask` is ignored.** GLM has no linear-attention path;
//!   `batched_prefill` accepts the argument for trait-shape parity only.
//! - **Regime is per-call uniform (structurally).** A single forward is either
//!   all-prefill (`L > 1`) or all-decode (`L == 1`); the regime is decided by
//!   the single query length `L` (the `[B, L]` input's seq dim), so prefill and
//!   decode rows can never be mixed in one forward. `per_row_lens` is the count
//!   of REAL tokens each row writes to the cache and legitimately differs per
//!   row in B>1 batched prefill of different-length prompts — it is NOT the
//!   regime and is not required to be uniform.
//! - **Causal mask.** When the caller passes `mask = None` with `L > 1` (the
//!   B=1 prefix-forward + chunked-prefill paths), the model builds its own
//!   lower-right-aligned additive causal mask internally — mirroring the
//!   implicit SDPA `mask_mode="causal"` behaviour of the Qwen full-attention
//!   path. When `mask = Some(..)` (e.g. `batched_prefill`'s engine mask), it is
//!   passed straight through. For decode (`L == 1`) no mask is needed: the
//!   single query attends to the whole valid cache.
//! - **Continuous batching (`--b-max > 1`).** Supported: `MlaLatentCache`
//!   implements per-row migration (`adopt_row_from`) and the scheduler wires
//!   `LayerCache::Mla` into row-adoption + the dtype-finder, so B>1 batched
//!   prefill (different-length prompts) and B>1 heterogeneous-offset decode
//!   are both correct (verified bit-identical vs B=1 serial in
//!   `tests/glm4_moe_lite_cb.rs`).
//!
//! GLM is the first text-only model in this engine: the scheduler-facing
//! `DenseVlMethods` surface is implemented as a text-only stub that errors.

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::nn::{Embedding, LayerCache, Linear, RmsNorm};
use crate::Result;

use super::config::Glm4MoeLiteConfig;
use super::decoder_layer::Glm4DecoderLayer;
use super::mla_cache::MlaLatentCache;

const LONG_PREFILL_BATCH_LIMIT_THRESHOLD: usize = 1024;
const LONG_PREFILL_BATCH_LIMIT: usize = 2;

pub struct Glm4MoeLiteModel {
    embed_tokens: Embedding,
    layers: Vec<Glm4DecoderLayer>,
    norm: RmsNorm,
    /// Output projection (separate weight; `tie_word_embeddings = false`).
    lm_head: Linear,
    cfg: Glm4MoeLiteConfig,
}

/// Slice per-row last hidden states from `hidden [B, S, H]` → `[B, 1, H]`.
///
/// For row `i`, extracts `hidden[i, last_positions[i], :]` then concatenates
/// along axis 0. Mirrors `qwen3_5_moe::model::per_row_slice_last`: used by
/// [`Glm4MoeLiteModel::batched_prefill`] to project per-row last-token logits
/// when prompts have ragged lengths under right-padding.
fn per_row_slice_last(
    hidden: &Array,
    last_positions: &[i32],
    target: StreamOrDevice,
) -> Result<Array> {
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

/// Build the lower-right-aligned additive causal mask `[1, 1, L, Lc]` for an
/// internal prefill forward where the engine did not supply a mask.
///
/// Query relative position `q` (0-based, within the `L` new tokens, which sit
/// at the END of the `Lc`-length cache history) attends to cache positions
/// `0..=(Lc - L + q)`; later positions are `-inf`-masked. Identical semantics
/// to `generate::build_chunked_prefill_attention_mask` with
/// `chunk_start = Lc - L`, `chunk_len = L`, and replicates the implicit
/// `mask_mode="causal"` behaviour Qwen relies on. B is always 1 here because
/// this mask=None path only fires on the B=1 prefix-forward / chunked-prefill
/// callers (batched_prefill always supplies an explicit mask); the
/// `[1, 1, L, Lc]` mask broadcasts across heads.
fn build_internal_causal_mask(l: i32, lc: i32, dtype: Dtype) -> Result<Array> {
    let chunk_start = lc - l;
    if chunk_start < 0 {
        return Err(anyhow!(
            "build_internal_causal_mask: cache len Lc={lc} < query len L={l}"
        ));
    }
    let q_len = l as usize;
    let kv_len = lc as usize;
    let cs = chunk_start as usize;
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; q_len * kv_len];
    for q in 0..q_len {
        // Attend to all already-cached positions [0..chunk_start].
        for k in 0..cs {
            flat[q * kv_len + k] = 0.0;
        }
        // Causal within the new tokens: position q attends to [0..=q].
        for k in 0..=q {
            flat[q * kv_len + cs + k] = 0.0;
        }
    }
    let arr_f32: Array = (&flat[..], &[1_i32, 1_i32, l, lc][..]).try_into()?;
    Ok(mlx::ops::cast::astype(&arr_f32, dtype)?)
}

impl Glm4MoeLiteModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = Glm4MoeLiteConfig::from_loader(loader)?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Glm4MoeLiteConfig) -> Result<Self> {
        if cfg.tie_word_embeddings {
            return Err(anyhow!(
                "Glm4MoeLiteModel: tie_word_embeddings expected false (got true)"
            ));
        }
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")
            .context("loading Glm4MoeLiteModel embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(
                Glm4DecoderLayer::from_loader(loader, i, &cfg)
                    .with_context(|| format!("loading Glm4MoeLiteModel layer {i}"))?,
            );
        }
        let norm = RmsNorm::from_loader(loader, "model.norm", cfg.rms_norm_eps)
            .context("loading Glm4MoeLiteModel norm")?;
        let lm_head =
            Linear::from_loader(loader, "lm_head").context("loading Glm4MoeLiteModel lm_head")?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg,
        })
    }

    pub fn config(&self) -> &Glm4MoeLiteConfig {
        &self.cfg
    }

    /// Conservative weight-bytes estimate for memory budgeting (4-bit).
    /// Mirrors `Qwen35MoeModel::approx_weight_bytes`:
    ///   attn  : MLA projections (q_a, q_b, kv_a, o, embed_q, unembed_out)
    ///   dense : layer-0 SwiGLU MLP (gate, up, down)
    ///   moe   : routed experts (gate, up, down per expert) + shared expert
    ///   embed + lm_head (separate, `tie_word_embeddings = false`)
    fn approx_weight_bytes(&self) -> usize {
        let cfg = &self.cfg;
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let q_lora = cfg.q_lora_rank as usize;
        let kv_lora = cfg.kv_lora_rank as usize;
        let q_head_dim = cfg.q_head_dim() as usize;
        let n_heads = cfg.num_attention_heads as usize;
        let v_head = cfg.v_head_dim as usize;
        let qk_rope = cfg.qk_rope_head_dim as usize;
        let qk_nope = cfg.qk_nope_head_dim as usize;
        let e = cfg.n_routed_experts as usize;
        let n_shared = cfg.n_shared_experts as usize;
        let me = cfg.moe_intermediate_size as usize;
        let inter = cfg.intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;
        let dense_layers = (cfg.first_k_dense_replace as usize).min(l);
        let moe_layers = l.saturating_sub(dense_layers);

        // MLA per-layer projection params (4-bit → /2 bytes).
        let q_a = h * q_lora;
        let q_b = q_lora * (n_heads * q_head_dim);
        let kv_a = h * (kv_lora + qk_rope);
        let o = (n_heads * v_head) * h;
        let embed_q = n_heads * qk_nope * kv_lora;
        let unembed_out = n_heads * kv_lora * v_head;
        let attn = (q_a + q_b + kv_a + o + embed_q + unembed_out) * l / 2;

        // Dense (layer-0) SwiGLU: gate, up [h×inter], down [inter×h].
        let dense = 3 * h * inter * dense_layers / 2;
        // Routed experts + shared expert SwiGLU per MoE layer.
        let routed = 3 * e * h * me * moe_layers / 2;
        let shared = 3 * n_shared * h * me * moe_layers / 2;

        let embed_head = 2 * vocab * h / 2;
        attn + dense + routed + shared + embed_head
    }

    /// Per-layer latent MLA cache, one-shot preallocated to `cap` (via
    /// `with_step(cap)`) so the first decode step never triggers a grow.
    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let cfg = &self.cfg;
        let n = cfg.num_hidden_layers as usize;
        Ok((0..n)
            .map(|_| {
                LayerCache::Mla(
                    MlaLatentCache::new(batch, cfg.kv_lora_rank, cfg.qk_rope_head_dim, dtype, cap)
                        .with_step(cap),
                )
            })
            .collect())
    }

    /// Run embed → decoder layers → final norm, returning hidden states
    /// `[B, S, H]`. Threads the per-row RoPE offset (read from the layer-0
    /// pre-update cache length) and the attention mask through every layer.
    fn run_layers(
        &self,
        input_ids: &Array,
        per_row_lens: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let in_dims = input_ids.shape();
        let in_s = in_dims.as_slice();
        let batch = in_s[0];
        let seq_len = in_s[1];

        let caches = cache.ok_or_else(|| anyhow!("glm4_moe_lite requires a cache"))?;
        if caches.len() != self.layers.len() {
            return Err(anyhow!(
                "glm4_moe_lite: cache.len()={} != num_layers={}",
                caches.len(),
                self.layers.len()
            ));
        }

        let prl: Vec<i32> = per_row_lens
            .map(|s| s.to_vec())
            .unwrap_or_else(|| vec![seq_len; batch as usize]);
        if prl.len() != batch as usize {
            return Err(anyhow!(
                "glm4_moe_lite: per_row_lens.len()={} != batch={}",
                prl.len(),
                batch
            ));
        }
        // Regime uniformity is guaranteed STRUCTURALLY: the regime (prefill
        // `L > 1` vs decode `L == 1`) is decided by the single query length
        // `seq_len`, which is one scalar shared by every row of the `[B, L]`
        // input — there is no way to mix prefill and decode rows in one
        // forward. `per_row_lens` is NOT regime: it is the number of REAL
        // (non-pad) tokens each row writes to the cache, which legitimately
        // DIFFERS per row in B>1 batched prefill of different-length prompts
        // ([7, 21], …); the engine `attention_mask` masks the padding and the
        // per-row latent write (`MlaLatentCache::write_per_row`) writes each
        // row's leading-N slab independently. Per-row validity (`0 <= n <= L`)
        // is enforced by `update_and_fetch_on`, so no extra check is needed
        // here (mirrors the Qwen full-attention batched-prefill contract).

        // RoPE offset = pre-update per-row cache length (uniform across layers);
        // read from layer 0 before any layer writes its cache this step.
        let offsets_vec = match &caches[0] {
            LayerCache::Mla(c) => c.offsets().to_vec(),
            _ => {
                return Err(anyhow!(
                    "glm4_moe_lite: expected LayerCache::Mla at layer 0"
                ))
            }
        };
        let scalar_offset = (batch == 1).then_some(offsets_vec[0]);
        let per_row_offset: Option<Array> = if scalar_offset.is_some() {
            None
        } else {
            Some((&offsets_vec[..], &[batch][..]).try_into()?)
        };

        // Build the internal causal mask when the engine did not supply one and
        // this is a multi-token (prefill) forward. Decode (L == 1) needs none.
        let owned_mask: Option<Array> = match mask {
            Some(_) => None,
            None if seq_len > 1 => {
                // B is always 1 here: mask=None only happens on the B=1
                // prefix-forward / chunked-prefill paths (batched_prefill
                // always passes an explicit mask).
                let lc = offsets_vec.iter().copied().max().unwrap_or(0) + seq_len;
                Some(build_internal_causal_mask(seq_len, lc, Dtype::Bfloat16)?)
            }
            None => None,
        };
        let effective_mask: Option<&Array> = mask.or(owned_mask.as_ref());

        let mut h = self.embed_tokens.forward_on(input_ids, target)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let LayerCache::Mla(c) = &mut caches[i] else {
                return Err(anyhow!(
                    "glm4_moe_lite: expected LayerCache::Mla at layer {i}"
                ));
            };
            h = if let Some(offset) = scalar_offset {
                layer.forward_on_scalar_offset(
                    &h,
                    offset,
                    c,
                    &prl,
                    effective_mask,
                    target,
                    i as i32,
                )?
            } else {
                let offset = per_row_offset
                    .as_ref()
                    .expect("per_row_offset must exist for batch > 1");
                layer.forward_on(&h, offset, c, &prl, effective_mask, target, i as i32)?
            };
        }
        self.norm.forward_on(&h, target)
    }

    /// Single-stream forward returning last-position logits `[B, 1, vocab]`.
    ///
    /// `position_ids` is ignored (RoPE offset comes from the cache);
    /// `decode_mask` is the engine's per-row decode mask (passed through).
    pub fn forward_on(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self.run_layers(input_ids, per_row_lens, decode_mask, cache, target)?;
        // Slice the final sequence position → [B, 1, H], then project. For
        // decode (S == 1) this is a no-op slice; for the B=1 prefill prefix +
        // chunked last-chunk paths the last real token is at column S-1.
        let dims_borrow = hidden.shape();
        let dims = dims_borrow.as_slice();
        let (b, s, hsz) = (dims[0], dims[1], dims[2]);
        let last_hidden = if s > 1 {
            mlx::ops::indexing::slice_strided_on(
                &hidden,
                &[0_i32, s - 1, 0][..],
                &[b, s, hsz][..],
                &[1_i32, 1, 1][..],
                target,
            )?
        } else {
            hidden
        };
        {
            self.lm_head.forward_on(&last_hidden, target)
        }
    }

    /// Batched prefill returning per-row last-position logits `[B, 1, vocab]`.
    ///
    /// `position_ids` and `linear_attention_mask` are ignored (see module docs);
    /// `attention_mask` is the engine's `[B, 1, T, T]` additive causal mask.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_prefill(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        attention_mask: &Array,
        _linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let hidden = self.run_layers(
            input_ids,
            Some(per_row_lens),
            Some(attention_mask),
            cache,
            target,
        )?;
        let last_positions: Vec<i32> = per_row_lens.iter().map(|&l| l - 1).collect();
        let last_hidden = per_row_slice_last(&hidden, &last_positions, target)?;
        {
            self.lm_head.forward_on(&last_hidden, target)
        }
    }

    /// Run transformer + final norm, returning hidden state `[B, S, H]` (no
    /// lm_head). Used by intermediate chunked-prefill chunks.
    pub fn forward_text_hidden(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        self.run_layers(input_ids, per_row_lens, decode_mask, cache, target)
    }

    pub fn project_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.lm_head.forward_on(hidden, target.into())
    }

    pub fn model_meta(&self) -> ModelMeta {
        let cfg = &self.cfg;
        ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: Some(cfg.v_head_dim),
            weight_bytes: self.approx_weight_bytes(),
            max_position_embeddings: cfg.max_position_embeddings,
            spatial_merge_size: 2,
        }
    }
}

impl Model for Glm4MoeLiteModel {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        Glm4MoeLiteModel::make_cache(self, batch, cap, dtype)
    }

    fn fresh_prefill_batch_limit(prompt_len: usize, b_max: usize) -> usize {
        if prompt_len >= LONG_PREFILL_BATCH_LIMIT_THRESHOLD {
            b_max.min(LONG_PREFILL_BATCH_LIMIT)
        } else {
            b_max
        }
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
        Glm4MoeLiteModel::forward_on(
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
        Glm4MoeLiteModel::batched_prefill(
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
        Glm4MoeLiteModel::forward_text_hidden(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    fn project_hidden_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array> {
        Glm4MoeLiteModel::project_hidden_on(self, hidden, target)
    }

    fn requires_position_ids(&self) -> bool {
        false
    }

    fn model_meta(&self) -> ModelMeta {
        Glm4MoeLiteModel::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.cfg.num_hidden_layers as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_prefill_batch_limit_keeps_short_prompts_batched() {
        assert_eq!(
            <Glm4MoeLiteModel as Model>::fresh_prefill_batch_limit(512, 4),
            4
        );
    }

    #[test]
    fn fresh_prefill_batch_limit_caps_long_prompts_at_two() {
        assert_eq!(
            <Glm4MoeLiteModel as Model>::fresh_prefill_batch_limit(2048, 4),
            2
        );
    }

    #[test]
    fn fresh_prefill_batch_limit_keeps_long_b2_batched() {
        assert_eq!(
            <Glm4MoeLiteModel as Model>::fresh_prefill_batch_limit(2048, 2),
            2
        );
    }
}

impl crate::core::scheduler::DenseVlMethods for Glm4MoeLiteModel {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn batched_prefill_vl(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _attention_mask: &mlx::Array,
        _linear_attention_mask: &mlx::Array,
        _per_row_lens: &[i32],
        _per_row_pixel_values: &[Option<&[mlx::Array]>],
        _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        _image_token_id: i32,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }

    fn estimate_vision_prefill_peak_bytes(
        &self,
        _pixel_values: &[mlx::Array],
        _grid_thw: &[(i32, i32, i32)],
    ) -> crate::Result<usize> {
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }

    fn compute_vision_embeds(
        &self,
        _pixel_values: &[mlx::Array],
        _grid_thw: &[(i32, i32, i32)],
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }

    #[allow(clippy::too_many_arguments)]
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
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_vl_hidden(
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
        Err(anyhow!(
            "Glm4MoeLiteModel is text-only: VL methods unsupported"
        ))
    }
}
