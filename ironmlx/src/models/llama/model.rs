//! Standard Llama-family top-level model.
//!
//! `embed_tokens → [LlamaDecoderLayer; num_hidden_layers] → norm → lm_head`.
//! Separate `lm_head` (`tie_word_embeddings = false` for MiniCPM5-1B).
//!
//! # Engine contract notes
//! - **`position_ids` is ignored.** Like `glm4_moe_lite`, the RoPE offset is
//!   derived from the per-row pre-update cache length
//!   ([`KVCache::offsets`]), so [`Model::requires_position_ids`] returns
//!   `false` and the scheduler passes a placeholder.
//! - **`linear_attention_mask` is ignored.** There is no linear-attention path;
//!   `batched_prefill` accepts the argument for trait-shape parity only.
//! - **Regime is per-call uniform (structurally).** A single forward is either
//!   all-prefill (`L > 1`) or all-decode (`L == 1`), decided by the shared
//!   query length `L`. `per_row_lens` is the count of REAL tokens each row
//!   writes and legitimately differs per row in B>1 batched prefill.
//! - **Causal mask.** When the caller passes `mask = None` with `L > 1`
//!   (B=1 prefix-forward / chunked-prefill), the model builds its own
//!   lower-right-aligned additive causal mask, mirroring the implicit SDPA
//!   `mask_mode = "causal"`. `Some(..)` (e.g. `batched_prefill`'s engine mask)
//!   is passed straight through. Decode (`L == 1`) needs no mask.
//!
//! Text-only: the scheduler-facing `DenseVlMethods` surface is a stub that
//! errors (mirrors `glm4_moe_lite`).

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::KVCache;
use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::nn::{Embedding, LayerCache, Linear, RmsNorm};
use crate::Result;

use super::config::LlamaConfig;
use super::decoder_layer::LlamaDecoderLayer;

// Long B>1 Llama prefill produces an oversized shape-specialized MLX graph.
// Admit one fresh row and let rolling admission chunk additional requests.
const LONG_PREFILL_BATCH_LIMIT_THRESHOLD: usize = 4096;
const LONG_PREFILL_BATCH_LIMIT: usize = 1;

pub struct LlamaModel {
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    /// Output projection (separate weight; `tie_word_embeddings = false`).
    lm_head: Linear,
    exact_batched_verify_precision_qualified: bool,
    cfg: LlamaConfig,
}

/// Slice per-row last hidden states from `hidden [B, S, H]` → `[B, 1, H]`.
///
/// For row `i`, extracts `hidden[i, last_positions[i], :]` then concatenates
/// along axis 0. Used by [`LlamaModel::batched_prefill`] to project per-row
/// last-token logits when prompts have ragged lengths under right-padding.
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
/// Query relative position `q` (within the `L` new tokens at the END of the
/// `Lc`-length cache history) attends to cache positions `0..=(Lc - L + q)`;
/// later positions are `-inf`. B is always 1 here (this `mask = None` path only
/// fires on the B=1 prefix-forward / chunked-prefill callers); the
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
        for k in 0..cs {
            flat[q * kv_len + k] = 0.0;
        }
        for k in 0..=q {
            flat[q * kv_len + cs + k] = 0.0;
        }
    }
    let arr_f32: Array = (&flat[..], &[1_i32, 1_i32, l, lc][..]).try_into()?;
    Ok(mlx::ops::cast::astype(&arr_f32, dtype)?)
}

impl LlamaModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg = LlamaConfig::from_loader(loader)?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: LlamaConfig) -> Result<Self> {
        if cfg.tie_word_embeddings {
            return Err(anyhow!(
                "LlamaModel: tie_word_embeddings = true not supported (MiniCPM5-1B \
                 ships a separate lm_head); got true"
            ));
        }
        let embed_tokens = Embedding::from_loader(loader, "model.embed_tokens")
            .context("loading LlamaModel embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(
                LlamaDecoderLayer::from_loader(loader, i, &cfg)
                    .with_context(|| format!("loading LlamaModel layer {i}"))?,
            );
        }
        let norm = RmsNorm::from_loader(loader, "model.norm", cfg.rms_norm_eps)
            .context("loading LlamaModel norm")?;
        let lm_head =
            Linear::from_loader(loader, "lm_head").context("loading LlamaModel lm_head")?;
        let exact_batched_verify_precision_qualified =
            super::speculative::exact_batched_verify_precision_qualified(
                loader.quant_meta_for("lm_head"),
                loader
                    .config_raw_value()
                    .get("torch_dtype")
                    .and_then(serde_json::Value::as_str),
            );
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            exact_batched_verify_precision_qualified,
            cfg,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// Conservative weight-bytes estimate for memory budgeting (8-bit affine ≈
    /// 1 byte/param; scales/biases overhead ignored).
    fn approx_weight_bytes(&self) -> usize {
        let cfg = &self.cfg;
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let hd = cfg.effective_head_dim() as usize;
        let nh = cfg.num_attention_heads as usize;
        let nkv = cfg.num_key_value_heads as usize;
        let inter = cfg.intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;

        // q,k,v,o projections per layer.
        let attn = (h * nh * hd + 2 * h * nkv * hd + nh * hd * h) * l;
        // SwiGLU: gate, up [h×inter], down [inter×h].
        let mlp = 3 * h * inter * l;
        // embed_tokens + lm_head (separate).
        let embed_head = 2 * vocab * h;
        attn + mlp + embed_head
    }

    /// Per-layer KV cache, one-shot preallocated to `cap` (via `with_step(cap)`)
    /// so the first decode step never triggers a grow.
    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let hd = self.cfg.effective_head_dim();
        let nkv = self.cfg.num_key_value_heads;
        Ok((0..self.layers.len())
            .map(|_| LayerCache::Full(KVCache::new(batch, nkv, hd, hd, dtype, cap).with_step(cap)))
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
        let exact_batched_verify = crate::nn::verify_qmm::is_armed()
            && self.exact_batched_verify_precision_qualified
            && (1..=8).contains(&batch)
            && (2..=5).contains(&seq_len);
        let _position_stable_qmm = exact_batched_verify.then(crate::nn::position_stable_qmm::scope);

        let caches = cache.ok_or_else(|| anyhow!("llama requires a cache"))?;
        if caches.len() != self.layers.len() {
            return Err(anyhow!(
                "llama: cache.len()={} != num_layers={}",
                caches.len(),
                self.layers.len()
            ));
        }

        let prl: Vec<i32> = per_row_lens
            .map(|s| s.to_vec())
            .unwrap_or_else(|| vec![seq_len; batch as usize]);
        if prl.len() != batch as usize {
            return Err(anyhow!(
                "llama: per_row_lens.len()={} != batch={}",
                prl.len(),
                batch
            ));
        }

        // RoPE offset = pre-update per-row cache length (uniform across layers);
        // read from layer 0 before any layer writes its cache this step.
        let offsets_vec = match &caches[0] {
            LayerCache::Full(c) => c.offsets().to_vec(),
            _ => return Err(anyhow!("llama: expected LayerCache::Full at layer 0")),
        };
        let offset: Array = (&offsets_vec[..], &[batch][..]).try_into()?;

        // Build the internal causal mask when the engine did not supply one and
        // this is a multi-token (prefill) forward. Decode (L == 1) needs none.
        let owned_mask: Option<Array> = match mask {
            Some(_) => None,
            None if seq_len > 1 => {
                let lc = offsets_vec.iter().copied().max().unwrap_or(0) + seq_len;
                Some(build_internal_causal_mask(seq_len, lc, Dtype::Bfloat16)?)
            }
            None => None,
        };
        let effective_mask: Option<&Array> = mask.or(owned_mask.as_ref());

        let mut h = self.embed_tokens.forward_on(input_ids, target)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let LayerCache::Full(c) = &mut caches[i] else {
                return Err(anyhow!("llama: expected LayerCache::Full at layer {i}"));
            };
            h = layer.forward_on(
                &h,
                &offset,
                &offsets_vec,
                c,
                &prl,
                effective_mask,
                exact_batched_verify,
                target,
            )?;
        }
        self.norm.forward_on(&h, target)
    }

    /// Single-stream forward returning last-position logits `[B, 1, vocab]`.
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
        self.lm_head.forward_on(&last_hidden, target)
    }

    /// Batched prefill returning per-row last-position logits `[B, 1, vocab]`.
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
        self.lm_head.forward_on(&last_hidden, target)
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
        let target = target.into();
        if self.exact_batched_verify_precision_qualified
            && hidden.ndim() == 3
            && hidden.shape().as_slice()[1] > 1
        {
            self.lm_head.forward_positions_isolated_on(hidden, target)
        } else {
            self.lm_head.forward_on(hidden, target)
        }
    }

    pub fn model_meta(&self) -> ModelMeta {
        let cfg = &self.cfg;
        ModelMeta {
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_size: cfg.hidden_size,
            head_dim: Some(cfg.effective_head_dim()),
            weight_bytes: self.approx_weight_bytes(),
            max_position_embeddings: cfg.max_position_embeddings,
            spatial_merge_size: 2,
        }
    }
}

impl Model for LlamaModel {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        LlamaModel::make_cache(self, batch, cap, dtype)
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
        LlamaModel::forward_on(
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
        LlamaModel::batched_prefill(
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
        LlamaModel::forward_text_hidden(
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
        LlamaModel::project_hidden_on(self, hidden, target)
    }

    fn requires_position_ids(&self) -> bool {
        false
    }

    fn supports_exact_batched_speculative_verify(
        &self,
        batch_width: usize,
        context_tokens: usize,
        verify_width: usize,
    ) -> bool {
        super::speculative::exact_batched_verify_qualified(
            self.exact_batched_verify_precision_qualified,
            batch_width,
            context_tokens,
            verify_width,
        )
    }

    fn supports_speculative_accepted_prefix_trim(&self) -> bool {
        true
    }

    fn model_meta(&self) -> ModelMeta {
        LlamaModel::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.cfg.num_hidden_layers as usize
    }
}

impl crate::core::scheduler::DenseVlMethods for LlamaModel {
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
        Err(anyhow!("LlamaModel is text-only: VL methods unsupported"))
    }

    fn estimate_vision_prefill_peak_bytes(
        &self,
        _pixel_values: &[mlx::Array],
        _grid_thw: &[(i32, i32, i32)],
    ) -> crate::Result<usize> {
        Err(anyhow!("LlamaModel is text-only: VL methods unsupported"))
    }

    fn compute_vision_embeds(
        &self,
        _pixel_values: &[mlx::Array],
        _grid_thw: &[(i32, i32, i32)],
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Err(anyhow!("LlamaModel is text-only: VL methods unsupported"))
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
        Err(anyhow!("LlamaModel is text-only: VL methods unsupported"))
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
        Err(anyhow!("LlamaModel is text-only: VL methods unsupported"))
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaModel;
    use crate::core::Model;

    #[test]
    fn fresh_prefill_batch_limit_keeps_short_prompts_batched() {
        assert_eq!(<LlamaModel as Model>::fresh_prefill_batch_limit(2048, 4), 4);
    }

    #[test]
    fn fresh_prefill_batch_limit_serializes_long_prompts() {
        assert_eq!(<LlamaModel as Model>::fresh_prefill_batch_limit(4095, 4), 4);
        assert_eq!(<LlamaModel as Model>::fresh_prefill_batch_limit(4096, 4), 1);
        assert_eq!(
            <LlamaModel as Model>::fresh_prefill_batch_limit(32_619, 4),
            1
        );
    }
}
