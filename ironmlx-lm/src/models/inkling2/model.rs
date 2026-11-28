//! Native looped Inkling model, integrated with the shared causal scheduler.
use super::{
    config::Inkling2Config, layers::DecoderLayer, metal::ActivationTables, ops::*,
    vision::VisionTower,
};
use crate::core::{
    cache::{
        layer::{LayerCache, LayerCacheSnapshot},
        GatedDeltaCache, KVCache,
    },
    model::{CacheCost, ModelMeta, PrefillPeak},
    vision::DenseVlMethods,
    Loader, Model,
};
use crate::nn::{Embedding, Linear, RmsNorm};
use anyhow::{ensure, Context, Result};
use mlx::ops::indexing;
use mlx::{Array, Dtype, StreamOrDevice};

pub struct Inkling2Model {
    config: Inkling2Config,
    oq: bool,
    embedding: Embedding,
    embed_norm: RmsNorm,
    norm: RmsNorm,
    head: Linear,
    layers: Vec<DecoderLayer>,
    final_query: Array,
    message_query: Option<Array>,
    vision: Option<VisionTower>,
    weight_bytes: usize,
    verification: std::sync::Mutex<Verification>,
    /// Set when a prefix cache will save and restore K/V: sliding-window
    /// layers then keep their whole history instead of just the window.
    full_kv_history: std::sync::atomic::AtomicBool,
    /// Whether the global layers' caches store K/V in 8-bit affine groups,
    /// which single-token decode reads directly and prefix caches save as
    /// they are. Off with `IRONMLX_INKLING_KV_INT8=0` or after `use_dense_kv`.
    int8_kv: std::sync::atomic::AtomicBool,
}

/// Positions a sliding-window layer keeps behind its window, so that
/// verification rows appended one at a time can be rolled back together.
const WINDOW_ROLLBACK: i32 = 64;

#[derive(Default)]
enum Verification {
    #[default]
    Idle,
    Serial(Vec<Vec<LayerCacheSnapshot>>),
    Batched(Vec<Array>),
}

impl Inkling2Model {
    pub fn from_loader(l: &Loader) -> Result<Self> {
        let config = Inkling2Config::from_loader(l)?;
        let c = &config.text_config;
        let p = "model.language_model";
        let tables = if super::layers::is_gpu(StreamOrDevice::default()) {
            Some(ActivationTables::new(
                c.mlp_gate_softcap,
                c.mlp_up_softcap,
                StreamOrDevice::default(),
            )?)
        } else {
            None
        };
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                DecoderLayer::load(l, c, i, tables.clone())
                    .with_context(|| format!("Inkling layer {i}"))
            })
            .collect::<Result<_>>()?;
        let vision = if l.contains("model.vision_tower.embeddings.patch_embedding.weight") {
            config
                .vision_config
                .as_ref()
                .map(|v| VisionTower::load(l, v))
                .transpose()?
        } else {
            None
        };
        Ok(Self {
            embedding: Embedding::from_loader(l, &format!("{p}.embed_tokens"))?,
            embed_norm: RmsNorm::from_loader(l, &format!("{p}.embed_norm"), c.rms_norm_eps)?,
            norm: RmsNorm::from_loader(l, &format!("{p}.norm"), c.rms_norm_eps)?,
            head: Linear::from_loader(l, "lm_head")?,
            layers,
            final_query: l.tensor(&format!("{p}.final_input.query"))?.clone(),
            message_query: if c.loop_config.is_some() {
                Some(l.tensor(&format!("{p}.message_input.query"))?.clone())
            } else {
                None
            },
            vision,
            weight_bytes: l.loaded_tensor_bytes(),
            verification: std::sync::Mutex::default(),
            full_kv_history: std::sync::atomic::AtomicBool::new(false),
            int8_kv: std::sync::atomic::AtomicBool::new(
                std::env::var_os("IRONMLX_INKLING_KV_INT8").as_deref()
                    != Some(std::ffi::OsStr::new("0")),
            ),
            oq: super::is_oq_checkpoint(l.config_raw_value()),
            config,
        })
    }
    /// The window a visit's K/V cache keeps; 0 keeps the whole history.
    fn kept_window(&self, layer: usize) -> i32 {
        let c = &self.config.text_config;
        let full = self
            .full_kv_history
            .load(std::sync::atomic::Ordering::Relaxed);
        if c.is_local(layer) && !full {
            c.sliding_window_size
        } else {
            0
        }
    }
    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }
    fn embed(&self, ids: &Array, t: StreamOrDevice) -> Result<Array> {
        self.embed_norm.forward_on(
            &self
                .embedding
                .forward_on(&ids.astype_on(Dtype::Int32, t)?, t)?,
            t,
        )
    }
    fn last_logits(
        &self,
        hidden: &Array,
        lens: Option<&[i32]>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        let b = hidden.shape_at(0);
        let s = hidden.shape_at(1);
        if b == 1 {
            let end = lens.map_or(s, |lens| lens[0]).max(1);
            if s == 1 {
                return self.project_hidden_on(hidden, t);
            }
            return self.project_hidden_on(&slice(hidden, 1, end - 1, end, t)?, t);
        }
        let positions: Vec<i32> = (0..b as usize)
            .map(|row| lens.map_or(s, |l| l[row]).max(1) - 1)
            .collect();
        let positions: Array = (positions.as_slice(), &[b, 1, 1][..]).try_into()?;
        self.project_hidden_on(&indexing::take_along_axis_on(hidden, &positions, 1, t)?, t)
    }
    fn run(
        &self,
        embeddings: Array,
        lens: Option<&[i32]>,
        caches: Option<&mut [LayerCache]>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        self.run_with_taps(embeddings, lens, caches, &[], None, t)
            .map(|(hidden, _)| hidden)
    }

    fn run_with_taps(
        &self,
        embeddings: Array,
        lens: Option<&[i32]>,
        caches: Option<&mut [LayerCache]>,
        taps: &[usize],
        verify_inputs: Option<&mut Vec<Array>>,
        t: StreamOrDevice,
    ) -> Result<(Array, Vec<Array>)> {
        super::layers::with_checkpoint_layout(self.oq, || {
            self.run_target(embeddings, lens, caches, taps, verify_inputs, t)
        })
    }

    fn run_target(
        &self,
        embeddings: Array,
        lens: Option<&[i32]>,
        caches: Option<&mut [LayerCache]>,
        taps: &[usize],
        mut verify_inputs: Option<&mut Vec<Array>>,
        t: StreamOrDevice,
    ) -> Result<(Array, Vec<Array>)> {
        let c = &self.config.text_config;
        let visits = c.visits();
        let (b, s) = (embeddings.shape_at(0), embeddings.shape_at(1));
        let lens = lens
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| vec![s; b as usize]);
        ensure!(
            lens.len() == b as usize && lens.iter().all(|n| *n >= 0 && *n <= s),
            "invalid Inkling per-row lengths"
        );
        let caches = caches.ok_or_else(|| {
            anyhow::anyhow!("Inkling requires per-visit KV and convolution caches")
        })?;
        ensure!(
            caches.len() == visits.len() * 2,
            "Inkling cache count differs from loop visits"
        );
        let mut x = embeddings;
        let dtype = x.dtype();
        let mut blocks = Vec::new();
        let mut prelude: Option<(Array, Vec<Array>)> = None;
        let mut bank: Option<Array> = None;
        let mut captured = Vec::with_capacity(taps.len());
        // Keep the GPU working while the host builds later recurrent visits.
        let submit_visits = self.oq && b == 1 && s <= 8 && super::layers::is_gpu(t);
        for (visit, i) in visits.iter().copied().enumerate() {
            if let Some(loop_config) = &c.loop_config {
                if i == loop_config.core_start {
                    if let Some((pre_x, pre_blocks)) = &prelude {
                        bank = Some(depth_states(
                            &blocks[pre_blocks.len() + 1..],
                            &norm(&x.astype_on(Dtype::Float32, t)?, t)?,
                            t,
                        )?);
                        x = pre_x.clone();
                        blocks = pre_blocks.clone();
                    } else {
                        prelude = Some((x.clone(), blocks.clone()));
                    }
                }
            }
            if taps.contains(&visit) {
                captured.push(x.clone());
            }
            let messages = if let (Some(states), Some(loop_config), Some(query)) =
                (&bank, &c.loop_config, &self.message_query)
            {
                if (loop_config.core_start..loop_config.core_end).contains(&i) {
                    Some(mix_depth_states(
                        states,
                        &slice(
                            query,
                            0,
                            (i - loop_config.core_start) as i32,
                            (i - loop_config.core_start + 1) as i32,
                            t,
                        )?
                        .squeeze_on(0, t)?,
                        t,
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
            let pair = &mut caches[visit * 2..visit * 2 + 2];
            let (left, right) = pair.split_at_mut(1);
            let (LayerCache::Full(kv), LayerCache::Linear(conv)) = (&mut left[0], &mut right[0])
            else {
                anyhow::bail!("Inkling cache layout must alternate KV and convolution state")
            };
            x = self.layers[i].forward(
                &x,
                &mut blocks,
                messages.as_ref(),
                kv,
                conv,
                &lens,
                verify_inputs.as_deref_mut(),
                t,
            )?;
            if submit_visits && (visit + 1) % 4 == 0 && visit + 1 < visits.len() {
                mlx::transforms::async_eval(&[&x])?;
            }
        }
        let final_input = depth_attention(
            &blocks,
            &norm(&x.astype_on(Dtype::Float32, t)?, t)?,
            &self.final_query,
            t,
        )?
        .squeeze_on(2, t)?;
        let hidden = self.norm.forward_on(&final_input, t)?.astype_on(dtype, t)?;
        if taps.contains(&visits.len()) {
            captured.push(hidden.clone());
        }
        Ok((hidden, captured))
    }
    fn splice(
        &self,
        ids: &Array,
        embeddings: Array,
        vision: Option<&Array>,
        image_token: i32,
        t: StreamOrDevice,
    ) -> Result<Array> {
        let Some(vision) = vision else {
            return Ok(embeddings);
        };
        let flat = ids
            .astype_on(Dtype::Int32, t)?
            .reshape_on((-1,), t)?
            .to_vec::<i32>()?;
        let mut next = 0;
        let mut indices = Vec::with_capacity(flat.len());
        let mut mask = Vec::with_capacity(flat.len());
        for token in flat {
            let image = token == image_token;
            mask.push(image);
            indices.push(if image {
                let i = next;
                next += 1;
                i
            } else {
                0
            });
        }
        ensure!(
            next == vision.shape_at(0),
            "Inkling image placeholder count differs from encoded image rows"
        );
        if next == 0 {
            return Ok(embeddings);
        }
        let (b, s, d) = (
            embeddings.shape_at(0),
            embeddings.shape_at(1),
            embeddings.shape_at(2),
        );
        let indices: Array = (indices.as_slice(), &[b, s][..]).try_into()?;
        let mask: Array = (mask.as_slice(), &[b, s, 1][..]).try_into()?;
        let values = indexing::take_on(vision, &indices, 0, t)?
            .reshape_on((b, s, d), t)?
            .astype_on(embeddings.dtype(), t)?;
        Ok(mask.where_on(&values, &embeddings, t)?)
    }
}
impl Model for Inkling2Model {
    fn cache_dtype(&self) -> Dtype {
        self.embedding.output_dtype()
    }
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        let c = &self.config.text_config;
        let mut caches = Vec::new();
        // Single-token decode of the oQ checkpoint reads 8-bit global K/V.
        let int8 = self.oq && batch == 1 && self.int8_kv.load(std::sync::atomic::Ordering::Relaxed);
        for i in c.visits() {
            let (_, kv, d) = c.attention_dims(i);
            let mut cache = KVCache::new(batch, kv, d, d, dtype, cap)
                .with_step(crate::core::cache::kv_growth_step(cap))
                .with_window(self.kept_window(i), WINDOW_ROLLBACK);
            if int8 && !c.is_local(i) {
                cache.enable_int8();
            }
            caches.push(LayerCache::Full(cache));
            // Reuse the scheduler's row-addressable rolling-state storage. All four
            // convolution windows are packed along channels; no SSM state is used.
            caches.push(LayerCache::Linear(GatedDeltaCache::new_with_cap(
                batch,
                c.sconv_kernel_size,
                2 * kv * d + 2 * c.hidden_size,
                1,
                1,
                1,
                Dtype::Float32,
                cap,
            )?));
        }
        Ok(caches)
    }
    fn keep_full_kv_history(&self) {
        self.full_kv_history
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    fn use_dense_kv(&self) {
        self.int8_kv
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
    fn prefix_cache_keeps_dense_kv(&self) -> bool {
        // Every attention path here takes K/V as one contiguous array.
        true
    }
    fn forward_on(
        &self,
        ids: &Array,
        _positions: &Array,
        lens: Option<&[i32]>,
        _mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        self.last_logits(&self.run(self.embed(ids, t)?, lens, cache, t)?, lens, t)
    }
    fn batched_prefill(
        &self,
        ids: &Array,
        _positions: &Array,
        _attention: &Array,
        _linear: &Array,
        lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        self.last_logits(
            &self.run(self.embed(ids, t)?, Some(lens), cache, t)?,
            Some(lens),
            t,
        )
    }
    fn forward_text_hidden(
        &self,
        ids: &Array,
        _positions: &Array,
        lens: Option<&[i32]>,
        _mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        self.run(self.embed(ids, t)?, lens, cache, t)
    }
    fn project_hidden_on(&self, hidden: &Array, t: StreamOrDevice) -> Result<Array> {
        let logits = self
            .head
            .forward_on(hidden, t)?
            .astype_on(Dtype::Float32, t)?;
        slice(
            &cap(&logits, self.config.text_config.final_logit_softcapping, t)?,
            2,
            0,
            self.config.text_config.unpadded_vocab_size,
            t,
        )
    }
    fn requires_position_ids(&self) -> bool {
        false
    }
    fn prefers_unsplit_b1_prefill(&self) -> bool {
        self.oq
    }
    fn vl_positions_sequential(&self) -> bool {
        true
    }
    fn model_meta(&self) -> ModelMeta {
        let c = &self.config.text_config;
        let mut kv_bytes = 0;
        let mut conv_bytes = 0;
        // A sliding-window layer holds its window, the rollback margin and
        // the widest write, whatever the sequence length.
        let chunk = super::b1_prefill_chunk_size(self.oq).unwrap_or(2048) as i32;
        for i in c.visits() {
            let (_, kv, d) = c.attention_dims(i);
            match self.kept_window(i) {
                0 => kv_bytes += kv * d * 4,
                window => {
                    let held = (window - 1 + WINDOW_ROLLBACK + chunk.max(window) + 255) / 256 * 256;
                    conv_bytes += held * kv * d * 4;
                }
            }
            conv_bytes += ((c.sconv_kernel_size - 1) * (2 * kv * d + 2 * c.hidden_size) + 1) * 4;
        }
        ModelMeta {
            num_hidden_layers: self.num_hidden_layers() as i32,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads,
            hidden_size: c.hidden_size,
            head_dim: Some(c.head_dim),
            weight_bytes: self.weight_bytes,
            max_position_embeddings: c.model_max_length,
            spatial_merge_size: self
                .config
                .vision_config
                .as_ref()
                .map_or(1, |v| v.adapter_merge_size),
            cache_cost: Some(CacheCost {
                bytes_per_token: kv_bytes as usize,
                fixed_bytes_per_sequence: conv_bytes as usize,
            }),
            // The oQ checkpoint's single-row prefill runs fused attention
            // (`band` / `band_window`), which never holds a score matrix:
            // measured on M5 over a 59K prompt, a chunk's transient is
            // about 1.0 GiB plus 0.3 MiB per token whatever the context
            // (1.6 GiB at 2048 tokens, 2.7 GiB at 4096, 2.0 GiB for the
            // first chunk). The figures below leave room above that.
            prefill_peak: self.oq.then_some(PrefillPeak {
                fixed_bytes: 1280 << 20,
                bytes_per_token: 400 << 10,
            }),
        }
    }
    fn num_hidden_layers(&self) -> usize {
        self.config.text_config.visits().len() * 2
    }
}
impl DenseVlMethods for Inkling2Model {
    fn batched_prefill_vl(
        &self,
        ids: &Array,
        _positions: &Array,
        _attention: &Array,
        _linear: &Array,
        lens: &[i32],
        pixels: &[Option<&[Array]>],
        grids: &[Option<&[(i32, i32, i32)]>],
        image_token: i32,
        cache: Option<&mut [LayerCache]>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        ensure!(
            pixels.len() == ids.shape_at(0) as usize && grids.len() == pixels.len(),
            "Inkling vision batch length mismatch"
        );
        let mut rows = Vec::new();
        for (row, pixel) in pixels.iter().enumerate() {
            let row_ids = slice(ids, 0, row as i32, row as i32 + 1, t)?;
            let embeds = self.embed(&row_ids, t)?;
            let vision = pixel
                .map(|p| {
                    self.compute_vision_embeds(
                        p,
                        grids[row].ok_or_else(|| anyhow::anyhow!("missing Inkling image grids"))?,
                        t,
                    )
                })
                .transpose()?;
            rows.push(self.splice(&row_ids, embeds, vision.as_ref(), image_token, t)?);
        }
        self.last_logits(
            &self.run(
                cat(&rows.iter().collect::<Vec<_>>(), 0, t)?,
                Some(lens),
                cache,
                t,
            )?,
            Some(lens),
            t,
        )
    }
    fn estimate_vision_prefill_tensor_bytes(
        &self,
        pixels: &[Array],
        grids: &[(i32, i32, i32)],
    ) -> Result<usize> {
        let c = self
            .config
            .vision_config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Inkling has no vision tower"))?;
        ensure!(
            pixels.len() == grids.len() && !pixels.is_empty(),
            "Inkling requires one grid per image"
        );
        crate::core::vision::estimate_transformer_vision_prefill_tensor_bytes(
            pixels,
            grids,
            crate::core::vision::VisionPrefillMemoryProfile {
                hidden_size: c.hidden_size as usize,
                intermediate_size: (c.hidden_size as f32 * c.mlp_ratio) as usize,
                num_attention_heads: c.num_attention_heads as usize,
                output_hidden_size: self.config.text_config.hidden_size as usize,
                spatial_merge_area: (c.adapter_merge_size * c.adapter_merge_size) as usize,
                activation_bytes: 4,
                attention: crate::core::vision::VisionAttention::FusedSdpa {
                    head_dim: (c.hidden_size / c.num_attention_heads) as usize,
                },
            },
        )
    }
    fn compute_vision_embeds(
        &self,
        pixels: &[Array],
        grids: &[(i32, i32, i32)],
        t: StreamOrDevice,
    ) -> Result<Array> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Inkling vision weights were not loaded"))?;
        let c = self
            .config
            .vision_config
            .as_ref()
            .expect("loaded vision config");
        ensure!(
            !pixels.is_empty() && pixels.len() == grids.len(),
            "Inkling requires one grid per image"
        );
        let features = pixels
            .iter()
            .zip(grids)
            .map(|(p, &(time, h, w))| {
                ensure!(
                    time == 1
                        && p.shape_at(1) == h * c.native_patch_size
                        && p.shape_at(2) == w * c.native_patch_size,
                    "Inkling image grid mismatch"
                );
                tower.forward(p, t)
            })
            .collect::<Result<Vec<_>>>()?;
        cat(&features.iter().collect::<Vec<_>>(), 0, t)
    }
    fn forward_vl_chunk(
        &self,
        ids: &Array,
        positions: &Array,
        lens: Option<&[i32]>,
        mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision: Option<&Array>,
        image_token: i32,
        t: StreamOrDevice,
    ) -> Result<Array> {
        self.last_logits(
            &self.forward_vl_hidden(ids, positions, lens, mask, cache, vision, image_token, t)?,
            lens,
            t,
        )
    }
    fn forward_vl_hidden(
        &self,
        ids: &Array,
        _positions: &Array,
        lens: Option<&[i32]>,
        _mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision: Option<&Array>,
        image_token: i32,
        t: StreamOrDevice,
    ) -> Result<Array> {
        self.run(
            self.splice(ids, self.embed(ids, t)?, vision, image_token, t)?,
            lens,
            cache,
            t,
        )
    }
}

use crate::models::dflash2::{
    DFlash2Target, DFlash2TargetCacheCost, DFlash2TargetForwardMode, DFlash2TargetOutput,
    DFlash2TargetSpec, DFlash2VerifyCapabilities, DFlash2VerifyShape,
};

impl Inkling2Model {
    fn dspark_prefill(
        &self,
        embeddings: Array,
        cache: Option<&mut [LayerCache]>,
        taps: &[usize],
        verify_inputs: Option<&mut Vec<Array>>,
        t: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        let (hidden, features) =
            self.run_with_taps(embeddings, None, cache, taps, verify_inputs, t)?;
        ensure!(
            features.len() == taps.len() && !features.is_empty(),
            "invalid Inkling DSpark feature IDs"
        );
        let context_hidden = cat(&features.iter().collect::<Vec<_>>(), -1, t)?;
        Ok(DFlash2TargetOutput {
            hidden,
            context_hidden,
        })
    }
}

impl DFlash2Target for Inkling2Model {
    fn dflash2_target_spec(&self) -> DFlash2TargetSpec {
        let c = &self.config.text_config;
        DFlash2TargetSpec {
            hidden_size: c.hidden_size,
            vocab_size: c.unpadded_vocab_size,
            max_position_embeddings: c.model_max_length,
            num_hidden_layers: c.visits().len() as i32,
            rms_norm_eps: c.rms_norm_eps,
            rope_theta: 1_000_000.0,
        }
    }

    fn dflash2_target_cache_cost(&self) -> DFlash2TargetCacheCost {
        let cache = self
            .model_meta()
            .cache_cost
            .expect("Inkling provides a cache cost");
        DFlash2TargetCacheCost {
            bytes_per_token: cache.bytes_per_token,
            fixed_bytes_per_sequence: cache.fixed_bytes_per_sequence,
        }
    }

    fn dflash2_verify_capabilities(&self) -> DFlash2VerifyCapabilities {
        DFlash2VerifyCapabilities {
            profile: "inkling2-row-exact".into(),
            row_bit_exact_qmm: true,
            row_bit_exact_attention: true,
            transactional_state_restore: true,
            supported_shapes: (2..=8)
                .map(|width| DFlash2VerifyShape {
                    batch_width: 1,
                    verify_width: width,
                })
                .collect(),
            lane_kernel_pack: None,
            flat_tree_max_nodes: 0,
            ragged_rows_only: false,
        }
    }

    fn dflash2_execution_fingerprint(&self) -> String {
        // A cached prefix holds either every layer's whole K/V or the
        // sliding-window layers' windows; the two do not restore into each
        // other.
        let full = self
            .full_kv_history
            .load(std::sync::atomic::Ordering::Relaxed);
        format!(
            "inkling2-dspark;oq={};m5={};kv={}",
            self.oq,
            super::layers::m5_available(),
            if full { "full" } else { "window" }
        )
    }

    fn dflash2_embed_on(&self, ids: &Array, t: StreamOrDevice) -> Result<Array> {
        self.embed(ids, t)
    }

    fn dflash2_supports_images(&self) -> bool {
        self.vision.is_some()
    }

    fn dflash2_load_vision(&self, _on_loaded: &dyn Fn(&Loader)) -> Result<bool> {
        ensure!(
            self.vision.is_some(),
            "Inkling checkpoint has no vision tower"
        );
        Ok(false)
    }

    fn dflash2_vision_embeds_on(
        &self,
        pixels: &[Array],
        grid: &[(i32, i32, i32)],
        t: StreamOrDevice,
    ) -> Result<Array> {
        self.compute_vision_embeds(pixels, grid, t)
    }

    fn dflash2_forward_target_vl_on(
        &self,
        ids: &Array,
        image: &Array,
        image_token: i32,
        _positions: &Array,
        cache: Option<&mut [LayerCache]>,
        taps: &[usize],
        t: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        let embeddings = self.splice(ids, self.embed(ids, t)?, Some(image), image_token, t)?;
        self.dspark_prefill(embeddings, cache, taps, None, t)
    }

    fn dflash2_forward_target_on(
        &self,
        ids: &Array,
        _positions: &Array,
        cache: Option<&mut [LayerCache]>,
        taps: &[usize],
        mode: DFlash2TargetForwardMode,
        t: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        let mut verification = self
            .verification
            .lock()
            .map_err(|_| anyhow::anyhow!("Inkling verify state poisoned"))?;
        *verification = Verification::Idle;
        if !mode.is_verify() || ids.shape_at(1) == 1 {
            return self.dspark_prefill(self.embed(ids, t)?, cache, taps, None, t);
        }
        ensure!(
            ids.shape_at(0) == 1 && (2..=8).contains(&ids.shape_at(1)),
            "Inkling DSpark verify shape is not supported"
        );
        let cache = cache.ok_or_else(|| anyhow::anyhow!("Inkling DSpark needs a target cache"))?;
        if self.oq && super::layers::is_gpu(t) {
            let mut inputs = Vec::with_capacity(cache.len() / 2);
            let output = super::layers::with_verify_rows(|| {
                self.dspark_prefill(self.embed(ids, t)?, Some(cache), taps, Some(&mut inputs), t)
            })?;
            *verification = Verification::Batched(inputs);
            return Ok(output);
        }
        let mut snapshots = Vec::with_capacity(ids.shape_at(1) as usize + 1);
        snapshots.push(
            cache
                .iter()
                .map(LayerCache::dflash2_transaction_snapshot)
                .collect::<Result<_>>()?,
        );
        let mut features = Vec::new();
        // Each row uses exactly the normal single-token path. Keep a small
        // checkpoint of the four convolution windows per visit for rejection.
        for i in 0..ids.shape_at(1) {
            let embedding = self.embed(&slice(ids, 1, i, i + 1, t)?, t)?;
            let output = self.dspark_prefill(embedding, Some(cache), taps, None, t)?;
            features.push(output.context_hidden);
            snapshots.push(
                cache
                    .iter()
                    .map(LayerCache::dflash2_transaction_snapshot)
                    .collect::<Result<_>>()?,
            );
        }
        // The shared verifier projects the returned hidden tensor itself.
        // Build its hidden rows from the final target tap.
        let context_hidden = cat(&features.iter().collect::<Vec<_>>(), 1, t)?;
        let width = context_hidden.shape_at(-1);
        let hidden = slice(
            &context_hidden,
            2,
            width - self.config.text_config.hidden_size,
            width,
            t,
        )?;
        *verification = Verification::Serial(snapshots);
        Ok(DFlash2TargetOutput {
            hidden,
            context_hidden,
        })
    }

    fn dflash2_restore_target_prefix_on(
        &self,
        cache: &mut [LayerCache],
        before: &[LayerCacheSnapshot],
        accepted: usize,
        target: StreamOrDevice,
    ) -> Result<()> {
        let mut state = self
            .verification
            .lock()
            .map_err(|_| anyhow::anyhow!("Inkling verify state poisoned"))?;
        let verification = std::mem::take(&mut *state);
        let selected = match &verification {
            _ if accepted == 0 => before,
            Verification::Serial(snapshots) => snapshots.get(accepted).ok_or_else(|| {
                anyhow::anyhow!("missing Inkling verify checkpoint at {accepted}")
            })?,
            Verification::Batched(inputs) => {
                ensure!(
                    cache.len() == inputs.len() * 2 && cache.len() == before.len(),
                    "Inkling verify cache shape mismatch"
                );
                let accepted = i32::try_from(accepted)?;
                for (layers, (snapshots, input)) in cache
                    .chunks_exact_mut(2)
                    .zip(before.chunks_exact(2).zip(inputs))
                {
                    let [LayerCache::Full(kv), LayerCache::Linear(conv)] = layers else {
                        anyhow::bail!("Inkling verify cache must alternate KV and convolution");
                    };
                    let [LayerCacheSnapshot::Full(kv_before), LayerCacheSnapshot::Linear(conv_before)] =
                        snapshots
                    else {
                        anyhow::bail!("Inkling verify checkpoint type mismatch");
                    };
                    ensure!(
                        accepted <= input.shape_at(1),
                        "Inkling verify prefix exceeds its block"
                    );
                    kv.restore_offsets(&[kv_before.offsets()[0] + accepted])?;
                    conv.restore(conv_before)?;
                    let keep = conv.conv_state().shape_at(1);
                    let window = cat(&[conv.conv_state(), input], 1, target)?;
                    conv.update_conv(slice(&window, 1, accepted, accepted + keep, target)?);
                    conv.advance(&[accepted])?;
                    conv.discard_speculative_prefix_capture();
                }
                return Ok(());
            }
            Verification::Idle => anyhow::bail!("Inkling verification state missing"),
        };
        ensure!(
            selected.len() == cache.len(),
            "Inkling verify checkpoint size mismatch"
        );
        for (layer, snapshot) in cache.iter_mut().zip(selected) {
            layer.restore(snapshot)?;
            layer.discard_speculative_prefix_capture();
        }
        Ok(())
    }

    fn dflash2_restore_target_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[LayerCacheSnapshot],
        accepted: &[usize],
        t: StreamOrDevice,
    ) -> Result<()> {
        ensure!(
            accepted.len() == 1,
            "Inkling DSpark verifies one request at a time"
        );
        self.dflash2_restore_target_prefix_on(cache, snapshots, accepted[0], t)
    }

    fn dflash2_project_hidden_on(&self, hidden: &Array, t: StreamOrDevice) -> Result<Array> {
        let width = hidden.shape_at(1);
        if width == 1 {
            return self.project_hidden_on(hidden, t);
        }
        if self.oq && super::layers::is_gpu(t) {
            return super::layers::with_verify_rows(|| self.project_hidden_on(hidden, t));
        }
        let rows = (0..width)
            .map(|i| self.project_hidden_on(&slice(hidden, 1, i, i + 1, t)?, t))
            .collect::<Result<Vec<_>>>()?;
        cat(&rows.iter().collect::<Vec<_>>(), 1, t)
    }
}
