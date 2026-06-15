use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::Loader;
use crate::models::gemma4::vision::{MultimodalEmbedder, VisionModel};
use crate::nn::{Embedding, Linear, RmsNorm};
use crate::Result;

use super::attention::{DiffusionGemmaAttention, LayerKv};
use super::config::{DiffusionGemmaConfig, DiffusionGemmaLayerKind, DiffusionGemmaTextConfig};
use super::moe::{DiffusionGemmaExperts, DiffusionGemmaRouter};

pub struct DiffusionGemmaCache {
    layers: Vec<Option<LayerKv>>,
}

pub struct DiffusionGemmaModel {
    pub config: DiffusionGemmaConfig,
    embed_tokens: Embedding,
    layers: Vec<DiffusionGemmaLayer>,
    norm: RmsNorm,
    self_conditioning: SelfConditioning,
    vision: Option<DiffusionGemmaVision>,
    embed_scale: f32,
}

pub(super) struct DiffusionGemmaEncoderInputs<'a> {
    pub pixel_values: Option<&'a [Array]>,
    pub image_grid_thw: Option<&'a [(i32, i32, i32)]>,
    pub mm_token_type_ids: Option<&'a [i32]>,
    pub image_token_id: i32,
}

struct DiffusionGemmaVision {
    tower: VisionModel,
    embedder: MultimodalEmbedder,
    config: crate::models::gemma4::Gemma4VisionConfig,
}

struct DiffusionGemmaLayer {
    self_attn: DiffusionGemmaAttention,
    mlp: GeGluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    router: DiffusionGemmaRouter,
    experts: DiffusionGemmaExperts,
    post_feedforward_layernorm_1: RmsNorm,
    post_feedforward_layernorm_2: RmsNorm,
    pre_feedforward_layernorm_2: RmsNorm,
    layer_scalar: Array,
    encoder_layer_scalar: Array,
}

struct GeGluMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

struct SelfConditioning {
    pre_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    eps: f32,
}

impl DiffusionGemmaCache {
    fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![None; n_layers],
        }
    }

    pub fn len(&self) -> i32 {
        self.layers
            .first()
            .and_then(|kv| kv.as_ref())
            .map(|kv| kv.keys.shape_at(2))
            .unwrap_or(0)
    }
}

impl DiffusionGemmaModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let config = DiffusionGemmaConfig::from_loader(loader)?;
        let text = &config.text_config;
        let embed_scale = (text.hidden_size as f32).sqrt();
        let embed_tokens = Embedding::from_loader(loader, "model.decoder.embed_tokens")
            .context("DiffusionGemmaModel: loading embed_tokens")?;
        let mut layers = Vec::with_capacity(text.num_hidden_layers as usize);
        for i in 0..text.num_hidden_layers as usize {
            layers.push(DiffusionGemmaLayer::from_loader(loader, text, i)?);
        }
        let norm = RmsNorm::from_loader(loader, "model.decoder.norm", text.rms_norm_eps)
            .context("DiffusionGemmaModel: loading decoder norm")?;
        let self_conditioning =
            SelfConditioning::from_loader(loader, "model.decoder.self_conditioning", text)?;
        let vision = if let Some(vision_config) = config.vision_config.clone() {
            if loader.contains("model.encoder.vision_tower.patch_embedder.input_proj.weight") {
                Some(DiffusionGemmaVision {
                    tower: VisionModel::from_loader_with_prefix(
                        loader,
                        vision_config.clone(),
                        "model.encoder.vision_tower",
                    )
                    .context("DiffusionGemmaModel: loading encoder vision_tower")?,
                    embedder: MultimodalEmbedder::from_loader(
                        loader,
                        "model.encoder.embed_vision",
                        vision_config.rms_norm_eps,
                    )
                    .context("DiffusionGemmaModel: loading encoder embed_vision")?,
                    config: vision_config,
                })
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            self_conditioning,
            vision,
            embed_scale,
        })
    }

    pub fn make_cache(&self) -> DiffusionGemmaCache {
        DiffusionGemmaCache::new(self.layers.len())
    }

    pub fn encode_tokens_on(
        &self,
        input_ids: &Array,
        cache: &mut DiffusionGemmaCache,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        self.encode_inputs_on(
            input_ids,
            DiffusionGemmaEncoderInputs {
                pixel_values: None,
                image_grid_thw: None,
                mm_token_type_ids: None,
                image_token_id: self.image_token_id(),
            },
            cache,
            target,
        )
    }

    pub(super) fn encode_inputs_on(
        &self,
        input_ids: &Array,
        inputs: DiffusionGemmaEncoderInputs<'_>,
        cache: &mut DiffusionGemmaCache,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        let shape = input_ids.shape();
        let dims = shape.as_slice();
        if dims.len() != 2 || dims[0] != 1 {
            return Err(anyhow!(
                "DiffusionGemma encoder supports input_ids [1,S], got {:?}",
                dims
            ));
        }
        let seq = dims[1];
        if seq <= 0 {
            return Ok(());
        }
        if let Some(types) = inputs.mm_token_type_ids {
            if types.len() != seq as usize {
                return Err(anyhow!(
                    "DiffusionGemma encoder mm_token_type_ids.len()={} != seq={seq}",
                    types.len()
                ));
            }
        }
        let offset = cache.len();
        let image_token_count = self.count_image_tokens(input_ids, inputs.image_token_id)?;
        let embed_input_ids = if image_token_count > 0 {
            self.image_token_ids_to_pad(input_ids, inputs.image_token_id)?
        } else {
            input_ids.clone()
        };
        let mut h = self.embed_tokens.forward_on(&embed_input_ids, target)?;
        h = &h * self.embed_scale;
        match (inputs.pixel_values, inputs.image_grid_thw) {
            (Some(pixels), Some(grids)) if !pixels.is_empty() => {
                let vision_embeds = self
                    .compute_vision_embeds(pixels, grids, target)?
                    .astype_on(h.dtype(), target)?;
                h = crate::models::qwen3_5::cross_modal::replace_image_tokens(
                    &h,
                    input_ids,
                    &vision_embeds,
                    inputs.image_token_id,
                )?;
            }
            (Some(_), None) => {
                return Err(anyhow!(
                    "DiffusionGemma encoder received pixel_values without image_grid_thw"
                ));
            }
            (None, Some(grids)) if !grids.is_empty() => {
                return Err(anyhow!(
                    "DiffusionGemma encoder received image_grid_thw without pixel_values"
                ));
            }
            _ if image_token_count > 0 => {
                return Err(anyhow!(
                    "DiffusionGemma encoder prompt has {image_token_count} image tokens but no pixel_values"
                ));
            }
            _ => {}
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = build_encoder_mask(
                offset,
                seq,
                layer.self_attn.layer_kind(),
                self.config.text_config.sliding_window,
                if self
                    .config
                    .text_config
                    .use_bidirectional_attention
                    .as_deref()
                    == Some("vision")
                {
                    inputs.mm_token_type_ids
                } else {
                    None
                },
                Dtype::Bfloat16,
                target,
            )?;
            let prior = cache.layers[i].as_ref();
            let (next_h, next_kv) = layer.forward_encoder_on(&h, &mask, offset, prior, target)?;
            cache.layers[i] = Some(next_kv);
            h = next_h;
        }
        let h = self.norm.forward_on(&h, target)?;
        mlx::transforms::eval(&[&h]).context("DiffusionGemmaModel: eval encoder hidden")?;
        Ok(())
    }

    pub fn decode_logits_on(
        &self,
        canvas_ids: &Array,
        cache: &DiffusionGemmaCache,
        self_conditioning_embeddings: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let encoder_len = cache.len();
        let mut h = self.embed_canvas_on(canvas_ids, self_conditioning_embeddings, target)?;
        for (layer, kv) in self.layers.iter().zip(cache.layers.iter()) {
            h = layer.forward_decoder_on(&h, kv.as_ref(), encoder_len, target)?;
        }
        h = self.norm.forward_on(&h, target)?;
        let mut logits = self.embed_tokens.as_output_on(&h, target)?;
        if let Some(c) = self.config.text_config.final_logit_softcapping {
            logits = softcap_on(&logits, c, target)?;
        }
        Ok(logits)
    }

    pub fn soft_embedding_weight_on(&self, target: impl Into<StreamOrDevice>) -> Result<Array> {
        self.embed_tokens.dense_weight_on(target)
    }

    pub fn embed_scale(&self) -> f32 {
        self.embed_scale
    }

    fn image_token_id(&self) -> i32 {
        self.config.image_token_id.unwrap_or(258_880)
    }

    fn count_image_tokens(&self, input_ids: &Array, image_token_id: i32) -> Result<usize> {
        let ids_i32 = mlx::ops::cast::astype(input_ids, Dtype::Int32)?;
        let ids = ids_i32.to_vec::<i32>()?;
        Ok(ids.iter().filter(|&&id| id == image_token_id).count())
    }

    fn image_token_ids_to_pad(&self, input_ids: &Array, image_token_id: i32) -> Result<Array> {
        let shape = input_ids.shape();
        let dims = shape.as_slice();
        if dims.len() != 2 {
            return Err(anyhow!(
                "DiffusionGemma image_token_ids_to_pad expects [B,S], got {dims:?}"
            ));
        }
        let ids_i32 = mlx::ops::cast::astype(input_ids, Dtype::Int32)?;
        let mut ids = ids_i32.to_vec::<i32>()?;
        for id in &mut ids {
            if *id == image_token_id {
                *id = self.config.text_config.pad_token_id;
            }
        }
        Ok((ids.as_slice(), dims).try_into()?)
    }

    fn compute_vision_embeds(
        &self,
        pixel_values: &[Array],
        grid_thw: &[(i32, i32, i32)],
        target: StreamOrDevice,
    ) -> Result<Array> {
        if pixel_values.is_empty() {
            return Err(anyhow!(
                "DiffusionGemmaModel::compute_vision_embeds: pixel_values cannot be empty"
            ));
        }
        if pixel_values.len() != grid_thw.len() {
            return Err(anyhow!(
                "DiffusionGemmaModel::compute_vision_embeds: pixel_values.len()={} must equal grid_thw.len()={}",
                pixel_values.len(),
                grid_thw.len()
            ));
        }
        let vision = self.vision.as_ref().ok_or_else(|| {
            anyhow!("DiffusionGemmaModel has no encoder vision_tower; use Loader::open_multimodal")
        })?;
        let pool = vision.config.pooling_kernel_size;
        let mut per_image = Vec::with_capacity(pixel_values.len());
        for (idx, pixels) in pixel_values.iter().enumerate() {
            let shape = pixels.shape();
            let dims = shape.as_slice();
            if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
                return Err(anyhow!(
                    "DiffusionGemmaModel::compute_vision_embeds image {idx} expected [1,3,H,W], got {dims:?}"
                ));
            }
            let features = vision.tower.forward_on(pixels, target)?;
            let projected = vision.embedder.forward_on(&features, target)?;
            let shape = projected.shape();
            let dims = shape.as_slice();
            if dims.len() != 3 || dims[0] != 1 {
                return Err(anyhow!(
                    "DiffusionGemmaModel::compute_vision_embeds image {idx} expected projected [1,N,H], got {dims:?}"
                ));
            }
            let (n, hidden) = (dims[1], dims[2]);
            let (t, gh, gw) = grid_thw[idx];
            if gh % pool != 0 || gw % pool != 0 {
                return Err(anyhow!(
                    "DiffusionGemmaModel::compute_vision_embeds image {idx}: grid {gh}x{gw} is not divisible by pooling_kernel_size={pool}"
                ));
            }
            let expected = t * (gh / pool) * (gw / pool);
            if expected != n {
                return Err(anyhow!(
                    "DiffusionGemmaModel::compute_vision_embeds image {idx}: grid_thw implies {expected} soft tokens but vision produced {n}; max per image {}",
                    vision.config.default_output_length
                ));
            }
            let sliced = mlx::ops::indexing::slice_strided_on(
                &projected,
                &[0_i32, 0, 0][..],
                &[1_i32, n, hidden][..],
                &[1_i32, 1, 1][..],
                target,
            )?;
            per_image.push(sliced.reshape_on((n, hidden), target)?);
        }
        if per_image.len() == 1 {
            Ok(per_image.pop().expect("len checked"))
        } else {
            let refs: Vec<&Array> = per_image.iter().collect();
            Ok(mlx::ops::shape::concatenate_on(&refs, 0, target)?)
        }
    }

    fn embed_canvas_on(
        &self,
        canvas_ids: &Array,
        self_conditioning_embeddings: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let mut inputs = self.embed_tokens.forward_on(canvas_ids, target)?;
        inputs = &inputs * self.embed_scale;
        let signal = match self_conditioning_embeddings {
            Some(sc) => sc.astype_on(inputs.dtype(), target)?,
            None => mlx::ops::constructors::zeros_like_on(&inputs, target)?,
        };
        self.self_conditioning.forward_on(&inputs, &signal, target)
    }
}

impl DiffusionGemmaLayer {
    fn from_loader(
        loader: &Loader,
        cfg: &DiffusionGemmaTextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("model.decoder.layers.{layer_idx}");
        Ok(Self {
            self_attn: DiffusionGemmaAttention::from_loader(
                loader,
                &format!("{prefix}.self_attn"),
                cfg,
                layer_idx,
            )?,
            mlp: GeGluMlp::from_loader(loader, &format!("{prefix}.mlp"))?,
            input_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.input_layernorm"),
                cfg.rms_norm_eps,
            )?,
            post_attention_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
            pre_feedforward_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.pre_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
            post_feedforward_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
            router: DiffusionGemmaRouter::from_loader(
                loader,
                &format!("{prefix}.router"),
                cfg.hidden_size,
                cfg.num_experts,
                cfg.top_k_experts,
                cfg.rms_norm_eps,
            )?,
            experts: DiffusionGemmaExperts::from_loader(loader, &format!("{prefix}.experts"))?,
            post_feedforward_layernorm_1: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_feedforward_layernorm_1"),
                cfg.rms_norm_eps,
            )?,
            post_feedforward_layernorm_2: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_feedforward_layernorm_2"),
                cfg.rms_norm_eps,
            )?,
            pre_feedforward_layernorm_2: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.pre_feedforward_layernorm_2"),
                cfg.rms_norm_eps,
            )?,
            layer_scalar: loader.tensor(&format!("{prefix}.layer_scalar"))?.clone(),
            encoder_layer_scalar: loader
                .tensor(&format!(
                    "model.encoder.language_model.layers.{layer_idx}.layer_scalar"
                ))?
                .clone(),
        })
    }

    fn forward_encoder_on(
        &self,
        x: &Array,
        mask: &Array,
        offset: i32,
        prior: Option<&LayerKv>,
        target: StreamOrDevice,
    ) -> Result<(Array, LayerKv)> {
        let residual = x.clone();
        let h = self.input_layernorm.forward_on(x, target)?;
        let (h, kv) = self
            .self_attn
            .forward_encoder_on(&h, Some(mask), offset, prior, target)?;
        let h = self.post_attention_layernorm.forward_on(&h, target)?;
        let h = &residual + &h;
        let h = self.feed_forward_on(&h, target)?;
        let h = &h * &self.encoder_layer_scalar;
        Ok((h, kv))
    }

    fn forward_decoder_on(
        &self,
        x: &Array,
        encoder_kv: Option<&LayerKv>,
        encoder_len: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward_on(x, target)?;
        let h = self
            .self_attn
            .forward_decoder_on(&h, None, encoder_kv, encoder_len, target)?;
        let h = self.post_attention_layernorm.forward_on(&h, target)?;
        let h = &residual + &h;
        let h = self.feed_forward_on(&h, target)?;
        Ok(&h * &self.layer_scalar)
    }

    fn feed_forward_on(&self, h: &Array, target: StreamOrDevice) -> Result<Array> {
        let residual = h.clone();
        let h1 = self.pre_feedforward_layernorm.forward_on(h, target)?;
        let h1 = self.mlp.forward_on(&h1, target)?;
        let h1 = self.post_feedforward_layernorm_1.forward_on(&h1, target)?;

        let shape = residual.shape();
        let dims = shape.as_slice();
        let (b, s, hidden) = (dims[0], dims[1], dims[2]);
        let flat = residual.reshape_on([b * s, hidden], target)?;
        let (weights, inds) = self.router.route_on(&flat, target)?;
        let h2 = self.pre_feedforward_layernorm_2.forward_on(&flat, target)?;
        let h2 = self.experts.forward_on(&h2, &inds, &weights, target)?;
        let h2 = h2.reshape_on((b, s, hidden), target)?;
        let h2 = self.post_feedforward_layernorm_2.forward_on(&h2, target)?;
        let h = &h1 + &h2;
        let h = self.post_feedforward_layernorm.forward_on(&h, target)?;
        Ok(&residual + &h)
    }
}

impl GeGluMlp {
    fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?,
            up: Linear::from_loader(loader, &format!("{prefix}.up_proj"))?,
            down: Linear::from_loader(loader, &format!("{prefix}.down_proj"))?,
        })
    }

    fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        let gate = self.gate.forward_on(x, target)?;
        let up = self.up.forward_on(x, target)?;
        let gate = crate::nn::gelu_tanh(&gate, target)?;
        let h = &gate * &up;
        self.down.forward_on(&h, target)
    }
}

impl SelfConditioning {
    fn from_loader(loader: &Loader, prefix: &str, cfg: &DiffusionGemmaTextConfig) -> Result<Self> {
        Ok(Self {
            pre_norm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.pre_norm"),
                cfg.rms_norm_eps,
            )?,
            gate: Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?,
            up: Linear::from_loader(loader, &format!("{prefix}.up_proj"))?,
            down: Linear::from_loader(loader, &format!("{prefix}.down_proj"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward_on(&self, inputs: &Array, signal: &Array, target: StreamOrDevice) -> Result<Array> {
        let normed = self.pre_norm.forward_on(signal, target)?;
        let gate = self.gate.forward_on(&normed, target)?;
        let up = self.up.forward_on(&normed, target)?;
        let gate = crate::nn::gelu_tanh(&gate, target)?;
        let signal = self.down.forward_on(&(&gate * &up), target)?;
        let h = inputs + &signal;
        Ok(mlx::fast::rms_norm_on(&h, None, self.eps, target)?)
    }
}

fn softcap_on(x: &Array, cap: f32, target: StreamOrDevice) -> Result<Array> {
    let cap_arr: Array = (&[cap][..], ()).try_into()?;
    let y = x.try_div_on(&cap_arr, target)?;
    let y = y.tanh_on(target)?;
    Ok(&y * cap)
}

fn build_encoder_mask(
    offset: i32,
    seq: i32,
    layer_kind: DiffusionGemmaLayerKind,
    sliding_window: i32,
    mm_token_type_ids: Option<&[i32]>,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<Array> {
    if offset < 0 || seq <= 0 {
        return Err(anyhow!(
            "DiffusionGemma encoder mask: offset={offset}, seq={seq}"
        ));
    }
    let key_len = offset + seq;
    let neg_inf = f32::NEG_INFINITY;
    let mut flat = vec![neg_inf; (seq * key_len) as usize];
    for q in 0..seq {
        let q_abs = offset + q;
        let min_k = match layer_kind {
            DiffusionGemmaLayerKind::Full => 0,
            DiffusionGemmaLayerKind::Sliding => (q_abs - sliding_window + 1).max(0),
        };
        for k in min_k..=q_abs {
            flat[(q * key_len + k) as usize] = 0.0;
        }
    }
    if offset == 0 {
        if let Some(types) = mm_token_type_ids {
            if types.len() != seq as usize {
                return Err(anyhow!(
                    "DiffusionGemma encoder mask: mm_token_type_ids.len()={} != seq={seq}",
                    types.len()
                ));
            }
            let key_len = key_len as usize;
            for q in 0..seq as usize {
                if !is_visual_token_type(types[q]) {
                    continue;
                }
                let mut start = q;
                while start > 0 && is_visual_token_type(types[start - 1]) {
                    start -= 1;
                }
                let mut end = q + 1;
                while end < types.len() && is_visual_token_type(types[end]) {
                    end += 1;
                }
                for k in start..end {
                    flat[q * key_len + k] = 0.0;
                }
            }
        }
    }
    let arr: Array = (&flat[..], &[1_i32, 1_i32, seq, key_len][..]).try_into()?;
    Ok(mlx::ops::cast::astype_on(&arr, dtype, target)?)
}

fn is_visual_token_type(token_type: i32) -> bool {
    token_type == 1 || token_type == 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_mask_allows_bidirectional_attention_inside_visual_blocks() {
        let mask = build_encoder_mask(
            0,
            4,
            DiffusionGemmaLayerKind::Sliding,
            1,
            Some(&[0, 1, 1, 0]),
            Dtype::Float32,
            StreamOrDevice::default(),
        )
        .unwrap();
        let values: Vec<f32> = mask.reshape(&[16_i32][..]).unwrap().to_vec().unwrap();

        assert_eq!(
            values[1 * 4 + 2],
            0.0,
            "image token attends future image token"
        );
        assert_eq!(
            values[2 * 4 + 1],
            0.0,
            "image token attends previous image token"
        );
        assert!(
            values[1 * 4 + 3].is_infinite() && values[1 * 4 + 3].is_sign_negative(),
            "image block must not see future text"
        );
    }
}
