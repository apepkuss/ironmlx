//! Qwen3.6 MoE model facade.
//!
//! This wrapper enforces Qwen3.6 MoE checkpoint structure before delegating
//! numeric execution to [`Qwen35MoeModel`]. Runtime entry points such as
//! `generate` and `serve` dispatch by execution architecture and can use
//! `Qwen35MoeModel` directly for these checkpoints.

use anyhow::Context;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::MtpCache;
use crate::core::memory_budget::ModelMeta;
use crate::core::{Loader, Model};
use crate::models::qwen3_5_moe::{Qwen35MoeModel, Qwen35MoeMtp};
use crate::models::vision::VisionTower;
use crate::nn::{LayerCache, MtpStepOutput};
use crate::Result;

use super::config::Qwen36MoeConfig;

pub struct Qwen36MoeModel {
    cfg: Qwen36MoeConfig,
    inner: Qwen35MoeModel,
}

impl Qwen36MoeModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let cfg =
            Qwen36MoeConfig::from_loader(loader).context("parsing Qwen36MoeConfig from loader")?;
        Self::from_loader_with_config(loader, cfg)
    }

    pub fn from_loader_with_config(loader: &Loader, cfg: Qwen36MoeConfig) -> Result<Self> {
        let inner =
            Qwen35MoeModel::from_loader_with_config(loader, cfg.as_qwen35_moe_config().clone())
                .context("loading shared Qwen3.6 MoE execution kernel")?;
        Ok(Self { cfg, inner })
    }

    pub fn config(&self) -> &Qwen36MoeConfig {
        &self.cfg
    }

    pub fn text(&self) -> &crate::models::qwen3_6_moe::Qwen36MoeTextModel {
        self.inner.text()
    }

    pub fn vision(&self) -> Option<&VisionTower> {
        self.inner.vision()
    }

    pub fn load_mtp_head(&self, loader: &Loader) -> Result<Qwen35MoeMtp> {
        self.inner.load_mtp_head(loader)
    }

    pub fn project_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.inner.project_hidden_on(hidden, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mtp_forward_hidden_on(
        &self,
        mtp: &Qwen35MoeMtp,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.inner.mtp_forward_hidden_on(
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mtp_forward_on(
        &self,
        mtp: &Qwen35MoeMtp,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        self.inner.mtp_forward_on(
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    pub fn compute_vision_embeds(
        &self,
        pixel_values: &[Array],
        grid_thw: &[(i32, i32, i32)],
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        if pixel_values.is_empty() {
            return Err(anyhow::anyhow!(
                "Qwen36MoeModel::compute_vision_embeds: pixel_values cannot be empty"
            ));
        }
        let pixels = if pixel_values.len() == 1 {
            pixel_values[0].clone()
        } else {
            let refs: Vec<&Array> = pixel_values.iter().collect();
            mlx::ops::shape::concatenate(&refs, 0)?
        };
        self.inner.compute_vision_embeds(&pixels, grid_thw, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl_chunk(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.inner.forward_vl_chunk(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_vl_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        vision_embeds_slice: Option<&Array>,
        image_token_id: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.inner.forward_vl_hidden(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn batched_prefill_vl(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&[Array]>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let mut row_pixel_values = Vec::with_capacity(per_row_pixel_values.len());
        for row in per_row_pixel_values {
            let Some(values) = row else {
                row_pixel_values.push(None);
                continue;
            };
            if values.is_empty() {
                return Err(anyhow::anyhow!(
                    "Qwen36MoeModel::batched_prefill_vl: row pixel_values cannot be empty"
                ));
            }
            if values.len() == 1 {
                row_pixel_values.push(Some(values[0].clone()));
            } else {
                let refs: Vec<&Array> = values.iter().collect();
                row_pixel_values.push(Some(mlx::ops::shape::concatenate(&refs, 0)?));
            }
        }
        let row_pixel_refs: Vec<Option<&Array>> =
            row_pixel_values.iter().map(|opt| opt.as_ref()).collect();
        self.inner.batched_prefill_vl(
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            &row_pixel_refs,
            per_row_grid_thw,
            image_token_id,
            cache,
            target,
        )
    }

    pub fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        self.inner.make_cache(batch, cap, dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.inner.forward_on(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

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
        self.inner.batched_prefill(
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            cache,
            target,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        self.inner.forward_text_hidden(
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            target,
        )
    }

    pub fn model_meta(&self) -> ModelMeta {
        self.inner.model_meta()
    }

    #[cfg(test)]
    pub(crate) fn from_cfg_for_test(cfg: Qwen36MoeConfig) -> Self {
        let inner = Qwen35MoeModel::from_cfg_for_test(cfg.as_qwen35_moe_config().clone());
        Self { cfg, inner }
    }
}

impl Model for Qwen36MoeModel {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>> {
        Qwen36MoeModel::make_cache(self, batch, cap, dtype)
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
        Qwen36MoeModel::forward_on(
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
        Qwen36MoeModel::batched_prefill(
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
        Qwen36MoeModel::forward_text_hidden(
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
        Qwen36MoeModel::model_meta(self)
    }

    fn num_hidden_layers(&self) -> usize {
        self.cfg.num_hidden_layers as usize
    }
}

impl crate::core::scheduler::DenseVlMethods for Qwen36MoeModel {
    fn batched_prefill_vl(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        attention_mask: &mlx::Array,
        linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        per_row_pixel_values: &[Option<&[mlx::Array]>],
        per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        image_token_id: i32,
        cache: Option<&mut [crate::nn::LayerCache]>,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen36MoeModel::batched_prefill_vl(
            self,
            input_ids,
            position_ids,
            attention_mask,
            linear_attention_mask,
            per_row_lens,
            per_row_pixel_values,
            per_row_grid_thw,
            image_token_id,
            cache,
            target,
        )
    }

    fn compute_vision_embeds(
        &self,
        pixel_values: &[mlx::Array],
        grid_thw: &[(i32, i32, i32)],
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen36MoeModel::compute_vision_embeds(self, pixel_values, grid_thw, target)
    }

    fn forward_vl_chunk(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen36MoeModel::forward_vl_chunk(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )
    }

    fn forward_vl_hidden(
        &self,
        input_ids: &mlx::Array,
        position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        vision_embeds_slice: Option<&mlx::Array>,
        image_token_id: i32,
        target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Qwen36MoeModel::forward_vl_hidden(
            self,
            input_ids,
            position_ids,
            per_row_lens,
            decode_mask,
            cache,
            vision_embeds_slice,
            image_token_id,
            target,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scheduler::DenseVlMethods;

    fn make_cfg() -> Qwen36MoeConfig {
        let cfg = crate::models::qwen3_5_moe::Qwen35MoeConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 0,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(8),
            vocab_size: 1024,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: false,
            full_attention_interval: 2,
            mtp_num_hidden_layers: 1,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_parameters: crate::models::qwen3_5_moe::RopeParams {
                partial_rotary_factor: 1.0,
                rope_theta: 1e7,
                mrope_section: vec![2, 1, 1],
            },
            num_experts: 8,
            num_experts_per_tok: 2,
            norm_topk_prob: false,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            mlp_only_layers: vec![],
            vision_config: None,
            max_position_embeddings: 32768,
        };
        Qwen36MoeConfig::from_inner_for_test(cfg)
    }

    fn assert_model_surface<M: Model + DenseVlMethods>(_: &M) {}

    #[test]
    fn qwen36_moe_model_implements_core_and_vl_traits() {
        let model = Qwen36MoeModel::from_cfg_for_test(make_cfg());
        assert_model_surface(&model);
        assert_eq!(model.num_hidden_layers(), 0);
        assert_eq!(model.config().hidden_size, 32);
    }

    #[test]
    fn text_only_vl_chunk_delegates_to_core_forward() {
        use crate::core::generate::{build_position_ids, IMAGE_TOKEN_ID};

        let model = Qwen36MoeModel::from_cfg_for_test(make_cfg());
        let input_ids: Array = (&[1_i32, 2, 3][..], &[1_i32, 3][..])
            .try_into()
            .expect("input_ids");
        let position_ids = build_position_ids(0, 3).expect("position_ids");

        let logits_text = model
            .forward_on(&input_ids, &position_ids, None, None, None, ())
            .expect("forward_on");
        let logits_vl = model
            .forward_vl_chunk(
                &input_ids,
                &position_ids,
                None,
                None,
                None,
                None,
                IMAGE_TOKEN_ID,
                (),
            )
            .expect("forward_vl_chunk text-only");

        let a: Vec<f32> = mlx::ops::astype(&logits_text, Dtype::Float32)
            .expect("astype text")
            .to_vec()
            .expect("to_vec text");
        let b: Vec<f32> = mlx::ops::astype(&logits_vl, Dtype::Float32)
            .expect("astype vl")
            .to_vec()
            .expect("to_vec vl");
        assert_eq!(a, b);
    }

    fn qwen36_model_dir() -> Option<std::path::PathBuf> {
        let path = match std::env::var("QWEN36_MOE_MODEL") {
            Ok(path) => std::path::PathBuf::from(path),
            Err(_) => {
                eprintln!("skip: set QWEN36_MOE_MODEL to a local Qwen3.6 MoE checkpoint");
                return None;
            }
        };
        if !path.exists() {
            eprintln!("skip: {} not found", path.display());
            return None;
        }
        Some(path)
    }

    #[test]
    #[ignore = "loads a full local Qwen3.6 MoE checkpoint"]
    fn loads_qwen36_moe_real_checkpoint_with_vision() {
        let Some(dir) = qwen36_model_dir() else {
            return;
        };
        let loader = crate::core::Loader::open_multimodal(&dir).expect("open_multimodal");
        assert!(crate::models::is_qwen36_moe_config(
            loader.config_raw_value()
        ));
        let model = Qwen36MoeModel::from_loader(&loader).expect("load model");
        assert!(model.vision().is_some(), "vision tower should be loaded");
        assert_eq!(model.config().num_experts, 256);
        assert_eq!(model.config().num_experts_per_tok, 8);
    }

    fn image_placeholder_string(token_count: usize) -> String {
        let mut out = String::new();
        out.push_str("<|vision_start|>");
        for _ in 0..token_count {
            out.push_str("<|image_pad|>");
        }
        out.push_str("<|vision_end|>");
        out
    }

    fn prepare_fixture_images(
        paths: &[&str],
        spatial_merge_size: i32,
    ) -> (Array, Vec<(i32, i32, i32)>, String) {
        let mut pixel_values = Vec::with_capacity(paths.len());
        let mut grids = Vec::with_capacity(paths.len());
        let mut prompt_prefix = String::new();

        for path in paths {
            let bytes = std::fs::read(path).expect("fixture image");
            let (pv, gh, gw) =
                crate::models::qwen3_5::image_processor::preprocess(&bytes).expect("preprocess");
            let token_count = ((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize;
            prompt_prefix.push_str(&image_placeholder_string(token_count));
            pixel_values.push(pv);
            grids.push((1, gh, gw));
        }

        let refs: Vec<&Array> = pixel_values.iter().collect();
        let concat = mlx::ops::shape::concatenate(&refs, 0).expect("concatenate pixel_values");
        mlx::transforms::eval(&[&concat]).expect("eval pixel_values");
        (concat, grids, prompt_prefix)
    }

    fn generation_request(
        tokenizer: &crate::core::Tokenizer,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        pixel_values: Option<Array>,
        image_grid_thw: Option<Vec<(i32, i32, i32)>>,
        image_spatial_merge_size: i32,
    ) -> crate::core::generate::GenerateRequest {
        crate::core::generate::GenerateRequest {
            prompt_ids,
            max_new_tokens,
            sampler: crate::core::sampler::Sampler::greedy(),
            stop_token_ids: tokenizer.eos_token_ids().to_vec(),
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: pixel_values.map(|pv| vec![pv]),
            image_grid_thw,
            image_spatial_merge_size,
            image_token_id: tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID),
            #[cfg(feature = "p5h-profile")]
            p5h_trace: None,
            #[cfg(feature = "p5h-profile")]
            p5h_root_span: None,
        }
    }

    #[test]
    #[ignore = "runs one-token generation on a full local Qwen3.6 MoE checkpoint"]
    fn qwen36_moe_text_generation_smoke_real_checkpoint() {
        let Some(dir) = qwen36_model_dir() else {
            return;
        };
        let loader = crate::core::Loader::open(&dir).expect("open");
        let tokenizer = crate::core::Tokenizer::from_loader(&loader).expect("tokenizer");
        let model = Qwen36MoeModel::from_loader(&loader).expect("model");
        let prompt = tokenizer
            .apply_chat_template(
                &[crate::core::Message {
                    role: "user".to_owned(),
                    content: "Say hi.".to_owned(),
                }],
                true,
                Some(&serde_json::json!({"enable_thinking": false})),
            )
            .expect("chat template");
        let prompt_ids = tokenizer
            .encode(&prompt, false)
            .expect("tokenize text prompt");
        let request = generation_request(&tokenizer, prompt_ids, 1, None, None, 2);
        let mut stream =
            crate::core::generate::GenerationStream::new_text_only(&model, &tokenizer, request)
                .expect("stream");
        assert!(stream.next_token().expect("next token").is_some());
    }

    #[test]
    #[ignore = "checks full local Qwen3.6 MoE text forward paths against a reference first token"]
    fn qwen36_moe_text_forward_paths_first_token_real_checkpoint() {
        let Some(dir) = qwen36_model_dir() else {
            return;
        };
        let loader = crate::core::Loader::open(&dir).expect("open");
        let tokenizer = crate::core::Tokenizer::from_loader(&loader).expect("tokenizer");
        let model = Qwen36MoeModel::from_loader(&loader).expect("model");
        let prompt = tokenizer
            .apply_chat_template(
                &[crate::core::Message {
                    role: "user".to_owned(),
                    content:
                        "Write one concise sentence explaining why reproducible benchmarks matter."
                            .to_owned(),
                }],
                true,
                Some(&serde_json::json!({"enable_thinking": false})),
            )
            .expect("chat template");
        let prompt_ids = tokenizer
            .encode(&prompt, false)
            .expect("tokenize text prompt");
        let len = prompt_ids.len() as i32;
        let input_ids: Array = (prompt_ids.as_slice(), &[1_i32, len][..])
            .try_into()
            .expect("input ids");
        let position_ids = crate::core::generate::build_position_ids(0, len).expect("position ids");
        let sample = |logits: &Array| {
            let vocab = logits.shape().as_slice()[2];
            let flat = logits.reshape((vocab,)).expect("reshape logits");
            let mut prng = mlx::random::key(0).expect("prng");
            crate::core::sampler::Sampler::greedy()
                .sample(&flat, &prompt_ids, &mut prng)
                .expect("sample")
        };
        // Verified against the pinned MLX runtime documented for Qwen3.6
        // quality checks; the token decodes as "Re" for a direct answer.
        let expected_first_token = 674_u32;

        let logits_no_cache = model
            .forward_on(&input_ids, &position_ids, None, None, None, ())
            .expect("forward no cache");
        assert_eq!(sample(&logits_no_cache), expected_first_token);

        let mut full_cache = model
            .make_cache(1, len + 4, Dtype::Bfloat16)
            .expect("full cache");
        let logits_full_cache = model
            .forward_on(
                &input_ids,
                &position_ids,
                None,
                None,
                Some(&mut full_cache),
                (),
            )
            .expect("forward full cache");
        assert_eq!(sample(&logits_full_cache), expected_first_token);

        let mut split_cache = model
            .make_cache(1, len + 4, Dtype::Bfloat16)
            .expect("split cache");
        let prefix_len = len - 1;
        let prefix_ids = mlx::ops::indexing::slice_strided(
            &input_ids,
            &[0_i32, 0][..],
            &[1_i32, prefix_len][..],
            &[1_i32, 1][..],
        )
        .expect("prefix ids");
        let prefix_pos =
            crate::core::generate::build_position_ids(0, prefix_len).expect("prefix pos");
        let prefix_hidden = model
            .forward_text_hidden(
                &prefix_ids,
                &prefix_pos,
                None,
                None,
                Some(&mut split_cache),
                (),
            )
            .expect("prefix hidden");
        mlx::transforms::eval(&[&prefix_hidden]).expect("eval prefix");
        let last_ids = mlx::ops::indexing::slice_strided(
            &input_ids,
            &[0_i32, prefix_len][..],
            &[1_i32, len][..],
            &[1_i32, 1][..],
        )
        .expect("last ids");
        let last_pos = crate::core::generate::build_position_ids(prefix_len, 1).expect("last pos");
        let logits_split_cache = model
            .forward_on(&last_ids, &last_pos, None, None, Some(&mut split_cache), ())
            .expect("forward split cache");
        assert_eq!(sample(&logits_split_cache), expected_first_token);

        let mut batch_cache = model
            .make_cache(1, len + 4, Dtype::Bfloat16)
            .expect("batch cache");
        let batch_pos =
            crate::core::generate::build_position_ids_batched(&[len], len).expect("batch pos");
        let attention_mask =
            crate::core::generate::build_batch_attention_mask(&[len], len, Dtype::Bfloat16)
                .expect("attention mask");
        let linear_mask =
            crate::core::generate::build_batch_linear_mask(&[len], len).expect("linear mask");
        let logits_batched = model
            .batched_prefill(
                &input_ids,
                &batch_pos,
                &attention_mask,
                &linear_mask,
                &[len],
                Some(&mut batch_cache),
                (),
            )
            .expect("batched prefill");
        assert_eq!(sample(&logits_batched), expected_first_token);
    }

    #[test]
    #[ignore = "runs one-token single-image generation on a full local Qwen3.6 MoE checkpoint"]
    fn qwen36_moe_single_image_generation_smoke_real_checkpoint() {
        let Some(dir) = qwen36_model_dir() else {
            return;
        };
        let loader = crate::core::Loader::open_multimodal(&dir).expect("open_multimodal");
        let tokenizer = crate::core::Tokenizer::from_loader(&loader).expect("tokenizer");
        let model = Qwen36MoeModel::from_loader(&loader).expect("model");
        let merge = model.model_meta().spatial_merge_size;
        let (pixel_values, grids, mut content) = prepare_fixture_images(
            &["tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg"],
            merge,
        );
        content.push_str("Describe this image briefly.");
        let prompt = tokenizer
            .apply_chat_template(
                &[crate::core::Message {
                    role: "user".to_owned(),
                    content,
                }],
                true,
                None,
            )
            .expect("chat template");
        let prompt_ids = tokenizer
            .encode(&prompt, false)
            .expect("tokenize single-image prompt");
        let request = generation_request(
            &tokenizer,
            prompt_ids,
            1,
            Some(pixel_values),
            Some(grids),
            merge,
        );
        let mut stream = crate::core::generate::GenerationStream::new(&model, &tokenizer, request)
            .expect("stream");
        assert!(stream.next_token().expect("next token").is_some());
    }

    #[test]
    #[ignore = "checks single-image first token on a full local Qwen3.6 MoE checkpoint"]
    fn qwen36_moe_single_image_forward_first_token_real_checkpoint() {
        let Some(dir) = qwen36_model_dir() else {
            return;
        };
        let loader = crate::core::Loader::open_multimodal(&dir).expect("open_multimodal");
        let tokenizer = crate::core::Tokenizer::from_loader(&loader).expect("tokenizer");
        let model = Qwen36MoeModel::from_loader(&loader).expect("model");
        let merge = model.model_meta().spatial_merge_size;
        let (pixel_values, grids, mut content) =
            prepare_fixture_images(&["tests/fixtures/p6_qwen35_vl/coco_sample.jpg"], merge);
        content.push_str(
            "Describe this image in one concise sentence. Mention the main animals and the furniture color.",
        );
        let prompt = tokenizer
            .apply_chat_template(
                &[crate::core::Message {
                    role: "user".to_owned(),
                    content,
                }],
                true,
                Some(&serde_json::json!({"enable_thinking": false})),
            )
            .expect("chat template");
        let prompt_ids = tokenizer
            .encode(&prompt, false)
            .expect("tokenize single-image prompt");
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
        let len = prompt_ids.len() as i32;
        let input_ids: Array = (prompt_ids.as_slice(), &[1_i32, len][..])
            .try_into()
            .expect("input ids");
        let image_token_id = tokenizer
            .token_to_id("<|image_pad|>")
            .map(|id| id as i32)
            .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID);
        let position_ids = crate::core::generate::build_position_ids_vl(
            &prompt_ids_i32,
            &grids,
            image_token_id,
            merge,
        )
        .expect("position ids");
        let vision_embeds = model
            .compute_vision_embeds(std::slice::from_ref(&pixel_values), &grids, ())
            .expect("vision embeds");
        let logits = model
            .forward_vl_chunk(
                &input_ids,
                &position_ids,
                None,
                None,
                None,
                Some(&vision_embeds),
                image_token_id,
                (),
            )
            .expect("forward vl chunk");

        let vocab = logits.shape().as_slice()[2];
        let flat = logits.reshape((vocab,)).expect("reshape logits");
        let mut prng = mlx::random::key(0).expect("prng");
        let got = crate::core::sampler::Sampler::greedy()
            .sample(&flat, &prompt_ids, &mut prng)
            .expect("sample");

        assert_eq!(got, 11_280_u32);
    }

    #[test]
    #[ignore = "runs one-token multi-image generation on a full local Qwen3.6 MoE checkpoint"]
    fn qwen36_moe_multi_image_generation_smoke_real_checkpoint() {
        let Some(dir) = qwen36_model_dir() else {
            return;
        };
        let loader = crate::core::Loader::open_multimodal(&dir).expect("open_multimodal");
        let tokenizer = crate::core::Tokenizer::from_loader(&loader).expect("tokenizer");
        let model = Qwen36MoeModel::from_loader(&loader).expect("model");
        let merge = model.model_meta().spatial_merge_size;
        let (pixel_values, grids, mut content) = prepare_fixture_images(
            &[
                "tests/fixtures/p6_qwen35_vl/multi_image/image_0.jpg",
                "tests/fixtures/p6_qwen35_vl/multi_image/image_1.jpg",
            ],
            merge,
        );
        content.push_str("Compare these images in one short sentence.");
        let prompt = tokenizer
            .apply_chat_template(
                &[crate::core::Message {
                    role: "user".to_owned(),
                    content,
                }],
                true,
                None,
            )
            .expect("chat template");
        let prompt_ids = tokenizer
            .encode(&prompt, false)
            .expect("tokenize multi-image prompt");
        let request = generation_request(
            &tokenizer,
            prompt_ids,
            1,
            Some(pixel_values),
            Some(grids),
            merge,
        );
        let mut stream = crate::core::generate::GenerationStream::new(&model, &tokenizer, request)
            .expect("stream");
        assert!(stream.next_token().expect("next token").is_some());
    }
}
