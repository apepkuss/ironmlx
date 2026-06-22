use ironmlx::core::speculative::{
    resolve_speculative_tokens, MtpSpeculativeConfig, MtpSpeculativeModel, MtpTextGenerationStream,
};
use ironmlx::core::{GenerateRequest, Loader, Sampler, Tokenizer, TokenizerConfig};
use ironmlx::models::{Qwen35Model, Qwen35MoeModel, Qwen36MoeModel};
use mlx::{Array, Dtype, StreamOrDevice};
use std::sync::Mutex;

#[test]
fn mtp_resolution_accepts_full_draft_and_uses_bonus_token() {
    let resolved = resolve_speculative_tokens(&[11, 12], &[11, 12, 13]).unwrap();

    assert_eq!(resolved.accepted_draft_len, 2);
    assert_eq!(resolved.tokens_to_append, vec![11, 12, 13]);
    assert_eq!(resolved.accepted_verify_input_len, 3);
    assert!(!resolved.needs_rollback);
}

#[test]
fn mtp_resolution_rejects_suffix_after_first_mismatch() {
    let resolved = resolve_speculative_tokens(&[11, 99, 100], &[11, 12, 13, 14]).unwrap();

    assert_eq!(resolved.accepted_draft_len, 1);
    assert_eq!(resolved.tokens_to_append, vec![11, 12]);
    assert_eq!(resolved.accepted_verify_input_len, 2);
    assert!(resolved.needs_rollback);
}

#[test]
fn mtp_resolution_rejects_entire_draft_on_first_token_mismatch() {
    let resolved = resolve_speculative_tokens(&[99, 100], &[11, 12, 13]).unwrap();

    assert_eq!(resolved.accepted_draft_len, 0);
    assert_eq!(resolved.tokens_to_append, vec![11]);
    assert_eq!(resolved.accepted_verify_input_len, 1);
    assert!(resolved.needs_rollback);
}

#[test]
fn mtp_resolution_requires_one_more_verified_token_than_draft() {
    let err = resolve_speculative_tokens(&[11, 12], &[11, 12]).unwrap_err();

    assert!(
        err.to_string().contains("verified tokens len"),
        "unexpected error: {err}"
    );
}

#[test]
fn mtp_config_rejects_zero_draft_tokens() {
    let err = MtpSpeculativeConfig::new(0, Sampler::greedy()).unwrap_err();

    assert!(
        err.to_string().contains("max_draft_tokens"),
        "unexpected error: {err}"
    );
}

#[test]
fn mtp_config_rejects_non_greedy_sampler() {
    let err = MtpSpeculativeConfig::new(4, Sampler::greedy().with_temperature(0.7)).unwrap_err();

    assert!(
        err.to_string().contains("greedy"),
        "unexpected error: {err}"
    );
}

#[test]
fn mtp_config_accepts_greedy_sampler() {
    let cfg = MtpSpeculativeConfig::new(4, Sampler::greedy()).unwrap();

    assert_eq!(cfg.max_draft_tokens, 4);
}

#[test]
fn qwen_text_models_expose_mtp_speculative_trait() {
    fn assert_mtp_model<M: MtpSpeculativeModel>() {}

    assert_mtp_model::<Qwen35Model>();
    assert_mtp_model::<Qwen35MoeModel>();
    assert_mtp_model::<Qwen36MoeModel>();
}

struct FakeMtpHead;

struct FakeMtpModel {
    text_hidden_inputs: Mutex<Vec<Vec<u32>>>,
    mtp_calls: Mutex<usize>,
    project_calls: Mutex<usize>,
}

impl FakeMtpModel {
    fn new() -> Self {
        Self {
            text_hidden_inputs: Mutex::new(Vec::new()),
            mtp_calls: Mutex::new(0),
            project_calls: Mutex::new(0),
        }
    }

    fn text_hidden_inputs(&self) -> Vec<Vec<u32>> {
        self.text_hidden_inputs.lock().unwrap().clone()
    }
}

impl ironmlx::core::Model for FakeMtpModel {
    fn make_cache(
        &self,
        _batch: i32,
        _cap: i32,
        _dtype: Dtype,
    ) -> ironmlx::Result<Vec<ironmlx::nn::LayerCache>> {
        Ok(Vec::new())
    }

    fn forward_on(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [ironmlx::nn::LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unreachable!("MTP stream uses forward_text_hidden + project_hidden_on")
    }

    fn batched_prefill(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        _per_row_lens: &[i32],
        _cache: Option<&mut [ironmlx::nn::LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unreachable!("MTP stream text-only constructor does not use batched prefill")
    }

    fn forward_text_hidden(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [ironmlx::nn::LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        self.text_hidden_inputs
            .lock()
            .unwrap()
            .push(input_ids.to_vec::<u32>()?);
        let shape = input_ids.shape();
        let dims = shape.as_slice();
        Ok(Array::zeros((dims[0], dims[1], 4_i32), Dtype::Float32)?)
    }

    fn model_meta(&self) -> ironmlx::core::memory_budget::ModelMeta {
        ironmlx::core::memory_budget::test_meta_qwen35()
    }

    fn num_hidden_layers(&self) -> usize {
        0
    }
}

impl MtpSpeculativeModel for FakeMtpModel {
    type MtpHead = FakeMtpHead;

    fn load_mtp_head(&self, _loader: &Loader) -> ironmlx::Result<Self::MtpHead> {
        Ok(FakeMtpHead)
    }

    fn make_mtp_cache(
        &self,
        _mtp: &Self::MtpHead,
        _batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> ironmlx::Result<ironmlx::core::cache::MtpCache> {
        ironmlx::core::cache::MtpCache::new_with_cap(1, 1, 1, 1, 1, dtype, cap)
    }

    fn project_hidden_on(
        &self,
        hidden: &Array,
        _target: impl Into<StreamOrDevice>,
    ) -> ironmlx::Result<Array> {
        let call_idx = {
            let mut calls = self.project_calls.lock().unwrap();
            let idx = *calls;
            *calls += 1;
            idx
        };
        let seq = hidden.shape().as_slice()[1] as usize;
        let mut flat = vec![0.0_f32; seq * 128];
        let tokens: Vec<usize> = match call_idx {
            0 => vec![10],
            1 => vec![11, 12, 13],
            _ => vec![0; seq],
        };
        for (row, token) in tokens.into_iter().enumerate().take(seq) {
            flat[row * 128 + token] = 100.0;
        }
        Ok((&flat[..], &[1_i32, seq as i32, 128_i32][..]).try_into()?)
    }

    fn mtp_hidden_size(&self, _mtp: &Self::MtpHead) -> i32 {
        4
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        Dtype::Float32
    }

    fn mtp_forward_hidden_on(
        &self,
        _mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        _position_ids: &Array,
        _mask: Option<&Array>,
        mtp_cache: Option<&mut ironmlx::core::cache::MtpCache>,
        _target: impl Into<StreamOrDevice>,
    ) -> ironmlx::Result<Array> {
        if let Some(cache) = mtp_cache {
            let seq = next_token_ids.shape().as_slice()[1];
            let k = Array::zeros((1_i32, 1_i32, seq, 1_i32), Dtype::Bfloat16)?;
            let v = Array::zeros((1_i32, 1_i32, seq, 1_i32), Dtype::Bfloat16)?;
            cache.layer_mut(0).update_and_fetch(&k, &v, &[seq])?;
        }
        Ok(hidden_states.clone())
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut ironmlx::core::cache::MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> ironmlx::Result<ironmlx::nn::MtpStepOutput> {
        let hidden_states = self.mtp_forward_hidden_on(
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )?;
        let call_idx = {
            let mut calls = self.mtp_calls.lock().unwrap();
            let idx = *calls;
            *calls += 1;
            idx
        };
        let token = if call_idx == 0 { 11 } else { 99 };
        let mut flat = vec![0.0_f32; 128];
        flat[token] = 100.0;
        Ok(ironmlx::nn::MtpStepOutput {
            hidden_states,
            logits: (&flat[..], &[1_i32, 1_i32, 128_i32][..]).try_into()?,
        })
    }
}

fn minimal_tokenizer() -> Tokenizer {
    const JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"p1":1,"p2":2,"t10":10,"t11":11,"t12":12,"t13":13,"t99":99},"unk_token":"[UNK]"}}"#;
    let path = std::env::temp_dir().join(format!(
        "ironmlx-mtp-tokenizer-{}-{:?}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, JSON).unwrap();
    let cfg = TokenizerConfig {
        chat_template: None,
        eos_token: None,
        bos_token: None,
        pad_token: None,
        eos_token_id: None,
    };
    let tok = Tokenizer::from_files(&path, &cfg).unwrap();
    let _ = std::fs::remove_file(path);
    tok
}

#[test]
fn mtp_stream_rolls_back_and_replays_accepted_prefix_after_partial_reject() {
    let model = FakeMtpModel::new();
    let mtp = FakeMtpHead;
    let tokenizer = minimal_tokenizer();
    let request = GenerateRequest {
        prompt_ids: vec![1, 2],
        max_new_tokens: 3,
        sampler: Sampler::greedy(),
        stop_token_ids: vec![127],
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    };
    let cfg = MtpSpeculativeConfig::new(2, request.sampler).unwrap();
    let mut stream =
        MtpTextGenerationStream::new_text_only(&model, &mtp, &tokenizer, request, cfg).unwrap();

    let first = stream.next_token().unwrap().unwrap();
    assert_eq!(first.token, 10);
    let stats = stream.stats();
    assert_eq!(stats.windows, 1);
    assert_eq!(stats.drafted_tokens, 2);
    assert_eq!(stats.accepted_draft_tokens, 1);
    assert_eq!(stats.rollback_count, 1);

    assert_eq!(
        model.text_hidden_inputs(),
        vec![vec![1, 2], vec![10, 11, 99], vec![10, 11]]
    );

    let second = stream.next_token().unwrap().unwrap();
    assert_eq!(second.token, 11);
    let third = stream.next_token().unwrap().unwrap();
    assert_eq!(third.token, 12);
    assert_eq!(third.finish_reason, Some("length"));
}
