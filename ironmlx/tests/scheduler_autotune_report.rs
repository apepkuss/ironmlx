use ironmlx::core::memory_budget::ModelMeta;
use ironmlx::core::scheduler_autotune::{
    build_scheduler_autotune_report, prompt_batch_limits_for_model, PromptBatchLimit,
    SchedulerAutotuneInput,
};
use ironmlx::core::Model;
use ironmlx::nn::LayerCache;
use mlx::{Array, Dtype, StreamOrDevice};

struct BatchLimitedModel;

impl Model for BatchLimitedModel {
    fn make_cache(
        &self,
        _batch: i32,
        _cap: i32,
        _dtype: Dtype,
    ) -> ironmlx::Result<Vec<LayerCache>> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn forward_on(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn batched_prefill(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        _per_row_lens: &[i32],
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn forward_text_hidden(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unimplemented!("not used by scheduler autotune tests")
    }

    fn fresh_prefill_batch_limit(prompt_len: usize, b_max: usize) -> usize {
        if prompt_len >= 1024 {
            b_max.min(2)
        } else {
            b_max
        }
    }

    fn model_meta(&self) -> ModelMeta {
        sample_meta()
    }

    fn num_hidden_layers(&self) -> usize {
        0
    }
}

fn sample_meta() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 28,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        hidden_size: 4096,
        head_dim: None,
        weight_bytes: 3 * 1024 * 1024 * 1024,
        max_position_embeddings: 32768,
        spatial_merge_size: 2,
    }
}

fn sample_input(total_ram_bytes: usize) -> SchedulerAutotuneInput {
    SchedulerAutotuneInput {
        model_name: "test-model".to_string(),
        meta: sample_meta(),
        prefill_chunk_size: 2048,
        b_max: 4,
        admission_deadline_ms: 5,
        admission_queue_max: 32,
        requested_max_cache_cap: 32768,
        effective_cap_max: 32768,
        decode_cadence_mid_chunk_cap: 256,
        total_ram_bytes,
    }
}

#[test]
fn report_is_diagnose_only_and_never_applies_parameters() {
    let report = build_scheduler_autotune_report(
        sample_input(64 * 1024 * 1024 * 1024),
        vec![PromptBatchLimit {
            prompt_len: 2048,
            limit: 2,
        }],
    );

    assert!(report.diagnose_only);
    let text = report.render_text();
    assert!(text.contains("diagnose-only"));
    assert!(text.contains("no runtime parameters changed"));
}

#[test]
fn report_warns_when_reserved_kv_exceeds_available_budget() {
    let report = build_scheduler_autotune_report(
        sample_input(8 * 1024 * 1024 * 1024),
        vec![PromptBatchLimit {
            prompt_len: 2048,
            limit: 2,
        }],
    );

    assert!(report
        .recommendations
        .iter()
        .any(|item| item.code == "memory_budget_overrun"));
}

#[test]
fn prompt_batch_limits_sample_model_trait_policy() {
    let samples = prompt_batch_limits_for_model::<BatchLimitedModel>(4);

    assert_eq!(
        samples,
        vec![
            PromptBatchLimit {
                prompt_len: 512,
                limit: 4,
            },
            PromptBatchLimit {
                prompt_len: 1024,
                limit: 2,
            },
            PromptBatchLimit {
                prompt_len: 2048,
                limit: 2,
            },
            PromptBatchLimit {
                prompt_len: 8192,
                limit: 2,
            },
        ]
    );
}
