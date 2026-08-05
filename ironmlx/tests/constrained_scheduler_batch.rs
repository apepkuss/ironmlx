mod common;

use common::constrained::{
    byte_vocab_size, weather_constraint_plan, weather_constraint_plan_with_options,
};
use ironmlx::core::constrained::{ToolChoiceConstraint, ToolConstraintOptions};
use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::{Model, Sampler, Scheduler};
use ironmlx::nn::LayerCache;
use mlx::{Array, Dtype, StreamOrDevice};

struct ConstraintBatchModel;

impl ConstraintBatchModel {
    fn logits(input_ids: &Array) -> ironmlx::Result<Array> {
        let batch = input_ids.shape().as_slice()[0] as usize;
        let vocab = byte_vocab_size();
        let mut values = vec![0.0_f32; batch * vocab];
        for row in 0..batch {
            values[row * vocab + b'x' as usize] = 100.0;
            values[row * vocab + b'g' as usize] = 90.0;
            values[row * vocab + b'e' as usize] = 80.0;
        }
        let logits: Array = (&values[..], &[batch as i32, vocab as i32][..]).try_into()?;
        Ok(logits.reshape((batch as i32, 1_i32, vocab as i32))?)
    }
}

impl Model for ConstraintBatchModel {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> ironmlx::Result<Vec<LayerCache>> {
        Ok(vec![LayerCache::Full(ironmlx::core::KVCache::new(
            batch, 1, 1, 1, dtype, cap,
        ))])
    }

    fn forward_on(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        Self::logits(input_ids)
    }

    fn batched_prefill(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        _per_row_lens: &[i32],
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        Self::logits(input_ids)
    }

    fn forward_text_hidden(
        &self,
        input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        let dims = input_ids.shape();
        let dims = dims.as_slice();
        Ok(Array::zeros((dims[0], dims[1], 4_i32), Dtype::Float32)?)
    }

    fn model_meta(&self) -> ironmlx::core::memory_budget::ModelMeta {
        ironmlx::core::memory_budget::test_meta_qwen35()
    }

    fn num_hidden_layers(&self) -> usize {
        1
    }
}

impl DenseVlMethods for ConstraintBatchModel {
    fn batched_prefill_vl(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _attention_mask: &Array,
        _linear_attention_mask: &Array,
        _per_row_lens: &[i32],
        _per_row_pixel_values: &[Option<&[Array]>],
        _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        _image_token_id: i32,
        _cache: Option<&mut [LayerCache]>,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unreachable!("constraint batch test is text-only")
    }

    fn estimate_vision_prefill_peak_bytes(
        &self,
        _pixel_values: &[Array],
        _grid_thw: &[(i32, i32, i32)],
    ) -> ironmlx::Result<usize> {
        unreachable!("constraint batch test is text-only")
    }

    fn compute_vision_embeds(
        &self,
        _pixel_values: &[Array],
        _grid_thw: &[(i32, i32, i32)],
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unreachable!("constraint batch test is text-only")
    }

    fn forward_vl_chunk(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _vision_embeds_slice: Option<&Array>,
        _image_token_id: i32,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unreachable!("constraint batch test is text-only")
    }

    fn forward_vl_hidden(
        &self,
        _input_ids: &Array,
        _position_ids: &Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&Array>,
        _cache: Option<&mut [LayerCache]>,
        _vision_embeds_slice: Option<&Array>,
        _image_token_id: i32,
        _target: StreamOrDevice,
    ) -> ironmlx::Result<Array> {
        unreachable!("constraint batch test is text-only")
    }
}

fn request(constraint: bool) -> GenerateRequest {
    GenerateRequest {
        prompt_ids: vec![1, 2],
        max_new_tokens: 4,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        constraint: constraint.then(weather_constraint_plan),
    }
}

#[test]
fn scheduler_b2_keeps_constrained_and_unconstrained_rows_isolated() {
    let model = ConstraintBatchModel;
    let mut scheduler = Scheduler::<ConstraintBatchModel>::new(
        2,
        4096,
        ironmlx::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler");
    let constrained_id = scheduler.admit(request(true)).expect("constrained admit");
    let ordinary_id = scheduler.admit(request(false)).expect("ordinary admit");

    let prefix = b"<tool_call><function="
        .iter()
        .map(|byte| u32::from(*byte))
        .collect::<Vec<_>>();
    scheduler
        .get_mut(constrained_id)
        .expect("constrained state")
        .constraint
        .as_mut()
        .expect("constraint session")
        .commit_tokens(&prefix)
        .expect("advance matcher to function name");

    let prefill = scheduler.prefill_admitted(&model).expect("B=2 prefill");
    assert_eq!(prefill.len(), 2);
    assert_eq!(
        (prefill[0].id, prefill[0].token),
        (constrained_id, u32::from(b'g'))
    );
    assert_eq!(
        (prefill[1].id, prefill[1].token),
        (ordinary_id, u32::from(b'x'))
    );

    let decode = scheduler.step(&model).expect("B=2 decode");
    assert_eq!(decode.len(), 2);
    assert_eq!(
        (decode[0].id, decode[0].token),
        (constrained_id, u32::from(b'e'))
    );
    assert_eq!(
        (decode[1].id, decode[1].token),
        (ordinary_id, u32::from(b'x'))
    );

    let constrained = scheduler.get(constrained_id).expect("constrained state");
    assert_eq!(
        constrained.generated_tokens,
        vec![u32::from(b'g'), u32::from(b'e')]
    );
    assert!(constrained.constraint.is_some());
    let ordinary = scheduler.get(ordinary_id).expect("ordinary state");
    assert_eq!(
        ordinary.generated_tokens,
        vec![u32::from(b'x'), u32::from(b'x')]
    );
    assert!(ordinary.constraint.is_none());
}

#[test]
fn scheduler_preserves_required_choice_plan_state() {
    let options = ToolConstraintOptions {
        choice: ToolChoiceConstraint::Required,
        allow_parallel_calls: false,
    };
    let mut scheduler = Scheduler::<ConstraintBatchModel>::new(
        1,
        4096,
        ironmlx::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler");
    let mut constrained = request(false);
    constrained.constraint = Some(weather_constraint_plan_with_options(&options));
    let id = scheduler.admit(constrained).expect("required admit");

    let state = scheduler.get_mut(id).expect("required state");
    let constraint = state.constraint.as_mut().expect("constraint session");
    constraint
        .commit_tokens(
            &b"ordinary answer"
                .iter()
                .map(|byte| u32::from(*byte))
                .collect::<Vec<_>>(),
        )
        .expect("thinking prefix remains legal");
    assert!(!constraint.is_accepting().expect("required accepting state"));
}
