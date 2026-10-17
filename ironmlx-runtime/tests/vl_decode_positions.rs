//! Tokens decoded after an image prompt continue at the prompt's MRoPE
//! positions (Qwen-VL `get_rope_index`): token index + the prompt's delta,
//! on all three streams.
//!
//! `GenerationStream` greedy decoding is compared with a reference loop over
//! the same model calls (one VL prefill, then one-token forwards) whose decode
//! positions are written out explicitly: with the delta the tokens must be
//! identical; the token-index variant (no delta) is reported for contrast.
//!
//! Run with:
//! ```text
//! QWEN_VL_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-4bit \
//! MLX_DIR=$HOME/.local/mlx \
//! cargo test -p ironmlx-runtime --release --test vl_decode_positions -- --ignored --nocapture
//! ```

use std::path::Path;

use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::model::Model;
use ironmlx_lm::core::model_input::{
    build_position_ids, build_position_ids_vl, vl_rope_delta, IMAGE_TOKEN_ID,
};
use ironmlx_lm::core::vision::DenseVlMethods;
use ironmlx_lm::core::{Loader, Tokenizer};
use ironmlx_lm::models::qwen3_5::image_processor;
use ironmlx_lm::models::Qwen35MoeModel;
use ironmlx_runtime::core::generate::GenerationStream;
use ironmlx_runtime::core::generation_types::GenerateRequest;
use mlx::{Array, StreamOrDevice};

const IMAGE: &str = "../ironmlx/tests/fixtures/qwen35_vl/coco_sample.jpg";
const NEW_TOKENS: usize = 96;

fn argmax(logits: &Array) -> u32 {
    let flat = logits.reshape((-1_i32,)).unwrap();
    mlx::ops::reduction::argmax(&flat, -1, false)
        .unwrap()
        .to_vec::<u32>()
        .unwrap()[0]
}

/// Greedy decoding with explicit decode positions `index + delta`.
fn reference(
    model: &Qwen35MoeModel,
    prompt_ids: &[u32],
    embeds: &Array,
    prompt_positions: &Array,
    delta: i32,
) -> Vec<u32> {
    let prompt_len = prompt_ids.len();
    let cap = ((prompt_len + NEW_TOKENS) as i32)
        .max(ironmlx_lm::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
    let mut cache = model.make_cache(1, cap, model.cache_dtype()).unwrap();
    let input: Array = (prompt_ids, &[1_i32, prompt_len as i32][..])
        .try_into()
        .unwrap();
    let logits = model
        .forward_vl_chunk(
            &input,
            prompt_positions,
            None,
            None,
            Some(&mut cache),
            Some(embeds),
            IMAGE_TOKEN_ID,
            StreamOrDevice::default(),
        )
        .unwrap();
    let mut tokens = vec![argmax(&logits)];
    while tokens.len() < NEW_TOKENS {
        let index = (prompt_len + tokens.len() - 1) as i32;
        let token: Array = (&[*tokens.last().unwrap()][..], &[1_i32, 1][..])
            .try_into()
            .unwrap();
        let logits = model
            .forward_on(
                &token,
                &build_position_ids(index + delta, 1).unwrap(),
                None,
                None,
                Some(&mut cache),
                StreamOrDevice::default(),
            )
            .unwrap();
        tokens.push(argmax(&logits));
    }
    tokens
}

#[test]
#[ignore = "real model: needs QWEN_VL_MOE_MODEL and MLX_DIR"]
fn decoded_tokens_follow_the_vl_prompt_positions() {
    let mlx_dir = std::env::var("MLX_DIR").expect("MLX_DIR");
    mlx::metal::set_metallib_path(&format!("{mlx_dir}/lib/mlx.metallib")).unwrap();
    let model_dir = std::env::var("QWEN_VL_MOE_MODEL").expect("QWEN_VL_MOE_MODEL");
    let loader = Loader::open_multimodal(Path::new(&model_dir)).unwrap();
    let tokenizer = Tokenizer::from_loader(&loader).unwrap();
    let model = Qwen35MoeModel::from_loader(&loader).unwrap();

    let (pixels, grid_h, grid_w) = image_processor::preprocess(
        &std::fs::read(IMAGE).unwrap(),
        image_processor::PixelBounds::for_model(loader.model_dir(), None).unwrap(),
    )
    .unwrap();
    mlx::transforms::eval(&[&pixels]).unwrap();
    let grids = vec![(1_i32, grid_h, grid_w)];
    let pads = (grid_h * grid_w / 4) as usize;
    let prompt = format!(
        "<|im_start|>user\n<|vision_start|>{}<|vision_end|>Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|image_pad|>".repeat(pads)
    );
    let prompt_ids = tokenizer.encode(&prompt, false).unwrap();
    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
    let delta = vl_rope_delta(&prompt_i32, &grids, IMAGE_TOKEN_ID, 2).unwrap();
    assert!(
        delta < 0,
        "an image block spans fewer positions than tokens"
    );

    let request = GenerateRequest {
        priority: Default::default(),
        prompt_ids: prompt_ids.clone(),
        max_new_tokens: NEW_TOKENS,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: Some(vec![pixels.clone()]),
        image_grid_thw: Some(grids.clone()),
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        constraint: None,
    };
    let mut stream = GenerationStream::new(&model, &tokenizer, request).unwrap();
    let mut generated = Vec::new();
    while let Some(event) = stream.next_token().unwrap() {
        generated.push(event.token);
        if event.finish_reason.is_some() {
            break;
        }
    }
    drop(stream);

    let embeds =
        DenseVlMethods::compute_vision_embeds(&model, &[pixels], &grids, StreamOrDevice::default())
            .unwrap();
    let positions = build_position_ids_vl(&prompt_i32, &grids, IMAGE_TOKEN_ID, 2).unwrap();
    let spec = reference(&model, &prompt_ids, &embeds, &positions, delta);
    let token_index = reference(&model, &prompt_ids, &embeds, &positions, 0);
    let first_difference = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(x, y)| x != y);
    eprintln!(
        "prompt {} tokens, delta {delta}: first difference from the stream: spec positions {:?}, token-index positions {:?}",
        prompt_ids.len(),
        first_difference(&generated, &spec),
        first_difference(&generated, &token_index),
    );
    eprintln!(
        "stream text: {:?}",
        tokenizer.decode(&generated, true).unwrap()
    );
    assert_eq!(
        generated, spec,
        "decoding must use the prompt's MRoPE positions"
    );
}
