use ironmlx::models::Qwen36MoeModel;
use mlx::Array;
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
            ironmlx::models::qwen3_5::image_processor::preprocess(&bytes).expect("preprocess");
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
    tokenizer: &ironmlx::core::Tokenizer,
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    pixel_values: Option<Array>,
    image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    image_spatial_merge_size: i32,
) -> ironmlx::core::generation_types::GenerateRequest {
    ironmlx::core::generation_types::GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: ironmlx::core::sampler::Sampler::greedy(),
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
            .unwrap_or(ironmlx::core::model_input::IMAGE_TOKEN_ID),
        constraint: None,
    }
}
#[test]
#[ignore = "runs one-token generation on a full local Qwen3.6 MoE checkpoint"]
fn qwen36_moe_text_generation_smoke_real_checkpoint() {
    let Some(dir) = qwen36_model_dir() else {
        return;
    };
    let loader = ironmlx::core::Loader::open(&dir).expect("open");
    let tokenizer = ironmlx::core::Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen36MoeModel::from_loader(&loader).expect("model");
    let prompt = tokenizer
        .apply_chat_template(
            &[ironmlx::core::Message {
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
        ironmlx::core::generate::GenerationStream::new_text_only(&model, &tokenizer, request)
            .expect("stream");
    assert!(stream.next_token().expect("next token").is_some());
}

#[test]
#[ignore = "runs one-token single-image generation on a full local Qwen3.6 MoE checkpoint"]
fn qwen36_moe_single_image_generation_smoke_real_checkpoint() {
    let Some(dir) = qwen36_model_dir() else {
        return;
    };
    let loader = ironmlx::core::Loader::open_multimodal(&dir).expect("open_multimodal");
    let tokenizer = ironmlx::core::Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen36MoeModel::from_loader(&loader).expect("model");
    let merge = model.model_meta().spatial_merge_size;
    let (pixel_values, grids, mut content) =
        prepare_fixture_images(&["tests/fixtures/qwen35_vl/multi_image/image_0.jpg"], merge);
    content.push_str("Describe this image briefly.");
    let prompt = tokenizer
        .apply_chat_template(
            &[ironmlx::core::Message {
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
    let mut stream = ironmlx::core::generate::GenerationStream::new(&model, &tokenizer, request)
        .expect("stream");
    assert!(stream.next_token().expect("next token").is_some());
}

#[test]
#[ignore = "runs one-token multi-image generation on a full local Qwen3.6 MoE checkpoint"]
fn qwen36_moe_multi_image_generation_smoke_real_checkpoint() {
    let Some(dir) = qwen36_model_dir() else {
        return;
    };
    let loader = ironmlx::core::Loader::open_multimodal(&dir).expect("open_multimodal");
    let tokenizer = ironmlx::core::Tokenizer::from_loader(&loader).expect("tokenizer");
    let model = Qwen36MoeModel::from_loader(&loader).expect("model");
    let merge = model.model_meta().spatial_merge_size;
    let (pixel_values, grids, mut content) = prepare_fixture_images(
        &[
            "tests/fixtures/qwen35_vl/multi_image/image_0.jpg",
            "tests/fixtures/qwen35_vl/multi_image/image_1.jpg",
        ],
        merge,
    );
    content.push_str("Compare these images in one short sentence.");
    let prompt = tokenizer
        .apply_chat_template(
            &[ironmlx::core::Message {
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
    let mut stream = ironmlx::core::generate::GenerationStream::new(&model, &tokenizer, request)
        .expect("stream");
    assert!(stream.next_token().expect("next token").is_some());
}
