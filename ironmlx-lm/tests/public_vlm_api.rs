//! Real-model validation through the public model-library API.
use ironmlx_lm::models::Qwen35Model;
use mlx::{Array, Dtype};
type ImageGrid = (i32, i32, i32);

/// Integration test: B=1 with a single image — `batched_prefill_vl` must
/// produce logits numerically equivalent to `forward_vl` on the same
/// single-stream input (both paths share vision encoder + scatter +
/// transformer + last-position project). Allows small bf16-roundoff
/// tolerance (max_abs < 1e-3) and requires bit-identical greedy argmax.
///
/// Run with:
/// ```
/// IRONMLX_MODEL_DIR=<path> cargo test -p ironmlx-lm --test public_vlm_api \
///   batched_prefill_vl_b1_matches_forward_vl -- --ignored --nocapture
/// ```
#[test]
#[ignore] // real-model heavy: needs IRONMLX_MODEL_DIR
fn batched_prefill_vl_b1_matches_forward_vl() {
    use ironmlx_lm::core::model_input::{
        build_batch_attention_mask, build_batch_linear_mask, build_position_ids_vl,
        build_position_ids_vl_batched, IMAGE_TOKEN_ID,
    };
    use ironmlx_lm::core::Loader;

    let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
        let glob = format!(
            "{}/.ironmlx/models/huggingface/mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
            std::env::var("HOME").unwrap()
        );
        let entries = std::fs::read_dir(&glob).expect("snapshots dir");
        entries
            .filter_map(|e| e.ok())
            .next()
            .expect("snapshot")
            .path()
            .to_string_lossy()
            .into_owned()
    });
    let model_path = std::path::PathBuf::from(model_dir);

    let loader = Loader::open_multimodal(&model_path).expect("Loader::open_multimodal");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");

    // Synthesize a real preprocessed pixel_values from image_0 fixture.
    let fixture_bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ironmlx/tests/fixtures/qwen35_vl/multi_image/image_0.jpg"
    ))
    .expect("image_0 fixture");
    let (pixel_values, grid_h, grid_w) =
        ironmlx_lm::models::qwen3_5::image_processor::preprocess(&fixture_bytes)
            .expect("preprocess");
    let merge_size = 2_i32;
    let grids_real: Vec<(i32, i32, i32)> = vec![(1, grid_h, grid_w)];
    let n_pads = (grid_h * grid_w / (merge_size * merge_size)) as usize;
    // Compose prompt: [1, 2, 3, IMG×n_pads, 4, 5]
    let mut prompt_ids: Vec<i32> = vec![1, 2, 3];
    prompt_ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, n_pads));
    prompt_ids.extend([4_i32, 5]);
    let prompt_len = prompt_ids.len() as i32;

    // forward_vl baseline (B=1)
    let input_ids_b1: Array = (&prompt_ids[..], &[1_i32, prompt_len][..])
        .try_into()
        .unwrap();
    let position_ids_b1 =
        build_position_ids_vl(&prompt_ids, &grids_real, IMAGE_TOKEN_ID, merge_size).unwrap();
    let mut cache_a = model.make_cache(1, prompt_len, Dtype::Bfloat16).unwrap();
    let logits_a = model
        .forward_vl(
            &input_ids_b1,
            &position_ids_b1,
            None,
            None,
            Some(&mut cache_a),
            Some(std::slice::from_ref(&pixel_values)),
            Some(&grids_real),
            IMAGE_TOKEN_ID,
            (),
        )
        .unwrap();

    // batched_prefill_vl B=1
    let position_ids_batched = build_position_ids_vl_batched(
        &[&prompt_ids[..]],
        &[Some(&grids_real[..])],
        IMAGE_TOKEN_ID,
        merge_size,
        prompt_len,
    )
    .unwrap();
    let attention_mask =
        build_batch_attention_mask(&[prompt_len], prompt_len, Dtype::Bfloat16).unwrap();
    let linear_mask = build_batch_linear_mask(&[prompt_len], prompt_len).unwrap();
    let per_row_pv: Vec<Option<&[Array]>> = vec![Some(std::slice::from_ref(&pixel_values))];
    let per_row_grids: Vec<Option<&[ImageGrid]>> = vec![Some(&grids_real[..])];
    let mut cache_b = model.make_cache(1, prompt_len, Dtype::Bfloat16).unwrap();
    let logits_b = model
        .batched_prefill_vl(
            &input_ids_b1,
            &position_ids_batched,
            &attention_mask,
            &linear_mask,
            &[prompt_len],
            &per_row_pv,
            &per_row_grids,
            IMAGE_TOKEN_ID,
            Some(&mut cache_b),
            (),
        )
        .unwrap();

    let a: Vec<f32> = mlx::ops::astype(&logits_a, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    let b_vec: Vec<f32> = mlx::ops::astype(&logits_b, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();

    assert_eq!(a.len(), b_vec.len(), "logits length");
    // bf16 round-trip can introduce ULP-level diffs; require <1e-3 max-abs diff
    // and bit-identical greedy argmax.
    let mut max_abs = 0.0_f32;
    for (av, bv) in a.iter().zip(b_vec.iter()) {
        let d = (av - bv).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(max_abs < 1e-3, "max-abs logits diff = {max_abs} >= 1e-3");

    let argmax_a = a
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap()
        .0;
    let argmax_b = b_vec
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax_a, argmax_b, "greedy argmax mismatch");
}
