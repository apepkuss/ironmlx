//! P5b smoke + first-token argmax alignment vs mlx-vlm baseline.
//!
//! mlx-vlm reference (verified live run, 2026-05-19): prompt "Once upon a
//! time", temp=0 greedy → first generated token id = 11 (`,` comma,
//! logit=22.125). Top-5: [11, 303, 264, 1017, 449].
//!
//! NOTE: The task spec originally stated id=310 (a `,` comma), but 310
//! actually decodes to ` to`; the comma is id=11. Verified by running
//! mlx-vlm directly and inspecting argmax of last-position logits.
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snapshot-path> \
//!     MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --test p5_qwen35_moe_smoke \
//!       -- --ignored --nocapture --test-threads=1

use mlx::Dtype;

use ironmlx::core::generate::build_position_ids;
use ironmlx::core::{Loader, Model, Tokenizer};
use ironmlx::models::qwen3_5_moe::MIN_KV_CACHE_CAP_FOR_GPU_PERF;
use ironmlx::models::Qwen35MoeModel;

fn locate_snapshot() -> String {
    if let Ok(p) = std::env::var("IRONMLX_MOE_MODEL_DIR") {
        return p;
    }
    let home = std::env::var("HOME").expect("HOME env");
    let snapshots =
        format!("{home}/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots");
    let entries = std::fs::read_dir(&snapshots).expect("snapshots dir missing");
    let first = entries
        .filter_map(|e| e.ok())
        .next()
        .expect("at least one snapshot");
    first.path().to_string_lossy().into_owned()
}

#[test]
#[ignore]
fn p5b_smoke_forward_shape_and_finite() {
    let dir = locate_snapshot();
    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");
    eprintln!("loaded model: {} layers", model.config().num_hidden_layers);

    let input_ids: mlx::Array = (&[100_i32, 200, 300, 400][..], &[1_i32, 4][..])
        .try_into()
        .unwrap();
    let pos = build_position_ids(0, 4).expect("build_position_ids");

    let mut cache = Model::make_cache(&model, 1, 16, Dtype::Bfloat16).expect("make_cache");
    let logits = Model::forward_on(
        &model,
        &input_ids,
        &pos,
        None,
        None,
        Some(&mut cache),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on");

    // Shape [1, 1, vocab=248320]
    let shape = logits.shape();
    let s = shape.as_slice();
    assert_eq!(s.len(), 3, "logits should be rank-3, got {:?}", s);
    assert_eq!(s[0], 1, "B");
    assert_eq!(s[1], 1, "S (last-position sliced)");
    assert_eq!(s[2], model.config().vocab_size, "vocab dim");

    // Finite values
    let v: Vec<f32> = mlx::ops::cast::astype(&logits, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "non-finite logits present");
    eprintln!("logits OK: shape={:?}, all finite", s);
}

#[test]
#[ignore]
fn p5b_first_token_argmax_matches_mlx_vlm_baseline() {
    // mlx-vlm baseline (verified 2026-05-19): "Once upon a time" greedy temp=0 →
    // first generated token id=11 (`,` comma, logit=22.125). Top-5: [11, 303, 264, 1017, 449].
    // The task spec originally said id=310, but 310 decodes to ` to`; comma is id=11.
    const EXPECTED_FIRST_TOKEN: i64 = 11;
    let dir = locate_snapshot();

    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");

    // Tokenize WITHOUT chat template (matches mlx-vlm's raw-text mode used in T0 baseline).
    // mlx-vlm called generate(prompt='Once upon a time') which tokenizes raw with BOS handled internally.
    let prompt = "Once upon a time";
    let prompt_ids = tokenizer
        .encode(prompt, /* add_special_tokens */ false)
        .expect("encode");
    eprintln!(
        "prompt token ids: {:?}",
        &prompt_ids[..prompt_ids.len().min(8)]
    );
    eprintln!("prompt token count: {}", prompt_ids.len());

    // Build [1, S] input
    let s = prompt_ids.len() as i32;
    let ids_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
    let input_ids: mlx::Array = (&ids_i32[..], &[1_i32, s][..])
        .try_into()
        .expect("input_ids");

    let pos = build_position_ids(0, s).expect("build_position_ids");

    let cap = (s + 4).max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
    let mut cache = Model::make_cache(&model, 1, cap, Dtype::Bfloat16).expect("make_cache");
    let logits = Model::forward_on(
        &model,
        &input_ids,
        &pos,
        None,
        None,
        Some(&mut cache),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on");

    // argmax of last-position logits [1, 1, vocab]
    let vocab_size = model.config().vocab_size as usize;
    let v: Vec<f32> = mlx::ops::cast::astype(&logits, Dtype::Float32)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(v.len(), vocab_size, "vocab length mismatch");

    let (argmax_idx, argmax_val) = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .expect("non-empty logits");
    let top5 = {
        let mut idx: Vec<usize> = (0..vocab_size).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.into_iter().take(5).collect::<Vec<_>>()
    };
    eprintln!("argmax token id: {} (logit={:.4})", argmax_idx, argmax_val);
    eprintln!("top-5 token ids: {:?}", top5);
    eprintln!(
        "expected (mlx-vlm baseline): {} (`,` comma)",
        EXPECTED_FIRST_TOKEN
    );

    assert_eq!(
        argmax_idx as i64, EXPECTED_FIRST_TOKEN,
        "first-token argmax mismatch vs mlx-vlm baseline.\n\
         got: token id {} (logit={:.4})\n\
         expected: token id {}\n\
         top-5: {:?}\n\
         \nDivergence diagnosis:\n\
         - If argmax is in top-5 of mlx-vlm output, likely numerical precision\n\
           (bf16 round-off on shared/routed accumulation).\n\
         - If argmax is wildly off, suspect axis ordering in softmax/topk/sum or\n\
           sigmoid placement (shared_expert_gate vs routed scores).",
        argmax_idx, argmax_val, EXPECTED_FIRST_TOKEN, top5
    );
}
