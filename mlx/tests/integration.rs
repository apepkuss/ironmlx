use ironmlx_core::{Array, Device, Stream, init};
use std::path::Path;

fn gpu_stream() -> Stream {
    init();
    Stream::default_stream(&Device::gpu())
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[test]
fn test_gpu_smoke() {
    let s = gpu_stream();
    let a = Array::from_float(3.0);
    let b = Array::from_float(4.0);
    let c = ironmlx_core::ops::add(&a, &b, &s).unwrap();
    assert!((c.item_f32().unwrap() - 7.0).abs() < 1e-6);
}

#[test]
fn test_cpu_smoke() {
    init();
    let s = Stream::default_stream(&Device::cpu());
    let a = Array::from_int(10);
    let b = Array::from_int(5);
    let c = ironmlx_core::ops::subtract(&a, &b, &s).unwrap();
    assert_eq!(c.item_i32().unwrap(), 5);
}

// ── default_stream() ──────────────────────────────────────────────────────────

#[test]
fn test_default_gpu_stream() {
    ironmlx_core::init();
    let s = ironmlx_core::default_stream(ironmlx_core::DeviceType::Gpu);
    let a = Array::from_float(2.0);
    let r = ironmlx_core::ops::square(&a, &s).unwrap();
    assert!((r.item_f32().unwrap() - 4.0).abs() < 1e-6);
}

// ── softmax / relu ────────────────────────────────────────────────────────────

#[test]
fn test_softmax() {
    let s = gpu_stream();
    let a = Array::from_slice_f32(&[1.0, 2.0, 3.0]);
    let b = ironmlx_core::ops::softmax(&a, &[-1], &s).unwrap();
    let v = b.to_vec_f32().unwrap();
    let total: f32 = v.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "softmax should sum to 1, got {total}"
    );
    assert!(
        v[2] > v[1] && v[1] > v[0],
        "larger input → larger probability"
    );
}

#[test]
fn test_relu() {
    let s = gpu_stream();
    let a = Array::from_slice_f32(&[-2.0, 0.0, 3.0]);
    let b = ironmlx_core::ops::relu(&a, &s).unwrap();
    let v = b.to_vec_f32().unwrap();
    assert!((v[0] - 0.0).abs() < 1e-6, "negative → 0");
    assert!((v[1] - 0.0).abs() < 1e-6, "zero → 0");
    assert!((v[2] - 3.0).abs() < 1e-6, "positive → unchanged");
}

// ── sum / mean with axes ──────────────────────────────────────────────────────

#[test]
fn test_sum_all() {
    let s = gpu_stream();
    let a = Array::from_slice_f32(&[1.0, 2.0, 3.0]);
    let r = ironmlx_core::ops::sum(&a, &[], false, &s).unwrap();
    assert!((r.item_f32().unwrap() - 6.0).abs() < 1e-5);
}

#[test]
fn test_sum_axis() {
    let s = gpu_stream();
    // shape [2, 3], sum along axis 0 → shape [3]
    let a = Array::from_slice_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = ironmlx_core::ops::sum(&a, &[0], false, &s).unwrap();
    assert_eq!(r.shape(), vec![3]);
    let v = r.to_vec_f32().unwrap();
    assert!((v[0] - 5.0).abs() < 1e-5);
    assert!((v[1] - 7.0).abs() < 1e-5);
    assert!((v[2] - 9.0).abs() < 1e-5);
}

#[test]
fn test_sum_keepdims() {
    let s = gpu_stream();
    let a = Array::from_slice_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = ironmlx_core::ops::sum(&a, &[1], true, &s).unwrap();
    assert_eq!(r.shape(), vec![2, 1]);
}

#[test]
fn test_mean_axis() {
    let s = gpu_stream();
    let a = Array::from_slice_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = ironmlx_core::ops::mean(&a, &[1], false, &s).unwrap();
    assert_eq!(r.shape(), vec![2]);
    let v = r.to_vec_f32().unwrap();
    assert!((v[0] - 2.0).abs() < 1e-5); // mean of [1,2,3]
    assert!((v[1] - 5.0).abs() < 1e-5); // mean of [4,5,6]
}

// ── reshape / transpose ───────────────────────────────────────────────────────

#[test]
fn test_reshape() {
    let s = gpu_stream();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_slice_f32_shape(&data, &[2, 3]);
    let b = ironmlx_core::ops::reshape(&a, &[3, 2], &s).unwrap();
    assert_eq!(b.shape(), vec![3, 2]);
    assert_eq!(b.size(), 6);
    assert_eq!(b.to_vec_f32().unwrap(), data);
}

#[test]
fn test_transpose() {
    let s = gpu_stream();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_slice_f32_shape(&data, &[2, 3]);
    let b = ironmlx_core::ops::transpose(&a, &s).unwrap();
    assert_eq!(b.shape(), vec![3, 2]);
}

#[test]
fn test_transpose_axes() {
    let s = gpu_stream();
    // shape [2, 3, 4] → transpose axes [2, 0, 1] → shape [4, 2, 3]
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let a = Array::from_slice_f32_shape(&data, &[2, 3, 4]);
    let b = ironmlx_core::ops::transpose_axes(&a, &[2, 0, 1], &s).unwrap();
    assert_eq!(b.shape(), vec![4, 2, 3]);
}

// ── Operator overloading ──────────────────────────────────────────────────────

#[test]
fn test_operator_add() {
    ironmlx_core::init();
    let a = Array::from_float(3.0);
    let b = Array::from_float(4.0);
    let c = (&a + &b).unwrap();
    assert!((c.item_f32().unwrap() - 7.0).abs() < 1e-6);
}

#[test]
fn test_operator_sub() {
    ironmlx_core::init();
    let a = Array::from_float(9.0);
    let b = Array::from_float(4.0);
    let c = (&a - &b).unwrap();
    assert!((c.item_f32().unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn test_operator_mul() {
    ironmlx_core::init();
    let a = Array::from_float(3.0);
    let b = Array::from_float(4.0);
    let c = (&a * &b).unwrap();
    assert!((c.item_f32().unwrap() - 12.0).abs() < 1e-6);
}

#[test]
fn test_operator_div() {
    ironmlx_core::init();
    let a = Array::from_float(8.0);
    let b = Array::from_float(2.0);
    let c = (&a / &b).unwrap();
    assert!((c.item_f32().unwrap() - 4.0).abs() < 1e-6);
}

// ── Array enhancements ────────────────────────────────────────────────────────

#[test]
fn test_array_clone() {
    let a = Array::from_float(1.5);
    let b = a.clone();
    assert!((b.item_f32().unwrap() - 1.5).abs() < 1e-6);
}

#[test]
fn test_to_vec_i32() {
    let data = vec![1i32, 2, 3, 4];
    let a = Array::from_slice_i32(&data);
    assert_eq!(a.to_vec_i32().unwrap(), data);
}

#[test]
fn test_array_2d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_slice_f32_shape(&data, &[2, 3]);
    assert_eq!(a.shape(), vec![2, 3]);
    assert_eq!(a.size(), 6);
    assert_eq!(a.to_vec_f32().unwrap(), data);
}

// ── Model-dependent tests (require Qwen3-0.6B-4bit) ────────────────────────
//
// Run with: cargo test -p ironmlx-core -- --ignored

const QWEN3_MODEL_DIR: &str = "/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8";

fn load_test_model() -> Option<(
    ironmlx_core::model::LlamaModel,
    ironmlx_core::generate::Tokenizer,
    ironmlx_core::model::ModelConfig,
)> {
    let dir = Path::new(QWEN3_MODEL_DIR);
    if !dir.exists() {
        return None;
    }
    init();
    let config =
        ironmlx_core::model::ModelConfig::from_file(dir.join("config.json").to_str().unwrap())
            .ok()?;
    let stream = Stream::new(&Device::gpu());
    let weights = ironmlx_core::model::load_model_weights(dir, &stream).ok()?;
    let model = ironmlx_core::model::build_model(&config, &weights).ok()?;
    let tokenizer =
        ironmlx_core::generate::Tokenizer::from_file(dir.join("tokenizer.json").to_str().unwrap())
            .ok()?;
    Some((model, tokenizer, config))
}

#[test]
#[ignore]
fn test_batch_generator_single_request() {
    let (model, tokenizer, config) = load_test_model().expect("Qwen3-0.6B-4bit not available");
    let mut batch = ironmlx_core::generate::BatchGenerator::new(&model);

    let prompt = tokenizer.encode("Hello").unwrap();
    let sampler = ironmlx_core::generate::SamplerConfig::greedy();
    let eos = config.eos_token_id as i32;

    let (uid, first_response) = batch.insert(&prompt, sampler, eos, 10).unwrap();
    assert_eq!(uid, 0);
    assert!(first_response.finish_reason.is_none() || first_response.token_id != 0);
    assert_eq!(batch.active_count(), 1);

    // Step a few times
    for _ in 0..5 {
        if !batch.is_active(uid) {
            break;
        }
        let responses = batch.step().unwrap();
        assert!(!responses.is_empty());
    }
}

#[test]
#[ignore]
fn test_batch_generator_multiple_requests() {
    let (model, tokenizer, config) = load_test_model().expect("Qwen3-0.6B-4bit not available");
    let mut batch = ironmlx_core::generate::BatchGenerator::new(&model);

    let eos = config.eos_token_id as i32;

    // Insert two requests
    let prompt_a = tokenizer.encode("What is 1+1?").unwrap();
    let prompt_b = tokenizer.encode("Say hello").unwrap();

    let (uid_a, _) = batch
        .insert(
            &prompt_a,
            ironmlx_core::generate::SamplerConfig::greedy(),
            eos,
            5,
        )
        .unwrap();
    let (uid_b, _) = batch
        .insert(
            &prompt_b,
            ironmlx_core::generate::SamplerConfig::greedy(),
            eos,
            5,
        )
        .unwrap();

    assert_ne!(uid_a, uid_b);
    assert_eq!(batch.active_count(), 2);

    // Step until both finish
    for _ in 0..10 {
        if batch.active_count() == 0 {
            break;
        }
        let responses = batch.step().unwrap();
        assert!(responses.len() <= 2);
    }
}

#[test]
#[ignore]
fn test_batch_generator_abort() {
    let (model, tokenizer, config) = load_test_model().expect("Qwen3-0.6B-4bit not available");
    let mut batch = ironmlx_core::generate::BatchGenerator::new(&model);

    let prompt = tokenizer.encode("Tell me a story").unwrap();
    let eos = config.eos_token_id as i32;

    let (uid, _) = batch
        .insert(
            &prompt,
            ironmlx_core::generate::SamplerConfig::greedy(),
            eos,
            100,
        )
        .unwrap();
    assert_eq!(batch.active_count(), 1);

    // Abort
    batch.remove(uid);
    assert_eq!(batch.active_count(), 0);
    assert!(!batch.is_active(uid));
}

#[test]
#[ignore]
fn test_stream_generate_produces_tokens() {
    let (model, tokenizer, config) = load_test_model().expect("Qwen3-0.6B-4bit not available");

    let prompt = tokenizer.encode("The answer is").unwrap();
    let sampler = ironmlx_core::generate::SamplerConfig::greedy();
    let eos = config.eos_token_id as i32;

    let mut tokens = Vec::new();
    let reason =
        ironmlx_core::generate::stream_generate(&model, &prompt, 20, &sampler, eos, |token| {
            tokens.push(token);
            true
        })
        .unwrap();

    assert!(!tokens.is_empty(), "should generate at least 1 token");
    assert!(
        reason == ironmlx_core::generate::StopReason::MaxTokens
            || reason == ironmlx_core::generate::StopReason::Eos
    );

    // Verify tokens can be decoded
    let text = tokenizer.decode(&tokens).unwrap();
    assert!(!text.is_empty());
}

#[test]
#[ignore]
fn test_stream_generate_stops_on_abort() {
    let (model, tokenizer, config) = load_test_model().expect("Qwen3-0.6B-4bit not available");

    let prompt = tokenizer.encode("Once upon a time").unwrap();
    let sampler = ironmlx_core::generate::SamplerConfig::greedy();
    let eos = config.eos_token_id as i32;

    let mut count = 0;
    let _reason =
        ironmlx_core::generate::stream_generate(&model, &prompt, 100, &sampler, eos, |_token| {
            count += 1;
            count < 5 // abort after 5 tokens
        })
        .unwrap();

    assert_eq!(count, 5);
}
