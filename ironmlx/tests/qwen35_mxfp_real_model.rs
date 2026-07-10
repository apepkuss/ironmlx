//! Real-checkpoint correctness gates for Qwen3.5 MXFP4 and MXFP8.
//!
//! Set `IRONMLX_TEST_REAL_MXFP=1` to run against the pinned local snapshots.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use ironmlx::core::{generate::build_position_ids, QuantMode, Tokenizer};
use ironmlx::models::Qwen35Model;
use ironmlx::Loader;
use mlx::{Array, Dtype};
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/mxfp_qwen35";

#[derive(Deserialize)]
struct LogitAnchor {
    token_id: usize,
    logit: f32,
}

#[derive(Deserialize)]
struct ReferenceFixture {
    model_id: String,
    revision: String,
    prompt: String,
    input_ids: Vec<u32>,
    vocab_size: usize,
    next_token_id: usize,
    logits_file: String,
    top_logits: Vec<LogitAnchor>,
}

struct Case {
    name: &'static str,
    model_id: &'static str,
    revision: &'static str,
    mode: QuantMode,
    bits: i32,
}

const CASES: [Case; 2] = [
    Case {
        name: "mxfp4",
        model_id: "mlx-community/Qwen3.5-4B-mxfp4",
        revision: "8e9cb97ec8ee0f6a04021220b7a6b5845353df56",
        mode: QuantMode::Mxfp4,
        bits: 4,
    },
    Case {
        name: "mxfp8",
        model_id: "mlx-community/Qwen3.5-4B-mxfp8",
        revision: "a34dd69c7f165c0db75d71061e1bd8f4aeb9eead",
        mode: QuantMode::Mxfp8,
        bits: 8,
    },
];

fn snapshot_dir(case: &Case) -> PathBuf {
    let cache_name = case.model_id.replace('/', "--");
    dirs::home_dir()
        .expect("home directory")
        .join(".ironmlx/models")
        .join(format!("models--{cache_name}"))
        .join("snapshots")
        .join(case.revision)
}

fn fixture(case: &Case) -> ReferenceFixture {
    let path = Path::new(FIXTURE_DIR).join(format!("{}.json", case.name));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn greedy_argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .expect("non-empty logits")
}

fn logits_vec(logits: &Array) -> Vec<f32> {
    mlx::ops::cast::astype(logits, Dtype::Float32)
        .expect("cast logits")
        .to_vec::<f32>()
        .expect("read logits")
}

fn run_case(case: &Case) {
    let reference = fixture(case);
    assert_eq!(reference.model_id, case.model_id);
    assert_eq!(reference.revision, case.revision);
    assert_eq!(reference.top_logits.len(), 64);

    let model_dir = snapshot_dir(case);
    assert!(
        model_dir.is_dir(),
        "missing snapshot {}",
        model_dir.display()
    );
    let loader = Loader::open(&model_dir).expect("Loader::open MXFP checkpoint");
    let qmeta = loader
        .quant_meta()
        .expect("global MXFP quantization metadata");
    assert_eq!(qmeta.mode, case.mode);
    assert_eq!(qmeta.bits, case.bits);
    assert_eq!(qmeta.group_size, 32);

    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let input_ids = tokenizer
        .encode(&reference.prompt, false)
        .expect("tokenizer.encode");
    assert_eq!(input_ids, reference.input_ids);

    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let seq = input_ids.len() as i32;
    let input: Array = (input_ids.as_slice(), &[1_i32, seq][..])
        .try_into()
        .expect("input ids array");
    let positions = build_position_ids(0, seq).expect("position ids");
    let mut cache = model
        .make_cache(1, seq + 1, Dtype::Bfloat16)
        .expect("make cache");
    let logits = model
        .forward_on(&input, &positions, Some(&[seq]), None, Some(&mut cache), ())
        .expect("MXFP forward");
    let logits = logits_vec(&logits);
    assert_eq!(logits.len(), reference.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(greedy_argmax(&logits), reference.next_token_id);

    let expected_path = Path::new(FIXTURE_DIR).join(&reference.logits_file);
    let expected = mlx::io::load_npy(
        expected_path
            .to_str()
            .expect("reference logits path must be UTF-8"),
    )
    .unwrap_or_else(|e| panic!("load {}: {e}", expected_path.display()));
    let expected = logits_vec(&expected);
    assert_eq!(logits.len(), expected.len());
    let max_abs_diff = logits
        .iter()
        .zip(&expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 0.5,
        "{} full-logit max abs diff {max_abs_diff} exceeds 0.5",
        case.name
    );
    eprintln!(
        "{}: next_token_id={}, full_logit_max_abs_diff={max_abs_diff:.6}",
        case.name, reference.next_token_id
    );

    let mut max_anchor_diff = 0.0_f32;
    for anchor in &reference.top_logits {
        let diff = (logits[anchor.token_id] - anchor.logit).abs();
        max_anchor_diff = max_anchor_diff.max(diff);
    }
    assert!(
        max_anchor_diff < 0.5,
        "{} top-logit max abs diff {max_anchor_diff} exceeds 0.5",
        case.name
    );
}

fn run_blocking_thread_case(case: &Case) {
    let reference = fixture(case);
    let loader = Loader::open(&snapshot_dir(case)).expect("Loader::open MXFP checkpoint");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let handle = std::thread::spawn(move || {
        let seq = reference.input_ids.len() as i32;
        let input: Array = (reference.input_ids.as_slice(), &[1_i32, seq][..])
            .try_into()
            .expect("input ids array");
        let positions = build_position_ids(0, seq).expect("position ids");
        let mut cache = model
            .make_cache(1, seq + 1, Dtype::Bfloat16)
            .expect("make cache");
        let logits = model
            .forward_on(&input, &positions, Some(&[seq]), None, Some(&mut cache), ())
            .expect("blocking-thread MXFP forward");
        let logits = logits_vec(&logits);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(greedy_argmax(&logits), reference.next_token_id);
    });
    handle
        .join()
        .expect("blocking-thread MXFP forward panicked");
}

#[test]
fn qwen35_mxfp_real_checkpoints_match_mlx_lm_when_requested() {
    if std::env::var_os("IRONMLX_TEST_REAL_MXFP").as_deref() != Some("1".as_ref()) {
        eprintln!("IRONMLX_TEST_REAL_MXFP=1 not set; skipping real MXFP checkpoints");
        return;
    }
    for case in &CASES {
        run_case(case);
        run_blocking_thread_case(case);
    }
}
