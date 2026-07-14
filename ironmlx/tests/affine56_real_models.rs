//! Real-checkpoint correctness gates for affine 5-bit and 6-bit models.
//!
//! Set `IRONMLX_TEST_REAL_AFFINE56=1` to run all six pinned checkpoints.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use ironmlx::core::{generate::build_position_ids, Model, QuantMode, Tokenizer};
use ironmlx::models::{Gemma4Model, Qwen35Model};
use ironmlx::Loader;
use mlx::{Array, Dtype, StreamOrDevice};
use serde::Deserialize;

const FIXTURE_DIR: &str = "tests/fixtures/affine56";
const PROMPT: &str = "What is 2+2?";
const PROMPT_SHA256: &str = "52cb6b5e4a038af1756708f98afb718a08c75b87b2f03dbee4dd9c8139c15c5e";
const GREEDY_STEPS: usize = 4;
const REFERENCE_MLX: &str = "0.32.1.dev20260710+938006e4a";
const REFERENCE_MLX_LM: &str = "0.31.3";
const REFERENCE_TRANSFORMERS: &str = "5.7.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Architecture {
    Gemma4,
    Qwen35,
}

impl Architecture {
    fn fixture_name(self) -> &'static str {
        match self {
            Self::Gemma4 => "gemma4",
            Self::Qwen35 => "qwen3_5",
        }
    }
}

#[derive(Deserialize)]
struct QuantizationFixture {
    mode: String,
    bits: i32,
    group_size: i32,
}

#[derive(Deserialize)]
struct LogitAnchor {
    token_id: usize,
    logit: f32,
}

#[derive(Deserialize)]
struct ReferenceRuntime {
    mlx: String,
    mlx_lm: String,
    transformers: String,
}

#[derive(Deserialize)]
struct ReferenceFixture {
    model_id: String,
    revision: String,
    architecture: String,
    quantization: QuantizationFixture,
    prompt: String,
    prompt_sha256: String,
    input_ids: Vec<u32>,
    vocab_size: usize,
    next_token_id: usize,
    greedy_token_ids: Vec<u32>,
    logits_file: String,
    top_logits: Vec<LogitAnchor>,
    reference: ReferenceRuntime,
}

#[derive(Debug)]
struct Case {
    name: &'static str,
    model_id: &'static str,
    revision: &'static str,
    architecture: Architecture,
    bits: i32,
    fixture: Option<&'static str>,
}

const CASES: [Case; 6] = [
    Case {
        name: "gemma4-4bit",
        model_id: "mlx-community/gemma-4-e2b-it-4bit",
        revision: "238767527555cb75a05732a84dff5d6ba0dd6809",
        architecture: Architecture::Gemma4,
        bits: 4,
        fixture: Some("gemma4-4bit"),
    },
    Case {
        name: "gemma4-5bit",
        model_id: "mlx-community/gemma-4-e2b-it-5bit",
        revision: "dc565aea8c49afb542497310a2d86bf1fd91391f",
        architecture: Architecture::Gemma4,
        bits: 5,
        fixture: Some("gemma4-5bit"),
    },
    Case {
        name: "gemma4-6bit",
        model_id: "mlx-community/gemma-4-e2b-it-6bit",
        revision: "ebd7756d4e55627e11ae043af9cad8ed6465a2e2",
        architecture: Architecture::Gemma4,
        bits: 6,
        fixture: Some("gemma4-6bit"),
    },
    Case {
        name: "qwen35-4bit",
        model_id: "mlx-community/Qwen3.5-2B-4bit",
        revision: "674aaa7240b91e8012fcad5d791b7dfe5ba90207",
        architecture: Architecture::Qwen35,
        bits: 4,
        fixture: Some("qwen35-4bit"),
    },
    Case {
        name: "qwen35-5bit",
        model_id: "mlx-community/Qwen3.5-2B-5bit",
        revision: "0934527791eb8008cd84b66550b8ab3eefd15b85",
        architecture: Architecture::Qwen35,
        bits: 5,
        fixture: Some("qwen35-5bit"),
    },
    Case {
        name: "qwen35-6bit",
        model_id: "mlx-community/Qwen3.5-2B-6bit",
        revision: "ba2bcf03dd5b502646de7e32b003cf538f2ca4d6",
        architecture: Architecture::Qwen35,
        bits: 6,
        fixture: Some("qwen35-6bit"),
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

fn load_fixture(case: &Case) -> Option<ReferenceFixture> {
    case.fixture.map(|name| {
        let path = Path::new(FIXTURE_DIR).join(format!("{name}.json"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
    })
}

fn greedy_argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .expect("non-empty logits")
}

fn logits_vec(logits: &Array) -> Vec<f32> {
    mlx::ops::cast::astype(logits, Dtype::Float32)
        .expect("cast logits")
        .to_vec::<f32>()
        .expect("read logits")
}

fn assert_finite_logits(case_name: &str, phase: &str, logits: &[f32]) {
    let non_finite: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| !value.is_finite())
        .take(8)
        .collect();
    assert!(
        non_finite.is_empty(),
        "{case_name} {phase} contains non-finite logits; first values: {non_finite:?}"
    );
}

fn generate_greedy(case_name: &str, model: &dyn Model, input_ids: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let prompt_len = input_ids.len() as i32;
    let mut cache = model
        .make_cache(1, prompt_len + GREEDY_STEPS as i32 + 1, model.cache_dtype())
        .expect("make cache");
    let input: Array = (input_ids, &[1_i32, prompt_len][..])
        .try_into()
        .expect("prompt input");
    let positions = build_position_ids(0, prompt_len).expect("prefill positions");
    let logits = model
        .forward_on(
            &input,
            &positions,
            Some(&[prompt_len]),
            None,
            Some(&mut cache),
            StreamOrDevice::default(),
        )
        .expect("prefill forward");
    let first_logits = logits_vec(&logits);
    assert_finite_logits(case_name, "prefill", &first_logits);

    let mut current_logits = first_logits.clone();
    let mut generated = Vec::with_capacity(GREEDY_STEPS);
    for step in 0..GREEDY_STEPS {
        let token = greedy_argmax(&current_logits) as u32;
        generated.push(token);
        if step + 1 == GREEDY_STEPS {
            break;
        }
        let decode_input: Array = (&[token][..], &[1_i32, 1][..])
            .try_into()
            .expect("decode input");
        let decode_positions =
            build_position_ids(prompt_len + step as i32, 1).expect("decode positions");
        let logits = model
            .forward_on(
                &decode_input,
                &decode_positions,
                Some(&[1]),
                None,
                Some(&mut cache),
                StreamOrDevice::default(),
            )
            .expect("decode forward");
        current_logits = logits_vec(&logits);
        assert_finite_logits(case_name, "decode", &current_logits);
    }
    (first_logits, generated)
}

fn with_model<R>(case: &Case, loader: &Loader, body: impl FnOnce(&dyn Model) -> R) -> R {
    let result = match case.architecture {
        Architecture::Gemma4 => {
            let model = Gemma4Model::from_loader(loader).expect("Gemma4Model::from_loader");
            body(&model)
        }
        Architecture::Qwen35 => {
            let model = Qwen35Model::from_loader(loader).expect("Qwen35Model::from_loader");
            body(&model)
        }
    };
    mlx::clear_cache();
    result
}

fn assert_reference(case: &Case, reference: &ReferenceFixture, logits: &[f32], generated: &[u32]) {
    assert_eq!(reference.model_id, case.model_id);
    assert_eq!(reference.revision, case.revision);
    assert_eq!(reference.architecture, case.architecture.fixture_name());
    assert_eq!(reference.quantization.mode, "affine");
    assert_eq!(reference.quantization.bits, case.bits);
    assert_eq!(reference.quantization.group_size, 64);
    assert_eq!(reference.prompt, PROMPT);
    assert_eq!(reference.prompt_sha256, PROMPT_SHA256);
    assert_eq!(reference.top_logits.len(), 64);
    assert_eq!(reference.greedy_token_ids.len(), GREEDY_STEPS);
    assert_eq!(reference.reference.mlx, REFERENCE_MLX);
    assert_eq!(reference.reference.mlx_lm, REFERENCE_MLX_LM);
    assert_eq!(reference.reference.transformers, REFERENCE_TRANSFORMERS);
    assert_eq!(logits.len(), reference.vocab_size);
    assert_eq!(greedy_argmax(logits), reference.next_token_id);
    assert_eq!(generated, reference.greedy_token_ids);

    let expected_path = Path::new(FIXTURE_DIR).join(&reference.logits_file);
    let expected = mlx::io::load_npy(
        expected_path
            .to_str()
            .expect("reference logits path must be UTF-8"),
    )
    .unwrap_or_else(|e| panic!("load {}: {e}", expected_path.display()));
    let expected = logits_vec(&expected);
    let signed_diffs: Vec<f32> = logits
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| actual - expected)
        .collect();
    let mean_signed_diff =
        signed_diffs.iter().map(|value| *value as f64).sum::<f64>() / signed_diffs.len() as f64;
    let mut centered_absolute_diffs: Vec<f32> = signed_diffs
        .iter()
        .map(|value| (*value as f64 - mean_signed_diff).abs() as f32)
        .collect();
    let mut absolute_diffs: Vec<f32> = signed_diffs.iter().map(|value| value.abs()).collect();
    let max_abs_diff = absolute_diffs.iter().copied().fold(0.0_f32, f32::max);
    let mean_abs_diff = absolute_diffs
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>()
        / absolute_diffs.len() as f64;
    let rmse = (absolute_diffs
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        / absolute_diffs.len() as f64)
        .sqrt();
    let centered_max_abs_diff = centered_absolute_diffs
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let centered_rmse = (signed_diffs
        .iter()
        .map(|value| (*value as f64 - mean_signed_diff).powi(2))
        .sum::<f64>()
        / signed_diffs.len() as f64)
        .sqrt();
    absolute_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    centered_absolute_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p99_abs_diff = absolute_diffs[(absolute_diffs.len() * 99) / 100];
    let centered_p99_abs_diff = centered_absolute_diffs[(centered_absolute_diffs.len() * 99) / 100];
    let max_anchor_diff = reference
        .top_logits
        .iter()
        .map(|anchor| (logits[anchor.token_id] - anchor.logit).abs())
        .fold(0.0_f32, f32::max);
    let mut actual_ranking: Vec<usize> = (0..logits.len()).collect();
    actual_ranking.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Equal));
    let top_64_overlap = actual_ranking[..64]
        .iter()
        .filter(|token_id| {
            reference
                .top_logits
                .iter()
                .any(|anchor| anchor.token_id == **token_id)
        })
        .count();
    eprintln!(
        "{}: next_token={}, max_abs={max_abs_diff:.6}, mean_abs={mean_abs_diff:.6}, mean_signed={mean_signed_diff:.6}, rmse={rmse:.6}, p99_abs={p99_abs_diff:.6}, centered_max_abs={centered_max_abs_diff:.6}, centered_rmse={centered_rmse:.6}, centered_p99_abs={centered_p99_abs_diff:.6}, anchor_max_abs={max_anchor_diff:.6}, top64_overlap={top_64_overlap}",
        case.name, reference.next_token_id
    );
    assert!(
        max_abs_diff < 1.0,
        "{} full-logit max abs diff {max_abs_diff} exceeds 1.0",
        case.name
    );
    assert!(
        max_anchor_diff < 1.0,
        "{} top-logit max abs diff {max_anchor_diff} exceeds 1.0",
        case.name
    );
    assert!(
        centered_max_abs_diff < 0.55,
        "{} centered max abs diff {centered_max_abs_diff} exceeds 0.55",
        case.name
    );
    assert!(
        centered_rmse < 0.10,
        "{} centered RMSE {centered_rmse} exceeds 0.10",
        case.name
    );
    assert!(
        centered_p99_abs_diff < 0.25,
        "{} centered P99 abs diff {centered_p99_abs_diff} exceeds 0.25",
        case.name
    );
    assert!(
        top_64_overlap >= 60,
        "{} top-64 overlap {top_64_overlap} is below 60",
        case.name
    );
}

fn run_case(case: &Case) {
    let model_dir = snapshot_dir(case);
    assert!(
        model_dir.is_dir(),
        "missing snapshot {}",
        model_dir.display()
    );
    let loader = Loader::open(&model_dir).expect("Loader::open affine checkpoint");
    let quant = loader.quant_meta().expect("global affine quant metadata");
    assert_eq!(quant.mode, QuantMode::Affine);
    assert_eq!(quant.bits, case.bits);
    assert_eq!(quant.group_size, 64);

    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let input_ids = tokenizer.encode(PROMPT, false).expect("tokenizer.encode");
    let reference = load_fixture(case);
    if let Some(reference) = &reference {
        assert_eq!(input_ids, reference.input_ids);
    }
    with_model(case, &loader, |model| {
        let (first_logits, generated) = generate_greedy(case.name, model, &input_ids);
        let (_, repeated) = generate_greedy(case.name, model, &input_ids);
        assert_eq!(
            generated, repeated,
            "{} generation is not deterministic",
            case.name
        );
        if let Some(reference) = &reference {
            assert_reference(case, reference, &first_logits, &generated);
        }
    });
}

fn run_blocking_thread_case(case_index: usize) {
    let handle = std::thread::spawn(move || {
        let case = &CASES[case_index];
        let loader = Loader::open(&snapshot_dir(case)).expect("blocking Loader::open");
        let tokenizer = Tokenizer::from_loader(&loader).expect("blocking tokenizer");
        let input_ids = tokenizer.encode(PROMPT, false).expect("blocking encode");
        with_model(case, &loader, |model| {
            let (logits, generated) = generate_greedy(case.name, model, &input_ids);
            assert!(logits.iter().all(|value| value.is_finite()));
            assert_eq!(generated.len(), GREEDY_STEPS);
        });
    });
    handle.join().expect("blocking affine forward panicked");
}

#[test]
fn affine_5bit_and_6bit_real_checkpoints_match_mlx_lm_when_requested() {
    if std::env::var_os("IRONMLX_TEST_REAL_AFFINE56").as_deref() != Some("1".as_ref()) {
        eprintln!("IRONMLX_TEST_REAL_AFFINE56=1 not set; skipping real affine checkpoints");
        return;
    }
    let case_filter = std::env::var("IRONMLX_TEST_REAL_AFFINE56_CASE").ok();
    for case in &CASES {
        if case_filter
            .as_deref()
            .is_some_and(|filter| filter != case.name)
        {
            continue;
        }
        run_case(case);
    }
    for (index, case) in CASES.iter().enumerate() {
        if case.fixture.is_some()
            && !case_filter
                .as_deref()
                .is_some_and(|filter| filter != case.name)
        {
            run_blocking_thread_case(index);
        }
    }
}
