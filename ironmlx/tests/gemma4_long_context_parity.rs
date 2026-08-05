//! Gemma4 12B long-context target-model parity diagnostics.
//!
//! These tests are ignored because they require a local 12B checkpoint and
//! generated fixtures from `tests/fixtures/gemma4_long_context/gen_reference.py`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use ironmlx::core::generate::build_position_ids;
use ironmlx::core::generate::GenerateRequest;
use ironmlx::core::sampler::Sampler;
use ironmlx::core::speculative::MtpSpeculativeConfig;
use ironmlx::core::tokenizer::Tokenizer;
use ironmlx::core::{Loader, Model};
use ironmlx::models::gemma4::{Gemma4AssistantModel, Gemma4DrafterGenerationStream, Gemma4Model};
use ironmlx::nn::LayerCache;
use mlx::{Array, Dtype, StreamOrDevice};
use serde::Deserialize;

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gemma4_long_context"
);
const PREFILL_STEP_SIZE: usize = 2048;
const LONG_CONTEXT_LOGIT_MAX_ABS_LIMIT: f32 = 5.5;
const LONG_CONTEXT_CASES: &[&str] = &["case_18000", "case_19900", "case_20000", "case_24000"];

#[derive(Debug, Deserialize)]
struct ExpectedCase {
    after_append_greedy_token: usize,
    after_append_token: i32,
}

#[derive(Debug, Deserialize)]
struct ExpectedDrafterRound {
    prefill_step_size: usize,
    draft_tokens_budget: usize,
    kv_offset: usize,
    draft_position: i32,
    first_token: u32,
    draft_tokens: Vec<u32>,
    draft_step_top_k: Vec<Vec<TopKRecord>>,
    verified_tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct TopKRecord {
    token: u32,
}

fn default_snapshot_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("GEMMA4_LONG_CONTEXT_MODEL") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
        eprintln!(
            "GEMMA4_LONG_CONTEXT_MODEL={} does not exist",
            path.display()
        );
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
}

fn default_drafter_snapshot_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("GEMMA4_LONG_CONTEXT_DRAFTER") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
        eprintln!(
            "GEMMA4_LONG_CONTEXT_DRAFTER={} does not exist",
            path.display()
        );
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".ironmlx/models/models--mlx-community--gemma-4-12B-it-assistant-4bit/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(FIXTURE_DIR).join(name)
}

fn fixture_exists(case: &str) -> bool {
    fixture_path(&format!("{case}_input_ids.npy")).exists()
        && fixture_path(&format!("{case}_expected.json")).exists()
        && fixture_path(&format!("{case}_expected_after_append_logits.npy")).exists()
}

fn drafter_fixture_exists(case: &str) -> bool {
    fixture_path(&format!("{case}_input_ids.npy")).exists()
        && fixture_path(&format!("{case}_expected_drafter_round.json")).exists()
}

fn load_ids(case: &str) -> Result<Vec<i32>> {
    let path = fixture_path(&format!("{case}_input_ids.npy"));
    let arr = mlx::io::load_npy(path.to_str().context("utf8 fixture path")?)
        .with_context(|| format!("load {}", path.display()))?;
    let arr = mlx::ops::cast::astype(&arr, Dtype::Int32)?;
    Ok(arr.to_vec::<i32>()?)
}

fn load_expected(case: &str) -> Result<ExpectedCase> {
    let path = fixture_path(&format!("{case}_expected.json"));
    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_str(&raw)?)
}

fn load_expected_drafter(case: &str) -> Result<ExpectedDrafterRound> {
    let path = fixture_path(&format!("{case}_expected_drafter_round.json"));
    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_str(&raw)?)
}

fn load_npy(case: &str, suffix: &str) -> Result<Array> {
    let path = fixture_path(&format!("{case}_{suffix}.npy"));
    mlx::io::load_npy(path.to_str().context("utf8 fixture path")?)
        .with_context(|| format!("load {}", path.display()))
}

fn ids_array(ids: &[i32]) -> Result<Array> {
    Ok((ids, &[1_i32, ids.len() as i32][..]).try_into()?)
}

fn flatten_logits(logits: &Array) -> Result<Array> {
    let flat = mlx::ops::cast::astype(logits, Dtype::Float32)?;
    let vocab = *flat.shape().as_slice().last().context("logits rank")?;
    Ok(flat.reshape((vocab,))?)
}

fn to_f32_vec(arr: &Array) -> Result<Vec<f32>> {
    Ok(mlx::ops::cast::astype(arr, Dtype::Float32)?.to_vec::<f32>()?)
}

fn max_abs_diff(a: &Array, b: &Array) -> Result<f32> {
    let av = to_f32_vec(a)?;
    let bv = to_f32_vec(b)?;
    ensure!(av.len() == bv.len(), "element count mismatch");
    Ok(av
        .iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max))
}

fn greedy_argmax(logits: &Array) -> Result<usize> {
    let flat = flatten_logits(logits)?;
    let values = to_f32_vec(&flat)?;
    let mut best = None;
    for (idx, &value) in values.iter().enumerate() {
        if best
            .map(|(_, best_value)| value > best_value)
            .unwrap_or(true)
        {
            best = Some((idx, value));
        }
    }
    best.map(|(idx, _)| idx).context("empty logits")
}

fn top_k(logits: &Array, k: usize) -> Result<Vec<usize>> {
    let flat = flatten_logits(logits)?;
    let values = to_f32_vec(&flat)?;
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    Ok(idx)
}

fn compare_logits(label: &str, got: &Array, expected: &Array, max_abs_limit: f32) -> Result<()> {
    let got_flat = flatten_logits(got)?;
    let expected_flat = flatten_logits(expected)?;
    ensure!(
        got_flat.shape().as_slice() == expected_flat.shape().as_slice(),
        "{label}: logits shape mismatch"
    );
    let got_argmax = greedy_argmax(&got_flat)?;
    let expected_argmax = greedy_argmax(&expected_flat)?;
    let got_top5 = top_k(&got_flat, 5)?;
    let expected_top5 = top_k(&expected_flat, 5)?;
    let err = max_abs_diff(&got_flat, &expected_flat)?;
    eprintln!(
        "{label}: argmax got={got_argmax} expected={expected_argmax} \
         max_abs={err:.4} got_top5={got_top5:?} expected_top5={expected_top5:?}"
    );
    assert_eq!(
        got_argmax, expected_argmax,
        "{label}: greedy argmax mismatch"
    );
    let got_set: BTreeSet<usize> = got_top5.iter().copied().collect();
    let expected_set: BTreeSet<usize> = expected_top5.iter().copied().collect();
    if got_set != expected_set {
        eprintln!("{label}: advisory top-5 token set mismatch");
    }
    assert!(
        err < max_abs_limit,
        "{label}: max_abs {err} >= {max_abs_limit}"
    );
    Ok(())
}

fn forward_chunk(
    model: &Gemma4Model,
    ids: &[i32],
    absolute_start: i32,
    cache: &mut [LayerCache],
) -> Result<Array> {
    let input_ids = ids_array(ids)?;
    let pos = build_position_ids(absolute_start, ids.len() as i32)?;
    let logits = Model::forward_on(
        model,
        &input_ids,
        &pos,
        None,
        None,
        Some(cache),
        StreamOrDevice::default(),
    )?;
    mlx::transforms::eval(&[&logits])?;
    Ok(logits)
}

fn after_append_logits(
    model: &Gemma4Model,
    prompt_ids: &[i32],
    append_token: i32,
) -> Result<Array> {
    let cap = prompt_ids.len() as i32 + 8;
    let mut cache = Model::make_cache(model, 1, cap, model.cache_dtype())?;
    let mut processed = 0usize;
    while prompt_ids.len() - processed > 1 {
        let remaining = (prompt_ids.len() - processed) - 1;
        let n = PREFILL_STEP_SIZE.min(remaining);
        let _ = forward_chunk(
            model,
            &prompt_ids[processed..processed + n],
            processed as i32,
            &mut cache,
        )?;
        processed += n;
    }
    let _prompt_logits = forward_chunk(
        model,
        &prompt_ids[processed..],
        processed as i32,
        &mut cache,
    )?;
    forward_chunk(model, &[append_token], prompt_ids.len() as i32, &mut cache)
}

fn check_after_append_case(model: &Gemma4Model, case: &str) -> Result<()> {
    let prompt_ids = load_ids(case)?;
    let expected = load_expected(case)?;
    let logits = after_append_logits(model, &prompt_ids, expected.after_append_token)?;
    let expected_logits = load_npy(case, "expected_after_append_logits")?;
    let got = greedy_argmax(&logits)?;

    assert_eq!(
        got, expected.after_append_greedy_token,
        "{case}: after-append greedy token mismatch"
    );
    compare_logits(
        case,
        &logits,
        &expected_logits,
        LONG_CONTEXT_LOGIT_MAX_ABS_LIMIT,
    )?;
    Ok(())
}

fn make_text_request(
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    prefill_chunk_size: usize,
) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: Vec::new(),
        prefill_chunk_size,
        decode_cadence_mid_chunk_cap: 256,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 3,
        image_token_id: 0,
        constraint: None,
    }
}

fn check_drafter_first_round_case(
    model: &Gemma4Model,
    drafter: &Gemma4AssistantModel,
    tokenizer: &Tokenizer,
    case: &str,
) -> Result<()> {
    let prompt_ids: Vec<u32> = load_ids(case)?.into_iter().map(|id| id as u32).collect();
    let expected = load_expected_drafter(case)?;
    let request = make_text_request(
        prompt_ids,
        expected.draft_tokens_budget + 1,
        expected.prefill_step_size,
    );
    let cfg = MtpSpeculativeConfig::new(expected.draft_tokens_budget, request.sampler)?;
    let mut stream = Gemma4DrafterGenerationStream::new(model, drafter, tokenizer, request, cfg)?;
    stream.set_trace_window_limit(1);

    let first = stream
        .next_token()?
        .context("Gemma4 drafter stream produced no first token")?;
    assert_eq!(
        first.token, expected.first_token,
        "{case}: first target token mismatch"
    );
    let trace = stream
        .trace_windows()
        .first()
        .context("Gemma4 drafter trace did not record first window")?;
    assert_eq!(
        trace.history_len,
        expected.kv_offset + 1,
        "{case}: trace history length mismatch"
    );
    assert_eq!(
        trace.verify_start_pos, expected.kv_offset as i32,
        "{case}: verify start position mismatch"
    );
    assert_eq!(
        expected.draft_position,
        (expected.kv_offset as i32 - 1).max(0),
        "{case}: fixture draft position mismatch"
    );
    for (idx, (&actual, &reference)) in trace
        .draft_tokens
        .iter()
        .zip(expected.draft_tokens.iter())
        .enumerate()
    {
        if actual == reference {
            continue;
        }
        let candidates: Vec<u32> = expected
            .draft_step_top_k
            .get(idx)
            .map(|records| records.iter().map(|record| record.token).collect())
            .unwrap_or_default();
        assert!(
            candidates.contains(&actual),
            "{case}: draft token {idx} mismatch and actual token {actual} \
             is outside reference top-k {candidates:?}; reference token {reference}"
        );
    }
    let comparable_draft_prefix = trace
        .draft_tokens
        .iter()
        .zip(expected.draft_tokens.iter())
        .take_while(|(actual, reference)| actual == reference)
        .count();
    let comparable_verified = comparable_draft_prefix + 1;
    assert!(
        trace.verified_tokens.len() >= comparable_verified
            && expected.verified_tokens.len() >= comparable_verified,
        "{case}: not enough verifier tokens to compare shared prefix"
    );
    assert_eq!(
        &trace.verified_tokens[..comparable_verified],
        &expected.verified_tokens[..comparable_verified],
        "{case}: verified tokens mismatch within shared draft prefix"
    );
    Ok(())
}

#[test]
#[ignore = "requires GEMMA4_LONG_CONTEXT_MODEL and generated Gemma4 long-context fixtures"]
fn gemma4_12b_long_context_20000_after_append_matches_reference() -> Result<()> {
    let case = "case_20000";
    if !fixture_exists(case) {
        eprintln!("skip: generate Gemma4 long-context fixtures first");
        return Ok(());
    }
    let Some(dir) = default_snapshot_dir() else {
        eprintln!("skip: no Gemma4 12B checkpoint");
        return Ok(());
    };

    let loader = Loader::open(&dir).context("Loader::open")?;
    let model = Gemma4Model::from_loader(&loader).context("Gemma4Model::from_loader")?;
    check_after_append_case(&model, case)
}

#[test]
#[ignore = "requires GEMMA4_LONG_CONTEXT_MODEL and generated Gemma4 long-context fixtures"]
fn gemma4_12b_long_context_after_append_cases_match_reference() -> Result<()> {
    let cases: Vec<&str> = LONG_CONTEXT_CASES
        .iter()
        .copied()
        .filter(|case| fixture_exists(case))
        .collect();
    if cases.is_empty() {
        eprintln!("skip: generate Gemma4 long-context fixtures first");
        return Ok(());
    }
    let Some(dir) = default_snapshot_dir() else {
        eprintln!("skip: no Gemma4 12B checkpoint");
        return Ok(());
    };

    let loader = Loader::open(&dir).context("Loader::open")?;
    let model = Gemma4Model::from_loader(&loader).context("Gemma4Model::from_loader")?;
    for case in cases {
        check_after_append_case(&model, case)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires GEMMA4_LONG_CONTEXT_MODEL, GEMMA4_LONG_CONTEXT_DRAFTER, and generated drafter fixtures"]
fn gemma4_12b_drafter_first_round_matches_reference() -> Result<()> {
    let cases: Vec<&str> = LONG_CONTEXT_CASES
        .iter()
        .copied()
        .filter(|case| drafter_fixture_exists(case))
        .collect();
    if cases.is_empty() {
        eprintln!("skip: generate Gemma4 drafter round fixtures first");
        return Ok(());
    }
    let Some(model_dir) = default_snapshot_dir() else {
        eprintln!("skip: no Gemma4 12B checkpoint");
        return Ok(());
    };
    let Some(drafter_dir) = default_drafter_snapshot_dir() else {
        eprintln!("skip: no Gemma4 12B drafter checkpoint");
        return Ok(());
    };

    let loader = Loader::open(&model_dir).context("Loader::open")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let model = Gemma4Model::from_loader(&loader).context("Gemma4Model::from_loader")?;
    let drafter_loader =
        Loader::open_gemma4_drafter(&drafter_dir).context("Loader::open_gemma4_drafter")?;
    let drafter = Gemma4AssistantModel::from_loader(&drafter_loader)
        .context("Gemma4AssistantModel::from_loader")?;
    for case in cases {
        check_drafter_first_round_case(&model, &drafter, &tokenizer, case)?;
    }
    Ok(())
}
