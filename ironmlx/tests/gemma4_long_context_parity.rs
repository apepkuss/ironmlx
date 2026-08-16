//! Gemma4 12B long-context target-model parity diagnostics.
//!
//! These tests are ignored because they require a local 12B checkpoint and
//! generated fixtures from `tests/fixtures/gemma4_long_context/gen_reference.py`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use ironmlx::core::cache::TurboQuantKVBits;
use ironmlx::core::generate::build_position_ids;
use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::Scheduler;
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

fn long_context_lengths() -> Result<Vec<usize>> {
    std::env::var("GEMMA4_LONG_CONTEXT_TOKENS")
        .unwrap_or_else(|_| "8192,32768,65536".to_string())
        .split(',')
        .map(|raw| {
            let value = raw.trim().parse::<usize>()?;
            ensure!(value > 0, "Gemma4 long-context length must be positive");
            Ok(value)
        })
        .collect()
}

fn exact_length_prompt_ids(tokenizer: &Tokenizer, context_tokens: usize) -> Result<Vec<u32>> {
    let seed = tokenizer.encode(
        "Long-context exact speculative verification must preserve every target token. ",
        false,
    )?;
    ensure!(!seed.is_empty(), "Gemma4 long-context seed encoded empty");
    Ok(seed.into_iter().cycle().take(context_tokens).collect())
}

fn collect_base_tokens(
    model: &Gemma4Model,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Result<Vec<u32>> {
    let mut stream = GenerationStream::new_text_only(model, tokenizer, request)?;
    let mut tokens = Vec::new();
    while let Some(event) = stream.next_token()? {
        tokens.push(event.token);
        if event.finish_reason.is_some() {
            break;
        }
    }
    Ok(tokens)
}

fn collect_drafter_tokens(
    model: &Gemma4Model,
    drafter: &Gemma4AssistantModel,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
    draft_tokens: usize,
) -> Result<(Vec<u32>, ironmlx::core::speculative::MtpSpeculativeStats)> {
    let cfg = MtpSpeculativeConfig::new(draft_tokens, request.sampler)?;
    let mut stream = Gemma4DrafterGenerationStream::new(model, drafter, tokenizer, request, cfg)?;
    let mut tokens = Vec::new();
    while let Some(event) = stream.next_token()? {
        tokens.push(event.token);
        if event.finish_reason.is_some() {
            break;
        }
    }
    Ok((tokens, stream.stats()))
}

fn record_scheduler_events(
    request_ids: &[ironmlx::core::scheduler::RequestId],
    outputs: &mut [Vec<u32>],
    finished: &mut [bool],
    events: Vec<ironmlx::core::scheduler::StepEvent>,
) -> Result<()> {
    for event in events {
        let row = request_ids
            .iter()
            .position(|id| *id == event.id)
            .context("scheduler emitted an unknown request id")?;
        outputs[row].push(event.token);
        finished[row] = event.finish_reason.is_some();
    }
    Ok(())
}

fn collect_scheduled_k3v4_base_tokens(
    model: &Gemma4Model,
    requests: Vec<GenerateRequest>,
    b_max: usize,
) -> Result<Vec<Vec<u32>>> {
    ensure!(!requests.is_empty(), "scheduler test requires requests");
    ensure!(requests.len() <= b_max, "request count exceeds b_max");
    let effective_cap_max = requests
        .iter()
        .map(|request| request.prompt_ids.len() + request.max_new_tokens)
        .max()
        .context("scheduler test requires requests")?;
    let mut scheduler =
        Scheduler::<Gemma4Model>::new(b_max, effective_cap_max, model.model_meta())?;
    let mut request_ids = Vec::with_capacity(requests.len());
    for mut request in requests {
        request.kv_cache_turboquant_bits = Some(TurboQuantKVBits::K3V4);
        request_ids.push(scheduler.admit(request)?);
    }

    let mut outputs = vec![Vec::new(); request_ids.len()];
    let mut finished = vec![false; request_ids.len()];
    record_scheduler_events(
        &request_ids,
        &mut outputs,
        &mut finished,
        scheduler.prefill_admitted(model)?,
    )?;
    while !finished.iter().all(|done| *done) {
        let events = scheduler.step(model)?;
        ensure!(
            !events.is_empty(),
            "scheduler stopped before all requests finished"
        );
        record_scheduler_events(&request_ids, &mut outputs, &mut finished, events)?;
    }
    Ok(outputs)
}

fn collect_scheduled_k3v4_drafter_tokens(
    model: &Gemma4Model,
    drafter: &Gemma4AssistantModel,
    requests: Vec<GenerateRequest>,
    b_max: usize,
    draft_tokens: usize,
) -> Result<(
    Vec<Vec<u32>>,
    ironmlx::core::speculative::MtpSpeculativeStats,
)> {
    ensure!(!requests.is_empty(), "scheduler test requires requests");
    ensure!(requests.len() <= b_max, "request count exceeds b_max");
    let effective_cap_max = requests
        .iter()
        .map(|request| request.prompt_ids.len() + request.max_new_tokens)
        .max()
        .context("scheduler test requires requests")?;
    let mut scheduler =
        Scheduler::<Gemma4Model>::new(b_max, effective_cap_max, model.model_meta())?;
    let mut request_ids = Vec::with_capacity(requests.len());
    for mut request in requests {
        request.kv_cache_turboquant_bits = Some(TurboQuantKVBits::K3V4);
        request_ids.push(scheduler.admit(request)?);
    }

    let cfg = MtpSpeculativeConfig::new(draft_tokens, Sampler::greedy())?;
    let mut outputs = vec![Vec::new(); request_ids.len()];
    let mut finished = vec![false; request_ids.len()];
    record_scheduler_events(
        &request_ids,
        &mut outputs,
        &mut finished,
        scheduler.prefill_admitted_gemma4_drafter_batch(model, drafter, cfg)?,
    )?;
    while !finished.iter().all(|done| *done) {
        let events = scheduler.step_gemma4_drafter_batch(model, drafter)?;
        ensure!(
            !events.is_empty(),
            "scheduler stopped before all requests finished"
        );
        record_scheduler_events(&request_ids, &mut outputs, &mut finished, events)?;
    }
    let stats = scheduler
        .gemma4_drafter_stats()
        .context("scheduler produced no Gemma4 drafter stats")?;
    Ok((outputs, stats))
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

#[test]
#[ignore = "requires GEMMA4_LONG_CONTEXT_MODEL and GEMMA4_LONG_CONTEXT_DRAFTER"]
fn gemma4_drafter_long_context_tokens_match_ordinary_q1_exactly() -> Result<()> {
    let model_dir = PathBuf::from(
        std::env::var("GEMMA4_LONG_CONTEXT_MODEL")
            .context("GEMMA4_LONG_CONTEXT_MODEL must point to a real Gemma4 checkpoint")?,
    );
    let drafter_dir = PathBuf::from(
        std::env::var("GEMMA4_LONG_CONTEXT_DRAFTER")
            .context("GEMMA4_LONG_CONTEXT_DRAFTER must point to a matching assistant checkpoint")?,
    );
    let output_tokens = std::env::var("GEMMA4_LONG_CONTEXT_OUTPUT_TOKENS")
        .unwrap_or_else(|_| "64".to_string())
        .parse::<usize>()?;
    let draft_tokens = std::env::var("GEMMA4_LONG_CONTEXT_DRAFT_TOKENS")
        .unwrap_or_else(|_| "2".to_string())
        .parse::<usize>()?;
    ensure!(output_tokens > 1, "output token count must exceed one");
    ensure!(draft_tokens > 0, "draft token count must be positive");

    let loader = Loader::open(&model_dir).context("Loader::open")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let model = Gemma4Model::from_loader(&loader).context("Gemma4Model::from_loader")?;
    let drafter_loader =
        Loader::open_gemma4_drafter(&drafter_dir).context("Loader::open_gemma4_drafter")?;
    let drafter = Gemma4AssistantModel::from_loader(&drafter_loader)
        .context("Gemma4AssistantModel::from_loader")?;

    for context_tokens in long_context_lengths()? {
        let prompt_ids = exact_length_prompt_ids(&tokenizer, context_tokens)?;
        let request = make_text_request(prompt_ids, output_tokens, PREFILL_STEP_SIZE);
        let ordinary_tokens = collect_base_tokens(&model, &tokenizer, request.clone())?;
        mlx::clear_cache();
        let (drafter_tokens_out, stats) =
            collect_drafter_tokens(&model, &drafter, &tokenizer, request, draft_tokens)?;

        assert_eq!(
            drafter_tokens_out, ordinary_tokens,
            "Gemma4 drafter output diverged from ordinary Q1 at context {context_tokens}"
        );
        assert_eq!(ordinary_tokens.len(), output_tokens);
        assert!(stats.windows > 0, "drafter produced no verify windows");
        assert!(stats.drafted_tokens > 0, "drafter proposed no tokens");
        assert!(
            stats.accepted_draft_tokens <= stats.drafted_tokens,
            "accepted draft tokens exceeded proposed tokens"
        );
        eprintln!(
            "Gemma4 long-context exact parity: context={context_tokens} output={} \
             windows={} drafted={} accepted={} rollback={}",
            ordinary_tokens.len(),
            stats.windows,
            stats.drafted_tokens,
            stats.accepted_draft_tokens,
            stats.rollback_count
        );
        mlx::clear_cache();
    }
    Ok(())
}

#[test]
#[ignore = "requires GEMMA4_LONG_CONTEXT_MODEL and GEMMA4_LONG_CONTEXT_DRAFTER"]
fn gemma4_k3v4_long_context_scheduler_uses_multi_token_verify_exactly() -> Result<()> {
    let model_dir = PathBuf::from(
        std::env::var("GEMMA4_LONG_CONTEXT_MODEL")
            .context("GEMMA4_LONG_CONTEXT_MODEL must point to a real Gemma4 checkpoint")?,
    );
    let drafter_dir = PathBuf::from(
        std::env::var("GEMMA4_LONG_CONTEXT_DRAFTER")
            .context("GEMMA4_LONG_CONTEXT_DRAFTER must point to a matching assistant checkpoint")?,
    );
    let context_tokens = std::env::var("GEMMA4_K3V4_CONTEXT_TOKENS")
        .unwrap_or_else(|_| "8192".to_string())
        .parse::<usize>()?;
    let output_tokens = std::env::var("GEMMA4_LONG_CONTEXT_OUTPUT_TOKENS")
        .unwrap_or_else(|_| "64".to_string())
        .parse::<usize>()?;
    let draft_tokens = std::env::var("GEMMA4_LONG_CONTEXT_DRAFT_TOKENS")
        .unwrap_or_else(|_| "2".to_string())
        .parse::<usize>()?;
    let active_request_counts = std::env::var("GEMMA4_K3V4_ACTIVE_REQUESTS")
        .unwrap_or_else(|_| "1,2".to_string())
        .split(',')
        .map(|raw| {
            raw.trim()
                .parse::<usize>()
                .context("parsing active requests")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        context_tokens > 1024,
        "K3V4 regression requires >1024 tokens"
    );
    ensure!(output_tokens > 1, "output token count must exceed one");
    ensure!(draft_tokens > 1, "regression requires multi-token drafting");
    ensure!(
        !active_request_counts.is_empty()
            && active_request_counts
                .iter()
                .all(|count| (1..=2).contains(count)),
        "active request counts must contain only 1 or 2"
    );

    let loader = Loader::open(&model_dir).context("Loader::open")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;
    let model = Gemma4Model::from_loader(&loader).context("Gemma4Model::from_loader")?;
    let drafter_loader =
        Loader::open_gemma4_drafter(&drafter_dir).context("Loader::open_gemma4_drafter")?;
    let drafter = Gemma4AssistantModel::from_loader(&drafter_loader)
        .context("Gemma4AssistantModel::from_loader")?;
    let prompt_ids = exact_length_prompt_ids(&tokenizer, context_tokens)?;
    let request = make_text_request(prompt_ids, output_tokens, PREFILL_STEP_SIZE);

    for active_requests in active_request_counts {
        let requests = vec![request.clone(); active_requests];
        let ordinary_tokens = collect_scheduled_k3v4_base_tokens(&model, requests.clone(), 4)?;
        mlx::clear_cache();
        let (scheduled_tokens, stats) =
            collect_scheduled_k3v4_drafter_tokens(&model, &drafter, requests, 4, draft_tokens)?;
        for (row, tokens) in scheduled_tokens.iter().enumerate() {
            assert_eq!(
                tokens, &ordinary_tokens[row],
                "K3V4 scheduler row {row} diverged at B={active_requests}"
            );
        }
        assert!(
            ordinary_tokens
                .iter()
                .all(|tokens| tokens.len() == output_tokens),
            "ordinary K3V4 scheduler emitted an unexpected token count"
        );
        assert!(
            stats.draft_attempts_by_position.get(1).copied().unwrap_or(0) > 0,
            "K3V4 long-context scheduler never attempted the second draft position at B={active_requests}"
        );
        eprintln!(
            "Gemma4 K3V4 scheduler exact parity: context={context_tokens} B={active_requests} \
             output={} attempts={:?} accepted={}",
            ordinary_tokens[0].len(),
            stats.draft_attempts_by_position,
            stats.accepted_draft_tokens
        );
        mlx::clear_cache();
    }
    Ok(())
}
