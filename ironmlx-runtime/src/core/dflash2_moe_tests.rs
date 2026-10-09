//! Real-checkpoint Qwen3.6 MoE DFlash2 generation tests.
//!
//! `QWEN36_MOE_DFLASH2_TARGET` selects one target bit width and
//! `DFLASH2_MODEL` the BF16 incoai draft checkpoint.
//! `QWEN36_MOE_DFLASH2_DRAFT_BITS` selects how that checkpoint is loaded:
//! `4` (default, the production setting) quantizes it at load to affine
//! 4-bit, `0` keeps BF16. The fixture reads the precision back from the
//! loaded projections and fails if it differs. Every comparison is against
//! the ordinary decoding of the same target checkpoint.

use super::*;
use crate::core::generate::GenerationStream;
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::{loader::Loader, model::Model, tokenizer::Tokenizer};
use ironmlx_lm::models::{dflash2::DFlash2DraftModel, Qwen35MoeModel};
use serial_test::serial;

const PROMPTS: [&str; 3] = [
    "Write a Python function that checks whether a number is prime, then explain it briefly.",
    "List three differences between TCP and UDP and give one use case for each.",
    "Translate into French: The library opens at nine and closes at six on weekdays.",
];

struct Fixture {
    target: Qwen35MoeModel,
    draft: DFlash2DraftModel,
    /// `q4` or `bf16`, as read back from the loaded draft projections.
    draft_label: &'static str,
    tokenizer: Tokenizer,
}

fn fixture() -> Option<Fixture> {
    let target_dir = std::path::PathBuf::from(std::env::var("QWEN36_MOE_DFLASH2_TARGET").ok()?);
    let draft_dir = std::path::PathBuf::from(std::env::var("DFLASH2_MODEL").ok()?);
    let target_loader = Loader::open(&target_dir).expect("open Qwen3.6 MoE target");
    let tokenizer = Tokenizer::from_loader(&target_loader).expect("tokenizer");
    let target = Qwen35MoeModel::from_loader(&target_loader).expect("load MoE target");
    assert!(
        target.dflash2_target_bits().is_some(),
        "target is not a qualified Qwen3.6 MoE DFlash2 recipe"
    );
    drop(target_loader);
    let draft_bits: i32 = std::env::var("QWEN36_MOE_DFLASH2_DRAFT_BITS")
        .map(|value| {
            value
                .parse()
                .expect("QWEN36_MOE_DFLASH2_DRAFT_BITS must be 0 or 4")
        })
        .unwrap_or(4);
    assert!(
        matches!(draft_bits, 0 | 4),
        "QWEN36_MOE_DFLASH2_DRAFT_BITS must be 0 or 4, got {draft_bits}"
    );
    let draft_loader = Loader::open_dflash2(&draft_dir).expect("open DFlash2 draft");
    let draft = DFlash2DraftModel::from_loader(
        &draft_loader,
        target.dflash2_target_spec(),
        (draft_bits != 0).then_some(draft_bits),
    )
    .expect("load DFlash2 draft");
    let precision = draft.projection_precision().expect("draft precision");
    assert_eq!(
        precision.bits,
        (draft_bits != 0).then_some(draft_bits),
        "loaded draft projections do not carry the requested precision"
    );
    let draft_label = if precision.bits == Some(4) {
        "q4"
    } else {
        "bf16"
    };
    eprintln!(
        "[qwen36-moe-dflash2 fixture] target={} target_bits={:?} draft={} requested_draft_bits={draft_bits} \
         loaded_projection_bits={:?} projections={} draft_label={draft_label}",
        target_dir.display(),
        target.dflash2_target_bits(),
        draft_dir.display(),
        precision.bits,
        precision.projections,
    );
    Some(Fixture {
        target,
        draft,
        draft_label,
        tokenizer,
    })
}

fn request(
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    stops: Vec<u32>,
    sampler: Sampler,
) -> GenerateRequest {
    GenerateRequest {
        priority: Default::default(),
        prompt_ids,
        max_new_tokens,
        sampler,
        stop_token_ids: stops,
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 1,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248_056,
        constraint: None,
    }
}

type Event = (u32, Option<String>);

fn drain(mut next: impl FnMut() -> Result<Option<GenerateEvent>>) -> Vec<Event> {
    let mut events = Vec::new();
    while let Some(event) = next().expect("generation step") {
        let finished = event.finish_reason.is_some();
        events.push((
            event.token,
            event.finish_reason.map(|reason| format!("{reason:?}")),
        ));
        if finished {
            break;
        }
    }
    events
}

/// Greedy DFlash2 generation must emit the same token IDs and finish reason
/// as ordinary decoding of the same checkpoint across multi-window runs, an
/// output-length tail that is not a multiple of the block, a budget smaller
/// than one block, and a stop token reached inside a draft window.
#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_greedy_generation_matches_ordinary() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let bits = fixture.target.dflash2_target_bits().expect("bits");
    let draft = fixture.draft_label;
    let mut long_prompt = String::new();
    while fixture
        .tokenizer
        .encode(&long_prompt, false)
        .expect("encode")
        .len()
        < 1_000
    {
        long_prompt.push_str(PROMPTS[0]);
        long_prompt.push(' ');
        long_prompt.push_str(PROMPTS[1]);
        long_prompt.push(' ');
    }
    let mut prompts = PROMPTS.map(str::to_owned).to_vec();
    prompts.push(long_prompt);

    let mut totals = (0_usize, 0_usize, 0_usize, 0_usize, 0_usize);
    for (index, prompt) in prompts.iter().enumerate() {
        let prompt_ids = fixture.tokenizer.encode(prompt, false).expect("encode");
        let ordinary_run = |max_new_tokens: usize, stops: Vec<u32>| {
            let mut stream = GenerationStream::new_text_only(
                &fixture.target,
                &fixture.tokenizer,
                request(prompt_ids.clone(), max_new_tokens, stops, Sampler::greedy()),
            )
            .expect("ordinary stream");
            drain(|| stream.next_token())
        };
        let dflash_run = |max_new_tokens: usize, stops: Vec<u32>| {
            let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                &fixture.target,
                &fixture.draft,
                &fixture.tokenizer,
                request(prompt_ids.clone(), max_new_tokens, stops, Sampler::greedy()),
                8,
                DFlash2P2Options::default(),
            )
            .expect("DFlash2 stream");
            let events = drain(|| stream.next_token());
            (events, stream.metrics())
        };

        // Multi-window run with a 61-token tail and a 3-token run whose
        // budget is smaller than one block.
        for max_new_tokens in [61, 3] {
            let expected = ordinary_run(max_new_tokens, Vec::new());
            let (actual, metrics) = dflash_run(max_new_tokens, Vec::new());
            assert_eq!(
                actual, expected,
                "affine{bits} draft={draft} prompt {index} max_new_tokens={max_new_tokens} diverged"
            );
            assert_eq!(metrics.generated_tokens, max_new_tokens);
            assert!(
                metrics.windows > 0 && metrics.drafted_tokens > 0,
                "no DFlash2 drafts"
            );
            eprintln!(
                "[qwen36-moe-dflash2 affine{bits} draft={draft}] prompt={index} max_new={max_new_tokens} \
                 windows={} drafted={} accepted={} rollbacks={} ordinary_windows={} exact",
                metrics.windows,
                metrics.drafted_tokens,
                metrics.accepted_draft_tokens,
                metrics.rollback_count,
                metrics.ordinary_windows,
            );
            totals.0 += metrics.windows;
            totals.1 += metrics.drafted_tokens;
            totals.2 += metrics.accepted_draft_tokens;
            totals.3 += metrics.rollback_count;
            totals.4 += metrics.ordinary_windows;
            if max_new_tokens == 61 {
                // Stop on a token emitted in the middle of the run: it is
                // reached inside a verify window, so the window is truncated
                // at the stop token.
                let stop = expected[30].0;
                let expected_stop = ordinary_run(61, vec![stop]);
                let (actual_stop, stop_metrics) = dflash_run(61, vec![stop]);
                assert_eq!(
                    actual_stop, expected_stop,
                    "affine{bits} draft={draft} prompt {index} stop token {stop} diverged"
                );
                assert!(expected_stop.last().is_some_and(|event| event.0 == stop));
                assert!(stop_metrics.generated_tokens < 61);
            }
        }
    }
    eprintln!(
        "[qwen36-moe-dflash2 affine{bits} draft={draft}] totals windows={} drafted={} accepted={} rollbacks={} ordinary_windows={}",
        totals.0, totals.1, totals.2, totals.3, totals.4
    );
    assert!(totals.3 > 0, "no rollback was exercised");
}

/// Upper chi-square quantile by the Wilson-Hilferty approximation.
fn chi_square_critical(df: f64, z: f64) -> f64 {
    let term = 1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt();
    df * term.powi(3)
}

/// Pearson goodness of fit of `counts` (over `trials`) against `expected`
/// probabilities; bins with an expected count below five are pooled.
fn chi_square(
    counts: &std::collections::HashMap<u32, usize>,
    expected: &[f32],
    trials: usize,
) -> (f64, f64) {
    let mut statistic = 0.0;
    let mut bins = 0_usize;
    let (mut pooled_expected, mut pooled_observed) = (0.0_f64, 0.0_f64);
    for (token, &probability) in expected.iter().enumerate() {
        let expected_count = f64::from(probability) * trials as f64;
        let observed = *counts.get(&(token as u32)).unwrap_or(&0) as f64;
        if expected_count >= 5.0 {
            statistic += (observed - expected_count).powi(2) / expected_count;
            bins += 1;
        } else {
            pooled_expected += expected_count;
            pooled_observed += observed;
        }
    }
    if pooled_expected >= 5.0 {
        statistic += (pooled_observed - pooled_expected).powi(2) / pooled_expected;
        bins += 1;
    } else {
        assert!(
            pooled_observed <= 5.0 + 4.0 * pooled_expected.sqrt(),
            "{pooled_observed} samples landed in bins with expected mass {pooled_expected}"
        );
    }
    let df = bins.saturating_sub(1).max(1) as f64;
    (statistic, chi_square_critical(df, 3.09))
}

/// Stateful exact speculative sampling on real MoE verify logits and real
/// drafter proposals: the first emitted token must follow the target's
/// processed distribution at the anchor position, and the second emitted
/// token (after the first draft was accepted) the distribution at the next
/// position. Proposals are deterministic, so acceptance is "sampled target
/// token equals draft token" and a rejection emits the sampled token.
#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_exact_sampling_matches_target_distribution() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let bits = fixture.target.dflash2_target_bits().expect("bits");
    let draft = fixture.draft_label;
    let trials: usize = std::env::var("QWEN36_MOE_DFLASH2_SAMPLING_TRIALS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4_000);
    let sampler = Sampler::greedy()
        .with_temperature(1.0)
        .with_top_k(20)
        .with_top_p(0.95)
        .with_seed(20_261_009);
    let model = &fixture.target;
    let taps = &fixture.draft.config().dflash_config.target_layer_ids;
    let prompt = fixture
        .tokenizer
        .encode("Here is a short story about a lighthouse keeper who", false)
        .expect("encode");
    let draft_len = 7_usize;
    let mut cache = model
        .make_cache(1, (prompt.len() + 32) as i32, model.cache_dtype())
        .expect("cache");
    let prompt_ids: Array = (&prompt[..], &[1_i32, prompt.len() as i32][..])
        .try_into()
        .expect("prompt ids");
    let prefill = model
        .dflash2_forward_target_on(
            &prompt_ids,
            &build_position_ids(0, prompt.len() as i32).expect("positions"),
            Some(&mut cache),
            taps,
            DFlash2TargetForwardMode::Prefill,
            StreamOrDevice::default(),
        )
        .expect("prefill");
    let last = slice_sequence_position(
        &prefill.hidden,
        prompt.len() as i32 - 1,
        StreamOrDevice::default(),
    )
    .expect("last hidden");
    let first_logits = model
        .dflash2_project_hidden_on(&last, StreamOrDevice::default())
        .expect("first logits");
    let current = mlx::ops::reduction::argmax(&first_logits, -1, false)
        .expect("argmax")
        .reshape((-1,))
        .expect("flatten")
        .to_vec::<u32>()
        .expect("read")[0];

    let mut draft_cache = fixture.draft.make_cache(0).expect("draft cache");
    let mut block = vec![current];
    block.resize(
        draft_len + 1,
        fixture.draft.config().dflash_config.mask_token_id,
    );
    let block: Array = (&block[..], &[1_i32, (draft_len + 1) as i32][..])
        .try_into()
        .expect("block");
    let draft_tokens = fixture
        .draft
        .propose_greedy_on(
            model,
            &block,
            &prefill.context_hidden,
            &mut draft_cache,
            StreamOrDevice::default(),
        )
        .expect("draft")
        .to_vec::<u32>()
        .expect("draft tokens");
    let mut verify = vec![current];
    verify.extend_from_slice(&draft_tokens);
    let verify_ids: Array = (&verify[..], &[1_i32, verify.len() as i32][..])
        .try_into()
        .expect("verify ids");
    let verified = model
        .dflash2_forward_target_on(
            &verify_ids,
            &build_position_ids(prompt.len() as i32, verify.len() as i32).expect("positions"),
            Some(&mut cache),
            taps,
            DFlash2TargetForwardMode::SampledVerify,
            StreamOrDevice::default(),
        )
        .expect("verify");
    let logits = model
        .dflash2_project_hidden_on(&verified.hidden, StreamOrDevice::default())
        .expect("verify logits");
    mlx::transforms::eval(&[&logits]).expect("eval logits");

    let rows = logits
        .reshape(&[(draft_len + 1) as i32, logits.shape().as_slice()[2]][..])
        .expect("rows");
    let expected = ironmlx_core::sampler::test_support::distributions(
        &sampler,
        &rows,
        &vec![&[][..]; draft_len + 1],
    )
    .expect("target distributions");
    let mut prng_state = mlx::random::key(sampler.seed).expect("prng");
    let mut first = std::collections::HashMap::<u32, usize>::new();
    let mut second = std::collections::HashMap::<u32, usize>::new();
    let mut second_trials = 0_usize;
    let mut accepted_lengths = vec![0_usize; draft_len + 1];
    for _ in 0..trials {
        let uniforms = prepare_uniforms(&mut prng_state, draft_len + 1)
            .expect("uniforms")
            .to_vec::<f32>()
            .expect("read uniforms");
        let prepared = prepare_dflash2_exact_sampling(&logits, sampler, draft_len, true)
            .expect("prepare")
            .expect("prepared sampling");
        let target_tokens = prepared.sample(&uniforms).expect("sample");
        let resolution = resolve_exact_deterministic_target_tokens(&draft_tokens, &target_tokens)
            .expect("resolve");
        let emitted = &resolution.tokens_to_append;
        assert_eq!(emitted.len(), resolution.accepted_draft_len + 1);
        assert_eq!(
            &emitted[..resolution.accepted_draft_len],
            &draft_tokens[..resolution.accepted_draft_len]
        );
        accepted_lengths[resolution.accepted_draft_len] += 1;
        *first.entry(emitted[0]).or_default() += 1;
        if emitted[0] == draft_tokens[0] {
            second_trials += 1;
            *second.entry(emitted[1]).or_default() += 1;
        }
    }
    let (statistic, critical) = chi_square(&first, &expected[0].probabilities(), trials);
    eprintln!(
        "[qwen36-moe-dflash2 affine{bits} draft={draft}] sampling trials={trials} accepted_length_histogram={accepted_lengths:?} \
         first chi2={statistic:.2} critical(p=0.001)={critical:.2}"
    );
    assert!(
        statistic <= critical,
        "first emitted token deviates from the target distribution"
    );
    if std::env::var_os("QWEN36_MOE_DFLASH2_SAMPLING_DEBUG").is_some() {
        let row1 = expected[1].probabilities();
        let logits1 = mlx::ops::cast::astype(&rows, mlx::Dtype::Float32)
            .expect("cast")
            .to_vec::<f32>()
            .expect("read");
        let vocab = row1.len();
        let row_logits = &logits1[vocab..2 * vocab];
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|&a, &b| row_logits[b].total_cmp(&row_logits[a]));
        for (rank, &token) in order.iter().take(24).enumerate() {
            eprintln!(
                "debug rank={rank} token={token} logit={} expected={} observed={}",
                row_logits[token],
                row1[token],
                second.get(&(token as u32)).unwrap_or(&0)
            );
        }
        for (&token, &count) in &second {
            if row1[token as usize] == 0.0 {
                let rank = order.iter().position(|&t| t == token as usize).unwrap();
                eprintln!(
                    "debug zero-mass token={token} count={count} rank={rank} logit={}",
                    row_logits[token as usize]
                );
            }
        }
    }
    if second_trials >= 200 {
        let (statistic, critical) =
            chi_square(&second, &expected[1].probabilities(), second_trials);
        eprintln!(
            "[qwen36-moe-dflash2 affine{bits} draft={draft}] sampling second trials={second_trials} chi2={statistic:.2} critical(p=0.001)={critical:.2}"
        );
        assert!(
            statistic <= critical,
            "second emitted token deviates from the target distribution"
        );
    } else {
        panic!(
            "only {second_trials} trials accepted the first draft; choose a lower-entropy prompt"
        );
    }
}

fn assert_cache_exact(label: &str, expected: &[LayerCache], actual: &[LayerCache]) {
    let exact = |what: String, left: &Array, right: &Array| {
        assert_eq!(left.shape(), right.shape(), "{label}: {what} shape");
        let bits = |array: &Array| {
            mlx::ops::cast::astype(array, mlx::Dtype::Float32)
                .expect("cast")
                .to_vec::<f32>()
                .expect("read")
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let mismatches = bits(left)
            .iter()
            .zip(bits(right).iter())
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(
            mismatches, 0,
            "{label}: {what}: {mismatches} elements differ"
        );
    };
    assert_eq!(expected.len(), actual.len(), "{label}: layer count");
    for (layer, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        match (expected, actual) {
            (LayerCache::Full(expected), LayerCache::Full(actual)) => {
                assert_eq!(
                    expected.offsets(),
                    actual.offsets(),
                    "{label}: layer {layer} KV offsets"
                );
                let offset = expected.offsets()[0];
                for (index, (left, right)) in expected
                    .diagnostic_buffers()
                    .into_iter()
                    .zip(actual.diagnostic_buffers())
                    .enumerate()
                {
                    let prefix = |array: &Array| {
                        let dims = array.shape();
                        let dims = dims.as_slice();
                        mlx::ops::indexing::slice_strided(
                            array,
                            &[0_i32, 0, 0, 0][..],
                            &[dims[0], dims[1], offset, dims[3]][..],
                            &[1_i32, 1, 1, 1][..],
                        )
                        .expect("KV prefix")
                    };
                    exact(
                        format!("layer {layer} KV buffer {index}"),
                        &prefix(left),
                        &prefix(right),
                    );
                }
            }
            (LayerCache::Linear(expected), LayerCache::Linear(actual)) => {
                assert_eq!(
                    expected.offsets(),
                    actual.offsets(),
                    "{label}: layer {layer} GDN offsets"
                );
                exact(
                    format!("layer {layer} GDN conv state"),
                    expected.conv_state(),
                    actual.conv_state(),
                );
                exact(
                    format!("layer {layer} GDN recurrent state"),
                    expected.recurrent_state(),
                    actual.recurrent_state(),
                );
            }
            _ => panic!("{label}: layer {layer} cache kind mismatch"),
        }
    }
}

/// Sampled DFlash2 generation that crosses many windows, including drafts
/// rejected by exact speculative sampling (residual corrections), must leave
/// the target cache exactly where ordinary token-by-token decoding of the
/// same emitted tokens leaves it: KV prefix and GDN conv/recurrent state of
/// every layer. A wrong rollback length or a stale recurrent state after a
/// rejection would diverge here; it would also make every later window draw
/// from the wrong conditional distribution. The per-window acceptance and
/// residual distribution is checked separately by
/// `qwen36_moe_dflash2_exact_sampling_matches_target_distribution`. Runs the
/// production adaptive draft budget and a fixed budget of 7 (more drafted
/// positions, more rejections per window).
#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_sampled_generation_state_matches_teacher_forced_decode() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let bits = fixture.target.dflash2_target_bits().expect("bits");
    let draft = fixture.draft_label;
    let model = &fixture.target;
    let sampler = Sampler::greedy()
        .with_temperature(1.0)
        .with_top_k(20)
        .with_top_p(0.95)
        .with_seed(20_261_009);
    let budget_setting = ironmlx_core::m5_profile::settings::DFLASH2_FIXED_BUDGET;
    let previous_budget = std::env::var(budget_setting).ok();
    let mut totals = (0_usize, 0_usize, 0_usize, 0_usize);
    for budget in [None, Some("7")] {
        match budget {
            Some(value) => std::env::set_var(budget_setting, value),
            None => std::env::remove_var(budget_setting),
        }
        for (index, prompt) in PROMPTS.iter().enumerate() {
            let prompt_ids = fixture.tokenizer.encode(prompt, false).expect("encode");
            let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                model,
                &fixture.draft,
                &fixture.tokenizer,
                request(prompt_ids.clone(), 64, Vec::new(), sampler),
                8,
                DFlash2P2Options::default(),
            )
            .expect("DFlash2 stream");
            let events = drain(|| stream.next_token());
            let metrics = stream.metrics();
            let label = format!(
                "affine{bits} draft={draft} budget={} prompt={index}",
                budget.unwrap_or("adaptive")
            );
            assert!(
                metrics.sampled,
                "{label}: request did not run sampled verification"
            );
            let emitted: Vec<u32> = events.iter().map(|event| event.0).collect();
            assert_eq!(emitted.len(), 64, "{label}: emitted length");
            assert!(
                stream.history.starts_with(&prompt_ids),
                "{label}: history prefix"
            );
            assert_eq!(
                &stream.history[prompt_ids.len()..],
                &emitted[..],
                "{label}: history"
            );

            // The committed cache holds a prefix of the history: everything
            // but the pending input token, or all of it when the length
            // limit cut the final bonus token.
            let committed = stream
                .target_cache
                .iter()
                .find_map(|cache| match cache {
                    LayerCache::Full(cache) => Some(cache.offsets()[0] as usize),
                    _ => None,
                })
                .expect("full-attention layer");
            assert!(
                committed + 1 == stream.history.len() || committed == stream.history.len(),
                "{label}: committed {committed} vs history {}",
                stream.history.len()
            );

            let mut reference = model
                .make_cache(1, (stream.history.len() + 8) as i32, model.cache_dtype())
                .expect("reference cache");
            let forward = |tokens: &[u32], start: usize, cache: &mut [LayerCache]| {
                let ids: Array = (tokens, &[1_i32, tokens.len() as i32][..])
                    .try_into()
                    .expect("ids");
                let positions =
                    build_position_ids(start as i32, tokens.len() as i32).expect("positions");
                let hidden = Model::forward_text_hidden(
                    model,
                    &ids,
                    &positions,
                    None,
                    None,
                    Some(cache),
                    StreamOrDevice::default(),
                )
                .expect("ordinary forward");
                mlx::transforms::eval(&[&hidden]).expect("eval ordinary");
            };
            forward(&prompt_ids, 0, &mut reference);
            for position in prompt_ids.len()..committed {
                forward(
                    &stream.history[position..=position],
                    position,
                    &mut reference,
                );
            }
            assert_cache_exact(&label, &reference, &stream.target_cache);
            eprintln!(
                "[qwen36-moe-dflash2 {label}] windows={} ordinary_windows={} drafted={} accepted={} \
                 exact_windows={} residual_corrections={} bonus={} rollbacks={} committed={committed} state exact",
                metrics.windows,
                metrics.ordinary_windows,
                metrics.drafted_tokens,
                metrics.accepted_draft_tokens,
                metrics.exact_sampling_windows,
                metrics.exact_residual_corrections,
                metrics.exact_bonus_samples,
                metrics.rollback_count,
            );
            totals.0 += metrics.exact_sampling_windows;
            totals.1 += metrics.exact_residual_corrections;
            totals.2 += metrics.accepted_draft_tokens;
            totals.3 += metrics.rollback_count;
        }
    }
    match previous_budget {
        Some(value) => std::env::set_var(budget_setting, value),
        None => std::env::remove_var(budget_setting),
    }
    eprintln!(
        "[qwen36-moe-dflash2 affine{bits} draft={draft}] sampled totals exact_windows={} residual_corrections={} accepted={} rollbacks={}",
        totals.0, totals.1, totals.2, totals.3
    );
    assert!(
        totals.1 > 0,
        "no rejected draft (residual correction) was exercised"
    );
    assert!(totals.2 > 0, "no accepted draft was exercised");
}

/// Greedy flat-tree DFlash2 generation must emit the same token IDs and
/// finish reason as ordinary decoding, actually execute tree windows, and
/// keep sampled requests on linear windows with a recorded fallback.
#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_tree_generation_matches_ordinary() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let bits = fixture.target.dflash2_target_bits().expect("bits");
    let draft = fixture.draft_label;
    let max_nodes = fixture
        .target
        .dflash2_verify_capabilities()
        .flat_tree_max_nodes;
    for (index, prompt) in PROMPTS.iter().enumerate() {
        let prompt_ids = fixture.tokenizer.encode(prompt, false).expect("encode");
        let mut ordinary = GenerationStream::new_text_only(
            &fixture.target,
            &fixture.tokenizer,
            request(prompt_ids.clone(), 61, Vec::new(), Sampler::greedy()),
        )
        .expect("ordinary stream");
        let expected = drain(|| ordinary.next_token());
        for nodes in [7, max_nodes] {
            let tree_run = |stops: Vec<u32>| {
                let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                    &fixture.target,
                    &fixture.draft,
                    &fixture.tokenizer,
                    request(prompt_ids.clone(), 61, stops, Sampler::greedy()),
                    8,
                    DFlash2P2Options {
                        tree_max_nodes: nodes,
                        position_keyed_sampling: false,
                    },
                )
                .expect("tree stream");
                let events = drain(|| stream.next_token());
                (events, stream.metrics())
            };
            let (actual, metrics) = tree_run(Vec::new());
            assert_eq!(
                actual, expected,
                "affine{bits} draft={draft} tree{nodes} prompt {index} diverged"
            );
            assert!(metrics.tree_windows > 0, "no tree window executed");
            assert_eq!(metrics.tree_fallback_linear_windows, 0);
            eprintln!(
                "[qwen36-moe-dflash2-tree affine{bits} draft={draft}] prompt={index} nodes={nodes} windows={} tree_windows={} tree_nodes={} accepted={} ordinary_windows={} exact",
                metrics.windows,
                metrics.tree_windows,
                metrics.tree_drafted_nodes,
                metrics.accepted_draft_tokens,
                metrics.ordinary_windows,
            );
            let stop = expected[30].0;
            let mut ordinary_stop = GenerationStream::new_text_only(
                &fixture.target,
                &fixture.tokenizer,
                request(prompt_ids.clone(), 61, vec![stop], Sampler::greedy()),
            )
            .expect("ordinary stop stream");
            let expected_stop = drain(|| ordinary_stop.next_token());
            let (actual_stop, _) = tree_run(vec![stop]);
            assert_eq!(
                actual_stop, expected_stop,
                "affine{bits} draft={draft} tree{nodes} stop diverged"
            );
        }
    }

    // Sampled requests are not tree-qualified: they run linear windows and
    // report the fallback.
    let prompt_ids = fixture.tokenizer.encode(PROMPTS[0], false).expect("encode");
    let sampler = Sampler::greedy()
        .with_temperature(1.0)
        .with_top_k(20)
        .with_top_p(0.95)
        .with_seed(7);
    let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
        &fixture.target,
        &fixture.draft,
        &fixture.tokenizer,
        request(prompt_ids, 32, Vec::new(), sampler),
        8,
        DFlash2P2Options {
            tree_max_nodes: max_nodes,
            position_keyed_sampling: false,
        },
    )
    .expect("sampled stream");
    let events = drain(|| stream.next_token());
    let metrics = stream.metrics();
    assert_eq!(events.len(), 32);
    assert_eq!(metrics.tree_windows, 0);
    assert!(metrics.tree_fallback_linear_windows > 0);
    eprintln!(
        "[qwen36-moe-dflash2-tree affine{bits} draft={draft}] sampled request: tree_windows=0 linear_fallback_windows={}",
        metrics.tree_fallback_linear_windows
    );
}
