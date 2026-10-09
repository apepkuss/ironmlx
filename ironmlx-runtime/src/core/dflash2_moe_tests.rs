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
                 policy={} budget_windows={:?} windows={} drafted={} accepted={} rollbacks={} ordinary_windows={} exact",
                metrics.budget_policy,
                metrics.budget_windows,
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
                "[qwen36-moe-dflash2 {label}] policy={} budget_windows={:?} windows={} ordinary_windows={} drafted={} accepted={} \
                 exact_windows={} residual_corrections={} bonus={} rollbacks={} committed={committed} state exact",
                metrics.budget_policy,
                metrics.budget_windows,
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

fn greedy_ordinary(fixture: &Fixture, prompt_ids: &[u32], max_new_tokens: usize) -> Vec<Event> {
    let mut stream = GenerationStream::new_text_only(
        &fixture.target,
        &fixture.tokenizer,
        request(
            prompt_ids.to_vec(),
            max_new_tokens,
            Vec::new(),
            Sampler::greedy(),
        ),
    )
    .expect("ordinary stream");
    drain(|| stream.next_token())
}

/// Teacher-forced ordinary decode of `stream`'s history up to its committed
/// target-cache length, compared bit for bit with the stream's target cache.
fn assert_committed_cache_matches_teacher_forced<M: DFlash2Target>(
    label: &str,
    model: &M,
    stream: &DFlash2TextGenerationStream<'_, M>,
    prompt_len: usize,
) {
    let committed = stream
        .target_cache
        .iter()
        .find_map(|cache| match cache {
            LayerCache::Full(cache) => Some(cache.offsets()[0] as usize),
            _ => None,
        })
        .expect("full-attention layer");
    let mut reference = model
        .make_cache(1, (stream.history.len() + 8) as i32, model.cache_dtype())
        .expect("reference cache");
    let forward = |tokens: &[u32], start: usize, cache: &mut [LayerCache]| {
        let ids: Array = (tokens, &[1_i32, tokens.len() as i32][..])
            .try_into()
            .expect("ids");
        let positions = build_position_ids(start as i32, tokens.len() as i32).expect("positions");
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
    // Prefill in the stream's request chunks: the GDN recurrence of one long
    // forward accumulates in a different order than chunked prefill.
    let chunk = stream.request.prefill_chunk_size.max(1);
    let mut start = 0;
    while start < prompt_len {
        let end = (start + chunk).min(prompt_len);
        forward(&stream.history[start..end], start, &mut reference);
        start = end;
    }
    for position in prompt_len..committed {
        forward(
            &stream.history[position..=position],
            position,
            &mut reference,
        );
    }
    assert_cache_exact(label, &reference, &stream.target_cache);
}

/// Window-cost policy across context buckets (2k and 8k boundaries are
/// crossed during generation): greedy output equals ordinary decoding and the
/// policy re-calibrates in the new cost regime.
#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_window_cost_recalibrates_across_context_buckets() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let bits = fixture.target.dflash2_target_bits().expect("bits");
    let draft = fixture.draft_label;
    let mut text = String::new();
    let mut ids = Vec::new();
    while ids.len() < 8_200 {
        for prompt in PROMPTS {
            text.push_str(prompt);
            text.push('\n');
        }
        ids = fixture.tokenizer.encode(&text, false).expect("encode");
    }
    for (prompt_len, max_new_tokens, boundary) in
        [(1_960_usize, 200_usize, 2_048_usize), (8_120, 160, 8_192)]
    {
        let prompt_ids = ids[..prompt_len].to_vec();
        let expected = greedy_ordinary(&fixture, &prompt_ids, max_new_tokens);
        let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
            &fixture.target,
            &fixture.draft,
            &fixture.tokenizer,
            request(
                prompt_ids.clone(),
                max_new_tokens,
                Vec::new(),
                Sampler::greedy(),
            ),
            8,
            DFlash2P2Options::default(),
        )
        .expect("DFlash2 stream");
        let actual = drain(|| stream.next_token());
        let metrics = stream.metrics();
        let label = format!("affine{bits} draft={draft} prompt={prompt_len} boundary={boundary}");
        assert_eq!(actual, expected, "{label}: diverged from ordinary decoding");
        assert_eq!(metrics.budget_policy, "window-cost", "{label}");
        assert!(
            prompt_len + 1 <= boundary && prompt_len + max_new_tokens > boundary,
            "{label}: generation does not cross the boundary"
        );
        eprintln!(
            "[qwen36-moe-dflash2 {label}] policy={} calibration_windows={} probe_windows={} budget_windows={:?} \
             windows={} drafted={} accepted={} rollbacks={} exact",
            metrics.budget_policy,
            metrics.budget_calibration_windows,
            metrics.budget_probe_windows,
            metrics.budget_windows,
            metrics.windows,
            metrics.drafted_tokens,
            metrics.accepted_draft_tokens,
            metrics.rollback_count,
        );
        // The first calibration has one window per calibration budget; a
        // second regime adds calibration windows for the budgets it has not
        // measured yet (the window that crossed the boundary already counts).
        assert!(
            metrics.budget_calibration_windows > 4,
            "{label}: no second calibration after crossing the bucket ({} calibration windows)",
            metrics.budget_calibration_windows
        );
        assert_committed_cache_matches_teacher_forced(&label, &fixture.target, &stream, prompt_len);
    }
}

/// Window-cost policy on text whose first part is hard to draft (the policy
/// settles on ordinary decoding) and whose second part repeats one line (easy
/// to draft): the policy must leave budget 0 again. Checks per window that the
/// raw target context kept for the paused draft cache stays contiguous and
/// bounded, then compares output and committed target cache with ordinary
/// decoding.
/// One window-cost DFlash2 run of a greedy request with per-window
/// observation; the correctness checks shared by the budget-0 tests have
/// already passed when it is returned.
struct WindowCostRun {
    label: String,
    prompt_tokens: usize,
    events: Vec<Event>,
    /// Per window: drafted width, measured time, whether it was a probe, and
    /// the generated tokens committed once it completed.
    widths: Vec<usize>,
    window_us: Vec<u64>,
    probe: Vec<bool>,
    window_end: Vec<usize>,
    max_pending: i32,
    probes: usize,
    calibrations: usize,
    records: Vec<WindowRecord>,
}

/// Everything observed about one window: role, accepted draft tokens, and the
/// state the next window starts from (pending target context, and the draft
/// cache position checked against the verify start when a drafting window
/// follows).
struct WindowRecord {
    role: &'static str,
    accepted: usize,
    pending_after: i32,
    /// (draft cache processed positions, pending, verify start) before the
    /// next window, when it drafts.
    next_position: Option<(i32, i32, i32)>,
}

/// The registered sustained-drafting criterion: after the first run of at
/// least 8 ordinary windows, a complete span of 50 windows with at least 45
/// drafting windows and at least 100 committed tokens. Returns the first such
/// span as (first window, end window, drafting windows, committed tokens) and
/// the budget-0 run it follows.
#[allow(clippy::type_complexity)]
fn sustained_drafting_after_budget_zero(
    widths: &[usize],
    committed: &[usize],
) -> (Option<(usize, usize)>, Option<(usize, usize, usize, usize)>) {
    assert_eq!(widths.len(), committed.len());
    let mut first_run = None;
    let mut index = 0;
    while index < widths.len() {
        if widths[index] == 0 {
            let first = index;
            while index < widths.len() && widths[index] == 0 {
                index += 1;
            }
            if index - first >= 8 {
                first_run = Some((first, index - first));
                break;
            }
        } else {
            index += 1;
        }
    }
    let Some((first, length)) = first_run else {
        return (None, None);
    };
    let span = (first + length..widths.len().saturating_sub(49)).find_map(|start| {
        let end = start + 50;
        let drafting = widths[start..end]
            .iter()
            .filter(|&&width| width > 0)
            .count();
        let tokens: usize = committed[start..end].iter().sum();
        (drafting >= 45 && tokens >= 100).then_some((start, end, drafting, tokens))
    });
    (first_run, span)
}

#[test]
fn sustained_drafting_judge_on_crafted_windows() {
    let judge = sustained_drafting_after_budget_zero;
    // 10 ordinary windows, then 60 windows at budget 1 committing 1 token each
    // (all drafting, 50 tokens per span), then 60 windows at budget 3
    // committing 3 tokens each: the first span that drafts enough commits too
    // little, a later span passes.
    let mut widths = vec![0; 10];
    let mut committed = vec![1; 10];
    widths.extend([1; 60]);
    committed.extend([1; 60]);
    widths.extend([3; 60]);
    committed.extend([3; 60]);
    let (run, span) = judge(&widths, &committed);
    assert_eq!(run, Some((0, 10)));
    let (start, end, drafting, tokens) = span.expect("a later span passes");
    assert_eq!(end - start, 50);
    assert!(drafting >= 45 && tokens >= 100);
    assert!(
        start > 10,
        "the first drafting span alone ({start}) commits too little"
    );
    // No span satisfies both: drafting spans commit 1 token per window, and
    // spans committing enough include too many ordinary windows.
    let mut widths = vec![0; 10];
    let mut committed = vec![1; 10];
    for _ in 0..20 {
        widths.extend([1, 1, 1, 1, 0, 0]);
        committed.extend([1, 1, 1, 1, 1, 1]);
    }
    widths.extend([1; 60]);
    committed.extend([1; 60]);
    assert!(judge(&widths, &committed).1.is_none());
    // The only qualifying windows form a tail shorter than 50 windows: 40
    // windows after the budget-0 run, all drafting 8 tokens each.
    let widths: Vec<usize> = [vec![0; 10], vec![7; 40]].concat();
    let committed: Vec<usize> = [vec![1; 10], vec![8; 40]].concat();
    assert!(
        judge(&widths, &committed).1.is_none(),
        "a tail shorter than 50 windows cannot pass"
    );
    // 62 windows follow the budget-0 run, but only the last 40 draft: every
    // complete span holds at most 30 drafting windows.
    let widths: Vec<usize> = [vec![0; 10], vec![1; 2], vec![0; 20], vec![7; 40]].concat();
    let committed: Vec<usize> = [vec![1; 10], vec![2; 2], vec![1; 20], vec![8; 40]].concat();
    assert!(
        judge(&widths, &committed).1.is_none(),
        "a qualifying tail of 40 windows cannot pass through a mixed span"
    );
    // No budget-0 run of 8 windows.
    let widths: Vec<usize> = [vec![0; 7], vec![3; 80]].concat();
    let committed: Vec<usize> = [vec![1; 7], vec![4; 80]].concat();
    assert_eq!(judge(&widths, &committed), (None, None));
}

impl WindowCostRun {
    fn window_start(&self, index: usize) -> usize {
        if index == 0 {
            0
        } else {
            self.window_end[index - 1]
        }
    }

    fn committed(&self, index: usize) -> usize {
        self.window_end[index] - self.window_start(index)
    }

    /// Runs of consecutive ordinary windows as (first window, length).
    fn ordinary_runs(&self) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut index = 0;
        while index < self.widths.len() {
            if self.widths[index] == 0 {
                let first = index;
                while index < self.widths.len() && self.widths[index] == 0 {
                    index += 1;
                }
                runs.push((first, index - first));
            } else {
                index += 1;
            }
        }
        runs
    }

    fn ordinary_median_us(&self) -> u64 {
        let mut ordinary: Vec<u64> = self
            .widths
            .iter()
            .zip(&self.window_us)
            .filter(|(width, _)| **width == 0)
            .map(|(_, us)| *us)
            .collect();
        ordinary.sort_unstable();
        ordinary
            .get(ordinary.len() / 2)
            .copied()
            .unwrap_or(u64::MAX)
    }

    /// First window at or after `from` whose next 50 windows (or the rest, at
    /// least 10) draft at least 90% of the time.
    fn sustained_drafting_from(&self, from: usize) -> Option<usize> {
        (from..self.widths.len()).find(|&index| {
            let span = &self.widths[index..(index + 50).min(self.widths.len())];
            span.len() >= 10
                && span.iter().filter(|&&width| width > 0).count() * 10 >= span.len() * 9
        })
    }

    fn compact_widths(&self) -> String {
        self.widths
            .iter()
            .map(|width| char::from(b'0' + *width as u8))
            .collect()
    }
}

/// The serve request's token stream for `content`: the Qwen chat template
/// with thinking disabled.
fn chat_prompt(fixture: &Fixture, content: &str) -> Vec<u32> {
    let prompt = format!(
        "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    fixture.tokenizer.encode(&prompt, false).expect("encode")
}

/// Runs `prompt_ids` through a window-cost DFlash2 stream, observing every
/// window, and checks the output against ordinary decoding, the pending
/// context bound, the draft cache position before every drafting window and
/// the committed target cache against a teacher-forced decode. With `warm`,
/// the two frozen warmup requests run through DFlash2 first in the same
/// process, as on a warmed serve.
fn run_window_cost_tracked(
    fixture: &Fixture,
    name: &str,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    warm: bool,
) -> WindowCostRun {
    let bits = fixture.target.dflash2_target_bits().expect("bits");
    let label = format!("affine{bits} draft={} {name}", fixture.draft_label);
    let expected = greedy_ordinary(fixture, prompt_ids, max_new_tokens);
    if warm {
        for content in [
            "请用三句话说明良好 API 错误信息应包含哪些内容。",
            "写一个 Python 函数，判断整数是否为偶数，并给出一个调用示例。",
        ] {
            let mut warmup = DFlash2TextGenerationStream::new_text_only_with_options(
                &fixture.target,
                &fixture.draft,
                &fixture.tokenizer,
                request(
                    chat_prompt(fixture, content),
                    512,
                    vec![248_046],
                    Sampler::greedy(),
                ),
                8,
                DFlash2P2Options::default(),
            )
            .expect("DFlash2 warmup stream");
            let tokens = drain(|| warmup.next_token()).len();
            eprintln!("[qwen36-moe-dflash2 {label}] warmup tokens={tokens}");
        }
    }
    let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
        &fixture.target,
        &fixture.draft,
        &fixture.tokenizer,
        request(
            prompt_ids.to_vec(),
            max_new_tokens,
            Vec::new(),
            Sampler::greedy(),
        ),
        8,
        DFlash2P2Options::default(),
    )
    .expect("DFlash2 stream");
    let mut widths = Vec::new();
    let mut window_us = Vec::new();
    let mut probe = Vec::new();
    let mut window_end = Vec::new();
    let mut last_budget_windows: Vec<usize> = Vec::new();
    let mut last_budget_window_us: Vec<u64> = Vec::new();
    let mut last_probes = 0;
    let mut last_calibrations = 0;
    let mut last_accepted = 0;
    let mut records = Vec::new();
    let mut max_pending = 0_i32;
    let mut events = Vec::new();
    let mut windows_seen = 0;
    let mut mismatches = Vec::new();
    loop {
        let event = stream.next_token().expect("generation step");
        let metrics = stream.metrics();
        if metrics.windows != windows_seen {
            windows_seen = metrics.windows;
            let width = metrics
                .budget_windows
                .iter()
                .enumerate()
                .find(|(b, n)| **n > last_budget_windows.get(*b).copied().unwrap_or(0))
                .map(|(b, _)| b)
                .expect("one new window");
            widths.push(width);
            window_us.push(
                metrics.budget_window_us[width]
                    - last_budget_window_us.get(width).copied().unwrap_or(0),
            );
            let is_probe = metrics.budget_probe_windows > last_probes;
            let role = if is_probe {
                "probe"
            } else if metrics.budget_calibration_windows > last_calibrations {
                "calibrate"
            } else {
                "exploit"
            };
            probe.push(is_probe);
            last_probes = metrics.budget_probe_windows;
            last_calibrations = metrics.budget_calibration_windows;
            let accepted = metrics.accepted_draft_tokens - last_accepted;
            last_accepted = metrics.accepted_draft_tokens;
            window_end.push(stream.history.len() - prompt_ids.len());
            last_budget_windows = metrics.budget_windows.clone();
            last_budget_window_us = metrics.budget_window_us.clone();
            let pending = stream.pending_context_hidden.shape().as_slice()[1];
            max_pending = max_pending.max(pending);
            let mut next_position = None;
            // Draft cache and pending context do not change while the window's
            // tokens are emitted, so this is the state the next window starts from.
            // Only when another window will run: at the end the last tokens are
            // still queued and no window follows.
            let more_windows = stream.emitted_new_tokens + stream.pending_tokens.len()
                < stream.request.max_new_tokens;
            if !stream.finished && more_windows && stream.current_draft_budget() > 0 {
                let (processed, _) = stream
                    .draft_cache
                    .position_signature()
                    .expect("draft position");
                let verify_start = (stream.history.len() - 1) as i32;
                next_position = Some((processed, pending, verify_start));
                if processed + pending != verify_start {
                    mismatches.push(format!(
                        "before window {}: processed={processed} pending={pending} verify_start={verify_start} \
                         emitted={} max_new={} next_budget={} queued={} last_widths={:?}",
                        widths.len() + 1,
                        stream.emitted_new_tokens,
                        stream.request.max_new_tokens,
                        stream.current_draft_budget(),
                        stream.pending_tokens.len(),
                        &widths[widths.len().saturating_sub(6)..]
                    ));
                }
            }
            records.push(WindowRecord {
                role,
                accepted,
                pending_after: pending,
                next_position,
            });
        }
        let Some(event) = event else { break };
        let finished = event.finish_reason.is_some();
        events.push((
            event.token,
            event.finish_reason.map(|reason| format!("{reason:?}")),
        ));
        if finished {
            break;
        }
    }
    let metrics = stream.metrics();
    let run = WindowCostRun {
        label,
        prompt_tokens: prompt_ids.len(),
        events,
        widths,
        window_us,
        probe,
        window_end,
        max_pending,
        probes: metrics.budget_probe_windows,
        calibrations: metrics.budget_calibration_windows,
        records,
    };
    let label = &run.label;
    // Window time per drafted width, split by whether the window directly
    // follows an ordinary window (and so also consumes its context).
    for width in 1..=7 {
        let mut after_ordinary = Vec::new();
        let mut after_drafting = Vec::new();
        for index in 1..run.widths.len() {
            if run.widths[index] == width {
                if run.widths[index - 1] == 0 {
                    after_ordinary.push(run.window_us[index]);
                } else {
                    after_drafting.push(run.window_us[index]);
                }
            }
        }
        let median = |values: &mut Vec<u64>| {
            values.sort_unstable();
            values.get(values.len() / 2).copied()
        };
        if !after_ordinary.is_empty() || !after_drafting.is_empty() {
            eprintln!(
                "[qwen36-moe-dflash2 {label}] width {width}: after ordinary n={} median_us={:?} all={after_ordinary:?}; after drafting n={} median_us={:?}",
                after_ordinary.len(),
                median(&mut after_ordinary.clone()),
                after_drafting.len(),
                median(&mut after_drafting),
            );
        }
    }
    eprintln!(
        "[qwen36-moe-dflash2 {label}] prompt_tokens={} windows={} ordinary_runs={:?} max_pending_context={} probes={} calibrations={} drafted={} accepted={} ordinary_median_us={} widths={}",
        run.prompt_tokens,
        run.widths.len(),
        run.ordinary_runs(),
        run.max_pending,
        run.probes,
        run.calibrations,
        metrics.drafted_tokens,
        metrics.accepted_draft_tokens,
        run.ordinary_median_us(),
        run.compact_widths(),
    );
    eprintln!(
        "[qwen36-moe-dflash2 {label}] committed_per_window={:?}",
        (0..run.widths.len())
            .map(|index| run.committed(index))
            .collect::<Vec<_>>()
    );
    for (index, record) in run.records.iter().enumerate() {
        eprintln!(
            "[qwen36-moe-dflash2 {label}] window {index} width={} role={} accepted={} committed={} us={} pending_after={} next_position={:?}",
            run.widths[index],
            record.role,
            record.accepted,
            run.committed(index),
            run.window_us[index],
            record.pending_after,
            record.next_position,
        );
    }
    for mismatch in &mismatches {
        eprintln!("[qwen36-moe-dflash2 {label}] draft position mismatch {mismatch}");
    }
    assert_eq!(
        run.events, expected,
        "{label}: diverged from ordinary decoding"
    );
    assert_eq!(metrics.budget_policy, "window-cost", "{label}");
    // Pending context is consumed by every drafting window; between them it
    // grows by one position per ordinary window (plus the prompt at the start).
    assert!(
        (run.max_pending as usize) <= prompt_ids.len() + 1 + 64 + 1,
        "{label}: pending context {} exceeds the probe bound",
        run.max_pending
    );
    assert_committed_cache_matches_teacher_forced(
        label,
        &fixture.target,
        &stream,
        prompt_ids.len(),
    );
    assert!(
        mismatches.is_empty(),
        "{label}: draft cache not contiguous: {mismatches:?}"
    );
    run
}

/// Recovery on a predictable copy section after `run` stayed on budget 0:
/// the copy must start, a budget-0 run of at least 8 windows must precede it
/// and drafting must be sustained within 128 tokens of it, committing at least
/// 2 tokens per drafting window at no more than 0.8x the ordinary time per
/// token.
fn assert_recovers_on_copy_section(fixture: &Fixture, run: &WindowCostRun) {
    let label = &run.label;
    let generated: Vec<u32> = run.events.iter().map(|(token, _)| *token).collect();
    let copy_start = (1..=generated.len())
        .find(|&end| {
            fixture
                .tokenizer
                .decode(&generated[..end], true)
                .expect("decode")
                .contains("The quick brown fox")
        })
        .expect("the copy section starts");
    let copy_window = (0..run.widths.len())
        .find(|&index| run.window_end[index] >= copy_start)
        .expect("a window reaches the copy section");
    let sustained = (copy_window..run.widths.len())
        .filter(|&index| run.window_start(index) + 1 >= copy_start)
        .find(|&index| run.sustained_drafting_from(index) == Some(index));
    let longest_before_copy = run
        .ordinary_runs()
        .into_iter()
        .filter(|(first, _)| *first < copy_window)
        .map(|(_, length)| length)
        .max()
        .unwrap_or(0);
    let ordinary_median = run.ordinary_median_us();
    let recovery = sustained.map(|start| {
        let delay_tokens = run.window_start(start).saturating_sub(copy_start - 1);
        let delay_windows = start.saturating_sub(copy_window);
        let delay_us: u64 = run.window_us[copy_window..start].iter().sum();
        let drafting: Vec<usize> = (start..run.widths.len())
            .filter(|&index| run.widths[index] > 0)
            .collect();
        let per_drafting_window = drafting
            .iter()
            .map(|&index| run.committed(index))
            .sum::<usize>() as f64
            / drafting.len().max(1) as f64;
        let after_tokens: usize = (start..run.widths.len()).map(|i| run.committed(i)).sum();
        let after_us: u64 = run.window_us[start..].iter().sum();
        (
            delay_tokens,
            delay_windows,
            delay_us,
            per_drafting_window,
            after_us as f64 / after_tokens.max(1) as f64,
        )
    });
    eprintln!(
        "[qwen36-moe-dflash2 {label}] copy_start_token={copy_start} longest_ordinary_run_before_copy={longest_before_copy} sustained_drafting_window={sustained:?} \
         recovery(delay_tokens,delay_windows,delay_us,tokens_per_drafting_window,us_per_token)={recovery:?} \
         ordinary_median_us={ordinary_median}"
    );
    assert!(
        longest_before_copy >= 8,
        "{label}: the policy never stayed on budget 0 before the copy (longest run {longest_before_copy})"
    );
    let (delay_tokens, _, _, per_drafting_window, us_per_token) =
        recovery.expect("drafting is sustained on the copy section");
    assert!(
        delay_tokens <= 128,
        "{label}: drafting resumed {delay_tokens} tokens into the copy section"
    );
    assert!(
        per_drafting_window >= 2.0,
        "{label}: {per_drafting_window:.2} tokens per drafting window after recovery"
    );
    assert!(
        us_per_token <= 0.8 * ordinary_median as f64,
        "{label}: {us_per_token:.0} us per token after recovery vs ordinary {ordinary_median} us"
    );
}

const COPY_SECTION: &str = "写完之后，另起一段，把下面这一行原样抄写 40 遍，每遍单独一行，不要编号：\nThe quick brown fox jumps over the lazy dog 0123456789";
const BASE64_CONTENT: &str =
    "随机生成 20 个互不相同、长度为 44 个字符的 Base64 字符串，每行一个，不要编号也不要解释。";

#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_window_cost_resumes_drafting_after_budget_zero() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let prompt_ids = chat_prompt(
        &fixture,
        &format!(
            "随机生成 25 个互不相同的 12 位密码，每个由大小写字母、数字和符号混合组成，每行一个，不要编号也不要解释。{COPY_SECTION}"
        ),
    );
    let run = run_window_cost_tracked(&fixture, "zero-resume", &prompt_ids, 700, false);
    assert_recovers_on_copy_section(&fixture, &run);
}

/// The Base64 request of the affine4 qualification run (serve token stream,
/// warmed process): budget 0 must be entered for at least 8 windows and a
/// later complete span of 50 windows must hold at least 45 drafting windows
/// and at least 100 committed tokens. It has no content change point, so no
/// recovery delay is asserted.
#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_window_cost_recovers_on_base64() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let prompt_ids = chat_prompt(&fixture, BASE64_CONTENT);
    assert_eq!(prompt_ids.len(), 45, "serve prompt_tokens for this request");
    let run = run_window_cost_tracked(&fixture, "base64", &prompt_ids, 700, true);
    let label = &run.label;
    let committed: Vec<usize> = (0..run.widths.len())
        .map(|index| run.committed(index))
        .collect();
    let (first_long, sustained) = sustained_drafting_after_budget_zero(&run.widths, &committed);
    let ordinary_median = run.ordinary_median_us();
    let summary = sustained.map(|(start, end, drafting, tokens)| {
        let us: u64 = run.window_us[start..end].iter().sum();
        let probes_before = run.probe[..start].iter().filter(|&&p| p).count();
        (
            start,
            end,
            drafting,
            tokens,
            us as f64 / tokens as f64,
            probes_before,
        )
    });
    eprintln!(
        "[qwen36-moe-dflash2 {label}] first_ordinary_run_ge8={first_long:?} ordinary_runs={:?} sustained(start,end,drafting_windows,committed_tokens,us_per_token,probes_before)={summary:?} ordinary_median_us={ordinary_median}",
        run.ordinary_runs()
    );
    assert!(
        first_long.is_some(),
        "{label}: no run of 8 ordinary windows (runs {:?})",
        run.ordinary_runs()
    );
    assert!(
        summary.is_some(),
        "{label}: no 50-window span after budget 0 with 45 drafting windows and 100 committed tokens"
    );
}

/// Qualification and formal check of the Base64 request followed by a copy
/// section: budget 0 before the copy, then recovery within the registered
/// content-change criteria.
#[test]
#[ignore = "loads a full local Qwen3.6 MoE target and the incoai DFlash2 draft"]
#[serial(mlx_metal)]
fn qwen36_moe_dflash2_window_cost_recovers_after_base64_on_copy() {
    let Some(fixture) = fixture() else {
        eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET and DFLASH2_MODEL");
        return;
    };
    let prompt_ids = chat_prompt(&fixture, &format!("{BASE64_CONTENT}{COPY_SECTION}"));
    let run = run_window_cost_tracked(&fixture, "base64-copy", &prompt_ids, 700, true);
    assert_recovers_on_copy_section(&fixture, &run);
}
