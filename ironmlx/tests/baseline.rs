//! P5e baseline / measurement infrastructure: wall-clock `Model::forward_on`
//! at PP=128/512/2048 with 1 outer-warmup call + 3 measured calls per length.
//! Each call additionally self-warms with one extra forward to drain prior
//! lazy graphs (so 8 total forward passes per PP). Output is median wall-clock
//! per length, printed via eprintln! for harvest by reports/p5e-*.md.
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --test baseline \
//!       -- --ignored --nocapture --test-threads=1
//!
//! Identical test body is reused for each Stage 1 feature experiment (T1-T3)
//! by toggling Cargo features (no test code changes).

use mlx::Dtype;
use std::time::Instant;

use ironmlx::core::generate::build_position_ids;
use ironmlx::core::{Loader, Model};
use ironmlx::models::Qwen35MoeModel;

const PROMPT_LENGTHS: [i32; 3] = [128, 512, 2048];
const RUNS: usize = 3;
const WARMUP: usize = 1;

fn locate_snapshot() -> String {
    if let Ok(p) = std::env::var("IRONMLX_MOE_MODEL_DIR") {
        return p;
    }
    let home = std::env::var("HOME").expect("HOME env");
    let glob =
        format!("{home}/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots");
    let entries = std::fs::read_dir(&glob).expect("snapshots dir");
    let first = entries
        .filter_map(|e| e.ok())
        .next()
        .expect("at least one snapshot");
    first.path().to_string_lossy().into_owned()
}

fn synth_token_ids(len: i32) -> Vec<i32> {
    // Deterministic pseudo-prompt: id = (10000 + i % 100). Stays within vocab
    // (vocab_size 248320) and produces well-defined embeddings for measurement.
    (0..len).map(|i| 10_000 + (i % 100)).collect()
}

fn run_once(model: &Qwen35MoeModel, prompt_len: i32) -> std::time::Duration {
    let ids: Vec<i32> = synth_token_ids(prompt_len);
    let input_ids: mlx::Array = (&ids[..], &[1_i32, prompt_len][..])
        .try_into()
        .expect("input_ids try_into");
    let pos = build_position_ids(0, prompt_len).expect("build_position_ids");

    let cap = prompt_len.max(ironmlx::models::qwen3_5_moe::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
    let mut cache = Model::make_cache(model, 1, cap, Dtype::Bfloat16).expect("make_cache");

    // Warmup forward + eval to drain pending lazy ops from prior calls and
    // JIT-prime kernels; result discarded.
    let warmup_logits = Model::forward_on(
        model,
        &input_ids,
        &pos,
        None,
        None,
        Some(&mut cache),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on warmup");
    mlx::transforms::eval(&[&warmup_logits]).expect("eval warmup");

    // Re-make cache so prefill runs from empty state every measurement.
    let mut cache = Model::make_cache(model, 1, cap, Dtype::Bfloat16).expect("make_cache");

    let start = Instant::now();
    let logits = Model::forward_on(
        model,
        &input_ids,
        &pos,
        None,
        None,
        Some(&mut cache),
        mlx::StreamOrDevice::default(),
    )
    .expect("forward_on");
    // Force eval to materialize all lazy ops before stopping the timer.
    mlx::transforms::eval(&[&logits]).expect("eval");
    start.elapsed()
}

#[test]
#[ignore]
fn p5e_prefill_wallclock_pp_sweep() {
    let dir = locate_snapshot();
    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");

    eprintln!(
        "[p5e_baseline] model loaded; running PP={:?}",
        PROMPT_LENGTHS
    );

    for &pp in &PROMPT_LENGTHS {
        // Warmup runs (not measured)
        for _ in 0..WARMUP {
            let _ = run_once(&model, pp);
        }

        // Measured runs
        let mut samples: Vec<f64> = (0..RUNS)
            .map(|_| run_once(&model, pp).as_secs_f64() * 1000.0)
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = samples[samples.len() / 2];
        let tok_per_s = (pp as f64) / (median_ms / 1000.0);

        eprintln!(
            "[p5e_baseline] PP={pp} runs={samples:?} median_ms={median_ms:.2} tok/s={tok_per_s:.1}",
        );
    }
}
