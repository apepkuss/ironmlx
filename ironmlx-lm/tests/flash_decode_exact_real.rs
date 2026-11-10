//! Opt-in bitwise digest of resident Flash greedy decode, so two builds can
//! be checked for identical output on the same prompt. Every step hashes its
//! logits; every other step also requests and hashes the hidden states a
//! DFlash2 draft taps. A run compares against the digest file, and fails if
//! the file is missing. `IRONMLX_FLASH_DECODE_DIGEST_RECORD=1` writes the
//! file instead and refuses to replace an existing one.
//!
//! Only steps whose context exceeds the QSA indexer budget build the decode
//! mask; the output line counts them as `qsa_mask_steps`.
use ironmlx_lm::{
    core::{tensor_file::ModelTensorFiles, Loader},
    models::qwen4_exp::text_model::{FlashExecutionOptions, FlashTextModel},
};
use mlx::Dtype;
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc, time::Instant};

const PREFILL_CHUNK: usize = 2048;
/// The target layers the incoai Flash-Next DFlash2 draft conditions on.
const CAPTURE_LAYERS: [usize; 5] = [5, 15, 25, 35, 45];

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT, M5 Max and at least 128 GiB RAM"]
fn real_flash_resident_decode_digest() {
    let snapshot = std::env::var("IRONMLX_FLASH_SNAPSHOT").expect("snapshot required");
    let digest_file =
        std::env::var("IRONMLX_FLASH_DECODE_DIGEST_FILE").expect("digest file required");
    let record = std::env::var("IRONMLX_FLASH_DECODE_DIGEST_RECORD").as_deref() == Ok("1");
    match (record, Path::new(&digest_file).exists()) {
        (true, true) => panic!("{digest_file} exists; record mode never replaces a digest"),
        (false, false) => {
            panic!("{digest_file} is missing; record it with IRONMLX_FLASH_DECODE_DIGEST_RECORD=1")
        }
        _ => {}
    }
    let steps = std::env::var("IRONMLX_FLASH_DECODE_STEPS")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(32);
    assert!(steps > 4, "at least five decode steps are required");
    let prompt = std::env::var("IRONMLX_FLASH_PROMPT_FILE")
        .map(|path| std::fs::read_to_string(path).unwrap())
        .unwrap_or_else(|_| {
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n".to_owned()
        });
    let tokenizer =
        tokenizers::Tokenizer::from_file(Path::new(&snapshot).join("tokenizer.json")).unwrap();
    let tokens = tokenizer
        .encode(prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(!tokens.is_empty(), "prompt must not be empty");
    let cap = tokens.len() + steps + 32;

    mlx::memory::set_cache_limit(512 * 1024 * 1024);
    mlx::memory::set_wired_limit(96usize << 30).unwrap();
    let loader = Loader::open_flash_text(Path::new(&snapshot), 82u64 << 30).unwrap();
    let files = Arc::new(ModelTensorFiles::open(Path::new(&snapshot)).unwrap());
    // Same QSA scratch envelope as the production load plan.
    let scratch = (PREFILL_CHUNK * cap * 4).max(64 * 1024 * 1024);
    let model = FlashTextModel::from_loader_with_paging(
        &loader,
        files,
        32 * 1024 * 1024,
        16 * 1024 * 1024,
        scratch,
        None,
        FlashExecutionOptions::from_environment().unwrap(),
    )
    .unwrap();
    drop(loader);

    let mut cache = model.new_cache(cap as i32).unwrap();
    // Every hashed step is a single-token decode: the last prompt token is
    // the first decode input.
    let (head, last) = tokens.split_at(tokens.len() - 1);
    for chunk in head.chunks(PREFILL_CHUNK) {
        model
            .forward_hidden_tokens_on(chunk, &mut cache, &[], ().into())
            .unwrap();
    }
    let vocab = model.config.vocab_size as usize;
    let mut hash = Sha256::new();
    let mut feed = last[0];
    let mut millis = Vec::with_capacity(steps);
    let picks = model.config.indexer_budget / model.config.indexer_compress_ratio;
    let mut mask_steps = 0;
    for step in 0..steps {
        let total = (tokens.len() + step) as i32;
        mask_steps += usize::from(total / model.config.indexer_compress_ratio > picks);
        let capture_layers: &[usize] = if step % 2 == 1 { &CAPTURE_LAYERS } else { &[] };
        let started = Instant::now();
        let output = model
            .forward_tokens_on(&[feed], &mut cache, capture_layers, ().into())
            .unwrap();
        // Timing covers plain decode steps only, after warm-up.
        if capture_layers.is_empty() && step >= 4 {
            millis.push(started.elapsed().as_secs_f64() * 1e3);
        }
        assert_eq!(output.captures.len(), capture_layers.len());
        for capture in &output.captures {
            for value in capture
                .astype(Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
            {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        let logits = output
            .logits
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let row = &logits[logits.len() - vocab..];
        let mut best = 0usize;
        for (index, &value) in row.iter().enumerate() {
            hash.update(value.to_bits().to_le_bytes());
            if value > row[best] {
                best = index;
            }
        }
        feed = best as u32;
        hash.update(feed.to_le_bytes());
    }
    let digest = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    millis.sort_by(f64::total_cmp);
    println!(
        "decode_digest={digest} prompt_tokens={} steps={steps} qsa_mask_steps={mask_steps} decode_ms_median={:.3}",
        tokens.len(),
        millis[millis.len() / 2]
    );
    if record {
        std::fs::write(&digest_file, format!("{digest}\n")).unwrap();
    } else {
        let expected = std::fs::read_to_string(&digest_file).unwrap();
        assert_eq!(expected.trim(), digest, "decode digest changed");
    }
}
