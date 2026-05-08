//! Synthetic prompt generator — tokenizer-aware, with per-run nonce.
//!
//! The output prompt encodes to exactly `target_tokens` tokens (± small
//! BPE round-trip drift) on the same tokenizer. The nonce prevents
//! prefix-cache hits across runs (omlx defaults to a tiered prefix cache;
//! without nonce, the second run's prefill would be ~0ms — invalidating
//! PP measurement).

use anyhow::{anyhow, bail, Result};
use tokenizers::Tokenizer;

/// Synthesize a prompt that encodes to (approximately) `target_tokens` tokens.
///
/// Returns `(prompt_text, actual_token_count_local)`. The actual count is the
/// post-round-trip local tokenizer count; small BPE drift (±2 tokens) is
/// tolerated. Authoritative server-side count comes from the response
/// `usage.prompt_tokens` field if available.
pub fn synthesize_prompt(
    tokenizer: &Tokenizer,
    target_tokens: usize,
    nonce: u64,
) -> Result<(String, usize)> {
    if target_tokens == 0 {
        bail!("synthesize_prompt: target_tokens must be > 0");
    }
    let unique_prefix = format!("Benchmark request {nonce} —");
    // ~10 tokens per filler chunk for any reasonable BPE; overshoot then truncate.
    let filler = " The quick brown fox jumps over the lazy dog.";
    let approx_filler_count = target_tokens.max(10) + 8;
    let text = format!("{unique_prefix}{}", filler.repeat(approx_filler_count));

    let encoded = tokenizer
        .encode(&text[..], false)
        .map_err(|e| anyhow!("tokenizer.encode: {e}"))?;
    let ids = encoded.get_ids();
    if ids.len() < target_tokens {
        bail!(
            "synthesize_prompt: filler tokenized to {} tokens; need >= {target_tokens}. \
             Increase filler size.",
            ids.len()
        );
    }
    let truncated_ids = &ids[..target_tokens];
    let decoded = tokenizer
        .decode(truncated_ids, false)
        .map_err(|e| anyhow!("tokenizer.decode: {e}"))?;

    // Round-trip sanity: re-encode and report actual count.
    let reencoded = tokenizer
        .encode(&decoded[..], false)
        .map_err(|e| anyhow!("tokenizer.encode (verify): {e}"))?;
    let actual_tokens = reencoded.get_ids().len();

    Ok((decoded, actual_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_test_tokenizer() -> Option<Tokenizer> {
        let path = std::env::var("IRON_BENCH_TEST_TOKENIZER").ok()?;
        Tokenizer::from_file(path).ok()
    }

    #[test]
    fn synth_round_trip_target_lengths() {
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — skipping synth_round_trip");
            return;
        };
        for target in [32_usize, 128, 512, 2048] {
            let (text, actual) = synthesize_prompt(&tok, target, 42).expect("synth ok");
            assert!(
                actual.abs_diff(target) <= 2,
                "target={target}, actual={actual}, text len={}",
                text.len()
            );
        }
    }

    #[test]
    fn synth_zero_target_errors() {
        // No tokenizer needed — early-returns on 0 before touching tokenizer.
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — running with stub still ok");
            // We can't construct a Tokenizer without a file; just verify error path
            // by NOT running the assertion. The 0-check is the first statement of
            // synthesize_prompt, before any tokenizer.encode call.
            return;
        };
        let r = synthesize_prompt(&tok, 0, 0);
        assert!(r.is_err(), "target_tokens=0 must return Err");
    }
}
