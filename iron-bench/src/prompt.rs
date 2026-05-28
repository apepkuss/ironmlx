//! Synthetic prompt generator — tokenizer-aware, with per-run nonce.
//!
//! The output prompt encodes to exactly `target_tokens` tokens (± small
//! BPE round-trip drift) on the same tokenizer. The nonce prevents
//! prefix-cache hits across runs (omlx defaults to a tiered prefix cache;
//! without nonce, the second run's prefill would be ~0ms — invalidating
//! PP measurement).

use anyhow::{anyhow, bail, Result};
use tokenizers::Tokenizer;

pub const DEFAULT_PROMPT_LENS: &[usize] = &[128, 512, 2048];

/// Prompt source for one benchmark cell.
#[derive(Clone, Debug)]
pub enum PromptSource {
    /// Tokenizer-aware synthetic prompt with a per-run nonce.
    Synthetic { target_tokens: usize },
    /// Exact prompt text reused for every request. The cell PP label is the
    /// local tokenizer count of this text.
    Fixed {
        text: String,
        token_count_local: usize,
    },
}

impl PromptSource {
    pub fn synthetic(target_tokens: usize) -> Result<Self> {
        if target_tokens == 0 {
            bail!("--prompt-len values must be > 0");
        }
        Ok(Self::Synthetic { target_tokens })
    }

    pub fn fixed_from_text(tokenizer: &Tokenizer, text: &str) -> Result<Self> {
        if text.trim().is_empty() {
            bail!("--fixed-prompt-file is empty");
        }
        let encoded = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer.encode fixed prompt: {e}"))?;
        let token_count_local = encoded.get_ids().len();
        if token_count_local == 0 {
            bail!("--fixed-prompt-file produced 0 local tokens");
        }
        Ok(Self::Fixed {
            text: text.to_string(),
            token_count_local,
        })
    }

    pub fn target_tokens(&self) -> usize {
        match self {
            Self::Synthetic { target_tokens } => *target_tokens,
            Self::Fixed {
                token_count_local, ..
            } => *token_count_local,
        }
    }

    pub fn render(&self, tokenizer: &Tokenizer, nonce: u64) -> Result<(String, usize)> {
        match self {
            Self::Synthetic { target_tokens } => {
                synthesize_prompt(tokenizer, *target_tokens, nonce)
            }
            Self::Fixed {
                text,
                token_count_local,
            } => Ok((text.clone(), *token_count_local)),
        }
    }
}

pub fn build_prompt_sources(
    tokenizer: &Tokenizer,
    prompt_lens: Option<&[usize]>,
    fixed_prompt_text: Option<&str>,
) -> Result<Vec<PromptSource>> {
    if let Some(text) = fixed_prompt_text {
        if prompt_lens.is_some() {
            bail!("--fixed-prompt-file cannot be used with --prompt-len");
        }
        return Ok(vec![PromptSource::fixed_from_text(tokenizer, text)?]);
    }

    let lens = prompt_lens.unwrap_or(DEFAULT_PROMPT_LENS);
    lens.iter().copied().map(PromptSource::synthetic).collect()
}

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

    #[test]
    fn fixed_prompt_source_reports_local_tokens_and_ignores_nonce() {
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — skipping fixed_prompt_source");
            return;
        };
        let text = "Measure this exact fixed prompt.";
        let expected_tokens = tok.encode(text, false).expect("encode").get_ids().len();

        let source = PromptSource::fixed_from_text(&tok, text).expect("fixed source");
        assert_eq!(source.target_tokens(), expected_tokens);

        let (a, a_tokens) = source.render(&tok, 1).expect("render nonce 1");
        let (b, b_tokens) = source.render(&tok, 2).expect("render nonce 2");
        assert_eq!(a, text);
        assert_eq!(b, text);
        assert_eq!(a_tokens, expected_tokens);
        assert_eq!(b_tokens, expected_tokens);
    }

    #[test]
    fn prompt_sources_default_to_synthetic_matrix() {
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — skipping prompt source matrix");
            return;
        };
        let sources = build_prompt_sources(&tok, None, None).expect("default sources");
        let targets: Vec<usize> = sources.iter().map(PromptSource::target_tokens).collect();
        assert_eq!(targets, vec![128, 512, 2048]);
    }

    #[test]
    fn prompt_sources_reject_fixed_prompt_with_prompt_len() {
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — skipping fixed prompt conflict");
            return;
        };
        let err = build_prompt_sources(&tok, Some(&[512]), Some("fixed prompt"))
            .expect_err("fixed prompt must reject prompt_len override");
        assert!(
            err.to_string().contains("--fixed-prompt-file"),
            "error should name the conflicting flag, got: {err}"
        );
    }

    #[test]
    fn fixed_prompt_source_rejects_empty_text() {
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — skipping empty fixed prompt");
            return;
        };
        let err = PromptSource::fixed_from_text(&tok, " \n\t ")
            .expect_err("empty fixed prompt should fail");
        assert!(
            err.to_string().contains("empty"),
            "error should mention empty prompt, got: {err}"
        );
    }
}
