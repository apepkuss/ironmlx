//! Server-side output reservation for validated native reasoning dialects.
//!
//! The budget is a decoding constraint, not text post-processing. Delimiters
//! are sampled, committed to the model cache, and charged to the same output
//! limit as all other generated tokens.

use std::{collections::VecDeque, sync::Arc};

use crate::core::native_output::{
    NativeOutputDecoderConfig, NativeOutputDialect, NativeOutputParser,
};
use crate::Result;
use llguidance::{api::TopLevelGrammar, toktrie::SimpleVob, Matcher, ParserFactory};

/// Automatic policy for one response. This is a capacity partition, not a
/// guarantee that the reserved space suffices for a correct or complete answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningBudget {
    pub reasoning_tokens: usize,
    pub answer_reserve: usize,
    pub framing_reserve: usize,
}

impl ReasoningBudget {
    /// Reserve a quarter of the output budget (at most 1024 tokens) for the
    /// answer/tool call. Use a byte-wise upper bound for native framing plus
    /// UTF-8 boundary slack, so alternate delimiter tokenizations also fit.
    /// Tiny requests retain their existing truncation semantics.
    pub fn automatic(dialect: NativeOutputDialect, total: usize) -> Option<Self> {
        let framing_reserve = match dialect {
            NativeOutputDialect::Qwen35
            | NativeOutputDialect::Qwen36
            | NativeOutputDialect::Qwen38 => "</think>\n\n".len() + 4,
            NativeOutputDialect::Gemma => "<|channel>thought\n<channel|>".len() + 4,
            _ => return None,
        };
        let answer_reserve = (total / 4).min(1024);
        if answer_reserve == 0 {
            return None;
        }
        let reasoning_tokens = total.checked_sub(answer_reserve + framing_reserve)?;
        (reasoning_tokens > 0).then_some(Self {
            reasoning_tokens,
            answer_reserve,
            framing_reserve,
        })
    }

    fn grammar(self, dialect: NativeOutputDialect) -> String {
        let (start, close, open) = match dialect {
            NativeOutputDialect::Gemma => (
                "start: (\"<|channel>thought\\n\" thought)? ANSWER\n",
                "<channel|>",
                "<|channel>",
            ),
            _ => ("start: thought ANSWER\n", "</think>", "<think>"),
        };
        // A lazy suffix finishes thought at the first native closing marker.
        // Excluding the opening delimiter from answer prevents Gemma's optional thought
        // branch from being bypassed and prevents reasoning from reopening.
        let close = serde_json::to_string(close).expect("literal");
        let open = serde_json::to_string(open).expect("literal");
        format!(
            "{start}thought[suffix={close}]: /(?s:.*)/\nANSWER: /(?s:.*)/ & ~(/(?s:.*)/ ({open} | {close}) /(?s:.*)/)\n"
        )
    }
}

pub(crate) struct ReasoningBudgetPlan {
    factory: Arc<ParserFactory>,
    grammar: TopLevelGrammar,
    dialect: NativeOutputDialect,
    budget: ReasoningBudget,
}

impl ReasoningBudgetPlan {
    pub(crate) fn new(
        factory: Arc<ParserFactory>,
        dialect: NativeOutputDialect,
        budget: ReasoningBudget,
    ) -> Result<Arc<Self>> {
        let plan = Arc::new(Self {
            factory,
            grammar: TopLevelGrammar::from_lark(budget.grammar(dialect)),
            dialect,
            budget,
        });
        plan.start_session()?;
        Ok(plan)
    }

    pub(crate) fn start_session(self: &Arc<Self>) -> Result<ReasoningBudgetSession> {
        let matcher = Matcher::new(self.factory.create_parser(self.grammar.clone()));
        anyhow::ensure!(
            !matcher.is_error(),
            "compile reasoning budget: {:?}",
            matcher.get_error()
        );
        Ok(ReasoningBudgetSession {
            plan: Arc::clone(self),
            matcher,
            forced_tokens: None,
            parser: NativeOutputParser::new(NativeOutputDecoderConfig {
                dialect: self.dialect,
                reasoning_enabled: true,
            }),
            utf8_pending: Vec::new(),
            tokens: Vec::new(),
            reasoning_tokens: 0,
        })
    }
}

pub(crate) struct ReasoningBudgetSession {
    plan: Arc<ReasoningBudgetPlan>,
    matcher: Matcher,
    forced_tokens: Option<VecDeque<u32>>,
    parser: NativeOutputParser,
    utf8_pending: Vec<u8>,
    tokens: Vec<u32>,
    reasoning_tokens: usize,
}

impl Clone for ReasoningBudgetSession {
    fn clone(&self) -> Self {
        Self {
            plan: Arc::clone(&self.plan),
            matcher: self.matcher.deep_clone(),
            forced_tokens: self.forced_tokens.clone(),
            parser: self.parser.clone(),
            utf8_pending: self.utf8_pending.clone(),
            tokens: self.tokens.clone(),
            reasoning_tokens: self.reasoning_tokens,
        }
    }
}

impl ReasoningBudgetSession {
    fn finishing_utf8(&self) -> bool {
        self.reasoning_tokens >= self.plan.budget.reasoning_tokens
            && !self.utf8_pending.is_empty()
            && self.parser.reasoning_close_suffix().is_some()
    }

    fn permits_utf8_completion(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }
        let mut joined = self.utf8_pending.clone();
        joined.extend_from_slice(bytes);
        match std::str::from_utf8(&joined) {
            Ok(_) => true,
            // At most three continuation bytes may finish the current code
            // point. Do not let a token start another partial code point and
            // indefinitely postpone the forced native close.
            Err(error) => error.error_len().is_none() && error.valid_up_to() == 0,
        }
    }

    fn force_if_needed(&mut self) -> Result<()> {
        if self.forced_tokens.is_some()
            || self.reasoning_tokens < self.plan.budget.reasoning_tokens
            || !self.utf8_pending.is_empty()
        {
            return Ok(());
        }
        let Some(suffix) = self.parser.reasoning_close_suffix() else {
            return Ok(());
        };
        let suffix = match self.plan.dialect {
            NativeOutputDialect::Gemma => suffix.to_owned(),
            _ => format!("{suffix}\n\n"),
        };
        // Prefer the whole native delimiter token, not a sampled spelling of
        // the same bytes. Qwen's added </think> token and ordinary '<', '/',
        // 'think' pieces need not have the same learned effect on the model.
        // Longest-match also completes an already-emitted partial delimiter.
        let env = self.matcher.tok_env()?;
        let tokens = env.tok_trie().greedy_tokenize(suffix.as_bytes());
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|&token| env.tok_trie().token(token).iter().copied())
            .collect();
        anyhow::ensure!(
            bytes == suffix.as_bytes(),
            "native reasoning close is not representable by the tokenizer"
        );
        self.forced_tokens = Some(tokens.into());
        Ok(())
    }

    pub(crate) fn compute_mask(&mut self) -> Result<SimpleVob> {
        self.force_if_needed()?;
        let mut mask = self.matcher.compute_mask_or_eos()?;
        if self.finishing_utf8() {
            let env = self.matcher.tok_env()?;
            let mut rejected = Vec::new();
            mask.iter_set_entries(|token| {
                if !self.permits_utf8_completion(env.tok_trie().token(token as u32)) {
                    rejected.push(token as u32);
                }
            });
            for token in rejected {
                mask.disallow_token(token);
            }
        }
        if let Some(&next) = self
            .forced_tokens
            .as_ref()
            .and_then(|tokens| tokens.front())
        {
            let allowed = mask.is_allowed(next);
            mask.set_all(false);
            if allowed {
                mask.allow_token(next);
            }
        }
        Ok(mask)
    }

    pub(crate) fn commit_token(&mut self, token: u32) -> Result<()> {
        self.force_if_needed()?;
        if self.finishing_utf8() {
            let env = self.matcher.tok_env()?;
            anyhow::ensure!(
                self.permits_utf8_completion(env.tok_trie().token(token)),
                "token postponed the reasoning UTF-8 boundary"
            );
        }
        let was_reasoning = self.parser.reasoning_close_suffix().is_some();
        if let Some(tokens) = &mut self.forced_tokens {
            if let Some(&next) = tokens.front() {
                anyhow::ensure!(token == next, "token bypassed native reasoning close");
                tokens.pop_front();
            }
        }
        self.matcher.consume_token(token)?;
        let env = self.matcher.tok_env()?;
        self.utf8_pending
            .extend_from_slice(env.tok_trie().token(token));
        let valid = match std::str::from_utf8(&self.utf8_pending) {
            Ok(text) => text.len(),
            Err(error) => {
                anyhow::ensure!(
                    error.error_len().is_none(),
                    "invalid UTF-8 in reasoning budget stream"
                );
                error.valid_up_to()
            }
        };
        if valid > 0 {
            self.parser
                .push(std::str::from_utf8(&self.utf8_pending[..valid])?)?;
            self.utf8_pending.drain(..valid);
        }
        self.tokens.push(token);
        if was_reasoning {
            self.reasoning_tokens += 1;
        }
        Ok(())
    }

    pub(crate) fn is_accepting(&mut self) -> Result<bool> {
        Ok(self.matcher.is_accepting()?
            && self.forced_tokens.as_ref().is_none_or(VecDeque::is_empty))
    }

    pub(crate) fn rollback(&mut self, count: usize) -> Result<()> {
        anyhow::ensure!(
            count <= self.tokens.len(),
            "reasoning budget rollback exceeds committed tokens"
        );
        if count == 0 {
            return Ok(());
        }
        // Replaying committed IDs reconstructs both the channel parser and the
        // optional forced suffix; draft forks never mutate their parent state.
        let mut restored = self.plan.start_session()?;
        for &token in &self.tokens[..self.tokens.len() - count] {
            restored.commit_token(token)?;
        }
        *self = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_utf8_cannot_postpone_closure_with_another_partial_code_point() {
        use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};

        let mut bytes = (0_u16..=255).map(|b| vec![b as u8]).collect::<Vec<_>>();
        bytes.push(b"\xff<eos>".to_vec());
        bytes.push(b"\x80\xe6\x80".to_vec());
        let trie = TokTrie::from(&TokRxInfo::new(bytes.len() as u32, 256), &bytes);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env).unwrap();
        factory.quiet();
        let plan = ReasoningBudgetPlan::new(
            Arc::new(factory),
            NativeOutputDialect::Qwen35,
            ReasoningBudget {
                reasoning_tokens: 8,
                answer_reserve: 32,
                framing_reserve: 14,
            },
        )
        .unwrap();
        let mut session = plan.start_session().unwrap();
        for &b in b"abcdef\xe6\x80" {
            session.commit_token(u32::from(b)).unwrap();
        }
        let mask = session.compute_mask().unwrap();
        assert!(mask.is_allowed(128));
        assert!(!mask.is_allowed(257));
        assert!(session.clone().commit_token(257).is_err());
        session.commit_token(128).unwrap();
        let mask = session.compute_mask().unwrap();
        assert_eq!(mask.num_set(), 1);
        assert!(mask.is_allowed(u32::from(b'<')));
        session.rollback(1).unwrap();
        assert!(!session.compute_mask().unwrap().is_allowed(257));
    }

    #[test]
    fn partition_fits_total_and_never_enlarges_small_requests() {
        for dialect in [NativeOutputDialect::Qwen35, NativeOutputDialect::Gemma] {
            for total in 0..8193 {
                if let Some(budget) = ReasoningBudget::automatic(dialect, total) {
                    assert!(budget.reasoning_tokens > 0);
                    assert!(budget.answer_reserve > 0 && budget.answer_reserve <= 1024);
                    assert_eq!(
                        budget.reasoning_tokens + budget.answer_reserve + budget.framing_reserve,
                        total
                    );
                }
            }
            assert!(ReasoningBudget::automatic(dialect, 8).is_none());
        }
        assert!(ReasoningBudget::automatic(NativeOutputDialect::Glm, 8192).is_none());
    }
}
