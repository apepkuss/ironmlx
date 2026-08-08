//! Token-level constrained decoding plans and request-local matcher state.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};
use llguidance::{token_bytes_from_tokenizer_json, Matcher, ParserFactory};
use serde_json::{Map, Value};

use mlx::Array;

use crate::core::tool_calling::{ToolDefinition, GEMMA_STRING_DELIMITER};
use crate::Result;

const MAX_SCHEMA_DEPTH: usize = 5;
const MAX_SCHEMA_NODES: usize = 256;
const MAX_TOP_LEVEL_PARAMETERS: usize = 64;

/// Request-level tool selection semantics compiled into one grammar plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoiceConstraint {
    Auto,
    Required,
    Function(String),
}

/// Controls whether plain text and multiple calls are legal for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolConstraintOptions {
    pub choice: ToolChoiceConstraint,
    pub allow_parallel_calls: bool,
}

impl Default for ToolConstraintOptions {
    fn default() -> Self {
        Self {
            choice: ToolChoiceConstraint::Auto,
            allow_parallel_calls: true,
        }
    }
}

/// Tokenizer-level state shared by every constrained request for one model.
pub struct ConstraintTokenizer {
    factory: Arc<ParserFactory>,
    vocab_size: usize,
    eos_token_ids: Arc<[u32]>,
}

impl fmt::Debug for ConstraintTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintTokenizer")
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl ConstraintTokenizer {
    pub fn from_tokenizer_json(
        tokenizer_json: &Value,
        eos_token_ids: &[u32],
        ordinary_special_tokens: &[&str],
    ) -> Result<Self> {
        let mut token_bytes = token_bytes_from_tokenizer_json(tokenizer_json)
            .context("build constrained-decoding token byte table")?;
        for token in ordinary_special_tokens {
            let matches = tokenizer_json["added_tokens"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|entry| {
                    entry.get("content").and_then(Value::as_str) == Some(*token)
                        && entry.get("special").and_then(Value::as_bool) == Some(true)
                })
                .collect::<Vec<_>>();
            anyhow::ensure!(
                matches.len() == 1,
                "constraint token `{token}` must resolve to exactly one special added token"
            );
            let token_id = matches[0]
                .get("id")
                .and_then(Value::as_u64)
                .and_then(|id| usize::try_from(id).ok())
                .ok_or_else(|| anyhow!("constraint token `{token}` has an invalid token id"))?;
            let bytes = token_bytes.get_mut(token_id).ok_or_else(|| {
                anyhow!("constraint token `{token}` is outside tokenizer vocabulary")
            })?;
            anyhow::ensure!(
                bytes.first() == Some(&llguidance::toktrie::TokTrie::SPECIAL_TOKEN_MARKER),
                "constraint token `{token}` is not encoded as a special token"
            );
            *bytes = token.as_bytes().to_vec();
        }
        let eos = eos_token_ids
            .first()
            .copied()
            .ok_or_else(|| anyhow!("constrained decoding requires at least one EOS token id"))?;
        anyhow::ensure!(
            (eos as usize) < token_bytes.len(),
            "EOS token {eos} is outside tokenizer vocabulary {}",
            token_bytes.len()
        );
        let info = TokRxInfo::new(token_bytes.len() as u32, eos);
        let trie = TokTrie::from(&info, &token_bytes).with_eos_tokens(eos_token_ids);
        let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&tok_env)
            .context("initialize constrained-decoding parser factory")?;
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
            vocab_size: token_bytes.len(),
            eos_token_ids: Arc::from(eos_token_ids),
        })
    }

    #[cfg(test)]
    pub(crate) fn byte_level() -> Result<Self> {
        let mut token_bytes = (0_u16..=255)
            .map(|byte| vec![byte as u8])
            .collect::<Vec<_>>();
        token_bytes.push(b"\xFF<eos>".to_vec());
        let eos = 256_u32;
        let info = TokRxInfo::new(token_bytes.len() as u32, eos);
        let trie = TokTrie::from(&info, &token_bytes);
        let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&tok_env)?;
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
            vocab_size: token_bytes.len(),
            eos_token_ids: Arc::from([eos]),
        })
    }

    #[cfg(test)]
    pub(crate) fn byte_level_gemma() -> Result<Self> {
        let mut token_bytes = (0_u16..=255)
            .map(|byte| vec![byte as u8])
            .collect::<Vec<_>>();
        token_bytes.push(b"<|tool_call>".to_vec());
        token_bytes.push(b"<tool_call|>".to_vec());
        token_bytes.push(GEMMA_STRING_DELIMITER.as_bytes().to_vec());
        token_bytes.push(b"<|channel>".to_vec());
        token_bytes.push(b"<channel|>".to_vec());
        token_bytes.push(b"\xFF<eos>".to_vec());
        let eos = 261_u32;
        let info = TokRxInfo::new(token_bytes.len() as u32, eos);
        let trie = TokTrie::from(&info, &token_bytes).with_eos_tokens(&[eos]);
        let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&tok_env)?;
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
            vocab_size: token_bytes.len(),
            eos_token_ids: Arc::from([eos]),
        })
    }

    pub fn compile_qwen_tools(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
    ) -> Result<ConstraintPlan> {
        self.compile_qwen_tools_with_output(tools, options, None)
    }

    pub fn compile_qwen_tools_with_output(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
        output_schema: Option<&Value>,
    ) -> Result<ConstraintPlan> {
        validate_tool_schemas(tools)?;
        if let Some(schema) = output_schema {
            validate_constraint_output_schema(schema)?;
        }
        let grammar_source = build_qwen_tool_grammar(tools, options, output_schema)?;
        let grammar = TopLevelGrammar::from_lark(grammar_source.clone());
        let matcher = Matcher::new(self.factory.create_parser(grammar.clone()));
        if matcher.is_error() {
            bail!(
                "compile constrained tool grammar: {}",
                matcher
                    .get_error()
                    .unwrap_or_else(|| "unknown parser error".to_string())
            );
        }
        Ok(ConstraintPlan {
            factory: Arc::clone(&self.factory),
            grammar,
            grammar_source: Arc::from(grammar_source),
            vocab_size: self.vocab_size,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        })
    }

    pub fn compile_gemma_tools(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
    ) -> Result<ConstraintPlan> {
        self.compile_gemma_tools_with_output(tools, options, None)
    }

    pub fn compile_gemma_tools_with_output(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
        output_schema: Option<&Value>,
    ) -> Result<ConstraintPlan> {
        validate_tool_schemas(tools)?;
        if let Some(schema) = output_schema {
            validate_constraint_output_schema(schema)?;
        }
        let grammar_source = build_gemma_tool_grammar(tools, options, output_schema)?;
        let grammar = TopLevelGrammar::from_lark(grammar_source.clone());
        let matcher = Matcher::new(self.factory.create_parser(grammar.clone()));
        if matcher.is_error() {
            bail!(
                "compile constrained Gemma tool grammar: {}",
                matcher
                    .get_error()
                    .unwrap_or_else(|| "unknown parser error".to_string())
            );
        }
        Ok(ConstraintPlan {
            factory: Arc::clone(&self.factory),
            grammar,
            grammar_source: Arc::from(grammar_source),
            vocab_size: self.vocab_size,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        })
    }

    pub fn compile_glm_tools(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
    ) -> Result<ConstraintPlan> {
        self.compile_glm_tools_with_output(tools, options, None)
    }

    pub fn compile_glm_tools_with_output(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
        output_schema: Option<&Value>,
    ) -> Result<ConstraintPlan> {
        validate_tool_schemas(tools)?;
        if let Some(schema) = output_schema {
            validate_constraint_output_schema(schema)?;
        }
        let grammar_source = build_glm_tool_grammar(tools, options, output_schema)?;
        let grammar = TopLevelGrammar::from_lark(grammar_source.clone());
        let matcher = Matcher::new(self.factory.create_parser(grammar.clone()));
        if matcher.is_error() {
            bail!(
                "compile constrained GLM tool grammar: {}",
                matcher
                    .get_error()
                    .unwrap_or_else(|| "unknown parser error".to_string())
            );
        }
        Ok(ConstraintPlan {
            factory: Arc::clone(&self.factory),
            grammar,
            grammar_source: Arc::from(grammar_source),
            vocab_size: self.vocab_size,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        })
    }

    pub fn compile_llama_tools(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
    ) -> Result<ConstraintPlan> {
        self.compile_llama_tools_with_output(tools, options, None)
    }

    pub fn compile_llama_tools_with_output(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
        output_schema: Option<&Value>,
    ) -> Result<ConstraintPlan> {
        validate_tool_schemas(tools)?;
        if let Some(schema) = output_schema {
            validate_constraint_output_schema(schema)?;
        }
        let grammar_source = build_llama_tool_grammar(tools, options, output_schema)?;
        let grammar = TopLevelGrammar::from_lark(grammar_source.clone());
        let matcher = Matcher::new(self.factory.create_parser(grammar.clone()));
        if matcher.is_error() {
            bail!(
                "compile constrained Llama tool grammar: {}",
                matcher
                    .get_error()
                    .unwrap_or_else(|| "unknown parser error".to_string())
            );
        }
        Ok(ConstraintPlan {
            factory: Arc::clone(&self.factory),
            grammar,
            grammar_source: Arc::from(grammar_source),
            vocab_size: self.vocab_size,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        })
    }

    pub fn compile_minicpm5_tools(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
    ) -> Result<ConstraintPlan> {
        self.compile_minicpm5_tools_with_output(tools, options, None)
    }

    pub fn compile_minicpm5_tools_with_output(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
        output_schema: Option<&Value>,
    ) -> Result<ConstraintPlan> {
        validate_tool_schemas(tools)?;
        if let Some(schema) = output_schema {
            validate_constraint_output_schema(schema)?;
        }
        let grammar_source = build_minicpm5_tool_grammar(tools, options, output_schema)?;
        let grammar = TopLevelGrammar::from_lark(grammar_source.clone());
        let matcher = Matcher::new(self.factory.create_parser(grammar.clone()));
        if matcher.is_error() {
            bail!(
                "compile constrained MiniCPM5 tool grammar: {}",
                matcher
                    .get_error()
                    .unwrap_or_else(|| "unknown parser error".to_string())
            );
        }
        Ok(ConstraintPlan {
            factory: Arc::clone(&self.factory),
            grammar,
            grammar_source: Arc::from(grammar_source),
            vocab_size: self.vocab_size,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        })
    }

    /// Compile a standalone JSON output constraint for Responses structured
    /// output. The schema is validated by IronMLX before llguidance sees it so
    /// request-time and post-generation validation use the same subset.
    pub fn compile_json_output(&self, schema: &Value) -> Result<ConstraintPlan> {
        validate_constraint_output_schema(schema)?;
        let grammar = TopLevelGrammar::from_json_schema(schema.clone());
        let matcher = Matcher::new(self.factory.create_parser(grammar.clone()));
        if matcher.is_error() {
            bail!(
                "compile constrained JSON output: {}",
                matcher
                    .get_error()
                    .unwrap_or_else(|| "unknown parser error".to_string())
            );
        }
        Ok(ConstraintPlan {
            factory: Arc::clone(&self.factory),
            grammar,
            grammar_source: Arc::from(serde_json::to_string(schema)?),
            vocab_size: self.vocab_size,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        })
    }
}

/// Immutable grammar plan carried by a generation request.
#[derive(Clone)]
pub struct ConstraintPlan {
    factory: Arc<ParserFactory>,
    grammar: TopLevelGrammar,
    grammar_source: Arc<str>,
    vocab_size: usize,
    eos_token_ids: Arc<[u32]>,
}

impl fmt::Debug for ConstraintPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintPlan")
            .field("vocab_size", &self.vocab_size)
            .field("grammar_bytes", &self.grammar_source.len())
            .finish_non_exhaustive()
    }
}

impl ConstraintPlan {
    pub fn start_session(&self) -> Result<ConstraintSession> {
        let matcher = Matcher::new(self.factory.create_parser(self.grammar.clone()));
        if matcher.is_error() {
            bail!(
                "start constrained-decoding matcher: {}",
                matcher
                    .get_error()
                    .unwrap_or_else(|| "unknown parser error".to_string())
            );
        }
        Ok(ConstraintSession {
            matcher,
            vocab_size: self.vocab_size,
            committed_tokens: 0,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        })
    }

    #[cfg(test)]
    fn grammar_source(&self) -> &str {
        &self.grammar_source
    }
}

/// Mutable grammar state owned by one generation request.
pub struct ConstraintSession {
    matcher: Matcher,
    vocab_size: usize,
    committed_tokens: usize,
    eos_token_ids: Arc<[u32]>,
}

impl Clone for ConstraintSession {
    fn clone(&self) -> Self {
        Self {
            matcher: self.matcher.deep_clone(),
            vocab_size: self.vocab_size,
            committed_tokens: self.committed_tokens,
            eos_token_ids: Arc::clone(&self.eos_token_ids),
        }
    }
}

impl fmt::Debug for ConstraintSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintSession")
            .field("vocab_size", &self.vocab_size)
            .field("committed_tokens", &self.committed_tokens)
            .field("stopped", &self.matcher.is_stopped())
            .finish()
    }
}

impl ConstraintSession {
    pub fn fork(&self) -> Self {
        self.clone()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn compute_mask(&mut self) -> Result<SimpleVob> {
        let mask = self
            .matcher
            .compute_mask_or_eos()
            .context("compute constrained token mask")?;
        anyhow::ensure!(
            mask.num_set() > 0,
            "constrained decoding reached a state with no valid next token"
        );
        Ok(mask)
    }

    /// Compute target masks for positions conditioned on a speculative draft
    /// prefix. An invalid draft token is never advanced; later masks remain at
    /// the mismatch state because those target positions cannot be committed.
    pub fn speculative_masks(&self, draft_tokens: &[u32]) -> Result<Vec<SimpleVob>> {
        let mut scratch = self.fork();
        let mut masks = Vec::with_capacity(draft_tokens.len() + 1);
        let mut mismatched = false;
        for token in draft_tokens {
            let mask = scratch.compute_mask()?;
            let allowed = mask.is_allowed(*token);
            masks.push(mask);
            if !mismatched && allowed {
                scratch.commit_token(*token)?;
            } else {
                mismatched = true;
            }
        }
        masks.push(scratch.compute_mask()?);
        Ok(masks)
    }

    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        anyhow::ensure!(
            (token as usize) < self.vocab_size,
            "constraint token {token} is outside vocabulary {}",
            self.vocab_size
        );
        if self.eos_token_ids.contains(&token) {
            anyhow::ensure!(
                self.is_accepting()?,
                "cannot commit EOS token {token} before constrained output is complete"
            );
            return Ok(());
        }
        self.matcher
            .consume_token(token)
            .with_context(|| format!("commit constrained token {token}"))?;
        self.committed_tokens = self.committed_tokens.saturating_add(1);
        Ok(())
    }

    pub fn is_accepting(&mut self) -> Result<bool> {
        self.matcher
            .is_accepting()
            .context("query constrained-decoding accepting state")
    }

    pub fn validate_tokens(&self, tokens: &[u32]) -> Result<usize> {
        let mut scratch = self.fork();
        for (index, token) in tokens.iter().copied().enumerate() {
            let mask = scratch.compute_mask()?;
            if !mask.is_allowed(token) {
                return Ok(index);
            }
            scratch
                .commit_token(token)
                .context("validate speculative token against constraint")?;
            if scratch.eos_token_ids.contains(&token) {
                return Ok(index + 1);
            }
        }
        Ok(tokens.len())
    }

    /// Keep a speculative resolution constraint-safe before it enters a pending queue.
    /// Draft tokens are masked individually, so only the final target correction or bonus
    /// may be invalid when an earlier token completes the grammar.
    pub fn truncate_invalid_speculative_bonus(&self, tokens: &mut Vec<u32>) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let valid_prefix = self.validate_tokens(tokens)?;
        anyhow::ensure!(
            valid_prefix >= tokens.len().saturating_sub(1),
            "speculative resolution violates the decoding constraint before its final token"
        );
        tokens.truncate(valid_prefix);
        anyhow::ensure!(
            !tokens.is_empty(),
            "speculative resolution contains no constraint-valid token"
        );
        Ok(())
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        for token in tokens {
            self.commit_token(*token)
                .context("commit constrained token sequence")?;
        }
        Ok(())
    }

    pub fn rollback(&mut self, tokens: usize) -> Result<()> {
        anyhow::ensure!(
            tokens <= self.committed_tokens,
            "cannot roll constraint back by {tokens} after {} committed tokens",
            self.committed_tokens
        );
        self.matcher
            .rollback(tokens)
            .context("roll back constrained-decoding matcher")?;
        self.committed_tokens -= tokens;
        Ok(())
    }
}

/// Apply a request-local token mask to one vocabulary logit vector.
pub fn apply_token_mask(logits: &Array, mask: &SimpleVob) -> Result<Array> {
    let shape = logits.shape();
    anyhow::ensure!(
        shape.as_slice().len() == 1,
        "constraint logits must be 1-D [vocab], got {:?}",
        shape.as_slice()
    );
    let vocab = shape.as_slice()[0] as usize;
    anyhow::ensure!(
        mask.len() <= vocab,
        "constraint mask vocabulary {} exceeds logits vocabulary {vocab}",
        mask.len()
    );
    let mut additive = vec![f32::NEG_INFINITY; vocab];
    for (token, value) in additive.iter_mut().enumerate().take(mask.len()) {
        if mask.is_allowed(token as u32) {
            *value = 0.0;
        }
    }
    let additive: Array = (&additive[..], &[vocab as i32][..]).try_into()?;
    let additive = mlx::ops::cast::astype(&additive, logits.dtype())?;
    mlx::ops::binary::add(logits, &additive).map_err(anyhow::Error::from)
}

/// Apply independent token masks to a compact `[B, vocab]` logit batch.
pub fn apply_batch_token_masks(logits: &Array, masks: &[Option<SimpleVob>]) -> Result<Array> {
    let shape = logits.shape();
    anyhow::ensure!(
        shape.as_slice().len() == 2,
        "constraint batch logits must be 2-D [B, vocab], got {:?}",
        shape.as_slice()
    );
    let batch = shape.as_slice()[0] as usize;
    let vocab = shape.as_slice()[1] as usize;
    anyhow::ensure!(
        masks.len() == batch,
        "constraint mask count {} does not match batch {batch}",
        masks.len()
    );
    if masks.iter().all(Option::is_none) {
        return Ok(logits.clone());
    }
    let mut additive = vec![0.0_f32; batch * vocab];
    for (row, mask) in masks.iter().enumerate() {
        let Some(mask) = mask else {
            continue;
        };
        anyhow::ensure!(
            mask.len() <= vocab,
            "row {row} constraint mask vocabulary {} exceeds logits vocabulary {vocab}",
            mask.len()
        );
        for token in 0..mask.len() {
            if !mask.is_allowed(token as u32) {
                additive[row * vocab + token] = f32::NEG_INFINITY;
            }
        }
        for token in mask.len()..vocab {
            additive[row * vocab + token] = f32::NEG_INFINITY;
        }
    }
    let additive: Array = (&additive[..], &[batch as i32, vocab as i32][..]).try_into()?;
    let additive = mlx::ops::cast::astype(&additive, logits.dtype())?;
    mlx::ops::binary::add(logits, &additive).map_err(anyhow::Error::from)
}

/// Apply per-row, per-position speculative target masks to `[B, S, vocab]`.
pub fn apply_speculative_token_masks(
    logits: &Array,
    masks: &[Option<Vec<SimpleVob>>],
) -> Result<Array> {
    let shape = logits.shape();
    anyhow::ensure!(
        shape.as_slice().len() == 3,
        "speculative logits must be 3-D [B, S, vocab], got {:?}",
        shape.as_slice()
    );
    let batch = shape.as_slice()[0] as usize;
    let steps = shape.as_slice()[1] as usize;
    let vocab = shape.as_slice()[2] as usize;
    anyhow::ensure!(masks.len() == batch, "speculative mask batch mismatch");
    if masks.iter().all(Option::is_none) {
        return Ok(logits.clone());
    }
    let mut additive = vec![0.0_f32; batch * steps * vocab];
    for (row, row_masks) in masks.iter().enumerate() {
        let Some(row_masks) = row_masks else {
            continue;
        };
        anyhow::ensure!(
            row_masks.len() <= steps,
            "row {row} has {} speculative masks for {steps} steps",
            row_masks.len()
        );
        for (step, mask) in row_masks.iter().enumerate() {
            anyhow::ensure!(
                mask.len() <= vocab,
                "row {row} step {step} constraint mask vocabulary {} exceeds logits vocabulary {vocab}",
                mask.len()
            );
            for token in 0..mask.len() {
                if !mask.is_allowed(token as u32) {
                    additive[(row * steps + step) * vocab + token] = f32::NEG_INFINITY;
                }
            }
            for token in mask.len()..vocab {
                additive[(row * steps + step) * vocab + token] = f32::NEG_INFINITY;
            }
        }
    }
    let additive: Array = (
        &additive[..],
        &[batch as i32, steps as i32, vocab as i32][..],
    )
        .try_into()?;
    let additive = mlx::ops::cast::astype(&additive, logits.dtype())?;
    mlx::ops::binary::add(logits, &additive).map_err(anyhow::Error::from)
}

/// Apply prefix-conditioned masks for one block-diffusion canvas.
///
/// Unlike speculative masks, every position is present and belongs to one
/// request. The caller must derive the masks sequentially from a fork of the
/// committed grammar state; this function only transfers that legal path to
/// the device-resident `[1, L, vocab]` logits.
pub fn apply_diffusion_token_masks(logits: &Array, masks: &[SimpleVob]) -> Result<Array> {
    let shape = logits.shape();
    anyhow::ensure!(
        shape.as_slice().len() == 3 && shape.as_slice()[0] == 1,
        "diffusion constraint logits must be [1, L, vocab], got {:?}",
        shape.as_slice()
    );
    let steps = shape.as_slice()[1] as usize;
    let vocab = shape.as_slice()[2] as usize;
    anyhow::ensure!(
        masks.len() == steps,
        "diffusion constraint mask count {} does not match canvas length {steps}",
        masks.len()
    );
    let mut additive = vec![f32::NEG_INFINITY; steps * vocab];
    for (step, mask) in masks.iter().enumerate() {
        anyhow::ensure!(
            mask.len() <= vocab,
            "diffusion step {step} mask vocabulary {} exceeds logits vocabulary {vocab}",
            mask.len()
        );
        for token in 0..mask.len() {
            if mask.is_allowed(token as u32) {
                additive[step * vocab + token] = 0.0;
            }
        }
    }
    let additive: Array = (&additive[..], &[1_i32, steps as i32, vocab as i32][..]).try_into()?;
    let additive = mlx::ops::cast::astype(&additive, logits.dtype())?;
    mlx::ops::binary::add(logits, &additive).map_err(anyhow::Error::from)
}

pub fn validate_tool_schemas(tools: &[ToolDefinition]) -> Result<()> {
    for tool in tools {
        let root = tool
            .parameters
            .as_object()
            .ok_or_else(|| anyhow!("parameters must be a JSON object schema"))?;
        if root.get("type").and_then(Value::as_str) != Some("object") {
            bail!("parameters.type must be `object`");
        }
        let properties = root
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("parameters.properties must be an object"))?;
        if properties.len() > MAX_TOP_LEVEL_PARAMETERS {
            bail!(
                "tool `{}` has {} parameters; constrained decoding supports at most {MAX_TOP_LEVEL_PARAMETERS}",
                tool.name,
                properties.len()
            );
        }
        let mut nodes = 0_usize;
        validate_schema_node(&tool.parameters, 0, &mut nodes)
            .with_context(|| format!("unsupported schema for tool `{}`", tool.name))?;
    }
    Ok(())
}

/// Validate one Responses `json_schema` output contract against IronMLX's
/// bounded constrained-decoding subset. OpenAI structured outputs require a
/// root object; strict mode additionally applies the same recursive object
/// rules as strict function tools.
pub fn validate_json_output_schema(schema: &Value, strict: bool) -> Result<()> {
    let root = schema
        .as_object()
        .ok_or_else(|| anyhow!("structured output schema must be a JSON object"))?;
    anyhow::ensure!(
        root.get("type").and_then(Value::as_str) == Some("object"),
        "structured output schema root type must be `object`"
    );
    anyhow::ensure!(
        root.get("properties").is_some(),
        "structured output schema root must declare properties"
    );
    let mut nodes = 0_usize;
    validate_schema_node(schema, 0, &mut nodes).context("unsupported structured output schema")?;
    if strict {
        validate_strict_schema_node(schema, "$structured_output")?;
    }
    Ok(())
}

fn validate_constraint_output_schema(schema: &Value) -> Result<()> {
    if schema.as_object().is_some_and(|object| {
        object.len() == 1 && object.get("type") == Some(&Value::String("object".to_owned()))
    }) {
        return Ok(());
    }
    validate_json_output_schema(schema, false)
}

/// Validate the additional JSON Schema rules required by OpenAI strict mode.
pub fn validate_strict_tool_schema(tool: &ToolDefinition) -> Result<()> {
    validate_strict_schema_node(&tool.parameters, "$").map_err(|error| {
        anyhow!(
            "strict schema for tool `{}` is invalid: {error:#}",
            tool.name
        )
    })
}

fn validate_strict_schema_node(schema: &Value, path: &str) -> Result<()> {
    let object = schema
        .as_object()
        .ok_or_else(|| anyhow!("schema node at `{path}` must be an object"))?;
    if type_contains(object.get("type"), "object") || object.contains_key("properties") {
        anyhow::ensure!(
            object.get("additionalProperties") == Some(&Value::Bool(false)),
            "object at `{path}` must set additionalProperties=false"
        );
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("object at `{path}` must declare properties"))?;
        let required = object
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().expect("schema validated"))
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        for name in properties.keys() {
            anyhow::ensure!(
                required.contains(name.as_str()),
                "property `{path}.{name}` must be listed in required"
            );
        }
    }
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        for (name, child) in properties {
            validate_strict_schema_node(child, &format!("{path}.{name}"))?;
        }
    }
    if let Some(items) = object.get("items") {
        validate_strict_schema_node(items, &format!("{path}[]"))?;
    }
    if let Some(branches) = object.get("anyOf").and_then(Value::as_array) {
        for (index, branch) in branches.iter().enumerate() {
            validate_strict_schema_node(branch, &format!("{path}.anyOf[{index}]"))?;
        }
    }
    Ok(())
}

fn validate_schema_node(schema: &Value, depth: usize, nodes: &mut usize) -> Result<()> {
    anyhow::ensure!(
        depth <= MAX_SCHEMA_DEPTH,
        "schema nesting exceeds {MAX_SCHEMA_DEPTH} levels"
    );
    *nodes = nodes.saturating_add(1);
    anyhow::ensure!(
        *nodes <= MAX_SCHEMA_NODES,
        "schema exceeds {MAX_SCHEMA_NODES} nodes"
    );
    let object = schema
        .as_object()
        .ok_or_else(|| anyhow!("schema node must be an object"))?;
    const ALLOWED: &[&str] = &[
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "description",
        "title",
        "default",
        "examples",
    ];
    for key in object.keys() {
        if !ALLOWED.contains(&key.as_str()) {
            bail!("unsupported JSON Schema keyword `{key}`");
        }
    }
    if let Some(kind) = object.get("type") {
        validate_type_keyword(kind)?;
    }
    if let Some(any_of) = object.get("anyOf") {
        let branches = any_of
            .as_array()
            .ok_or_else(|| anyhow!("anyOf must be an array"))?;
        anyhow::ensure!(!branches.is_empty(), "anyOf must not be empty");
        for branch in branches {
            validate_schema_node(branch, depth + 1, nodes)?;
        }
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or_else(|| anyhow!("properties must be an object"))?;
        for (name, property) in properties {
            validate_parameter_name(name)?;
            validate_schema_node(property, depth + 1, nodes)?;
        }
        validate_required(object, properties)?;
    } else if object.contains_key("required") {
        bail!("required requires properties");
    }
    if let Some(additional) = object.get("additionalProperties") {
        anyhow::ensure!(
            additional == &Value::Bool(false),
            "additionalProperties only supports false"
        );
    }
    if let Some(items) = object.get("items") {
        validate_schema_node(items, depth + 1, nodes)?;
    }
    if type_contains(object.get("type"), "array") && !object.contains_key("items") {
        bail!("array schemas require items");
    }
    if type_contains(object.get("type"), "object") && !object.contains_key("properties") {
        bail!("object schemas require properties");
    }
    if let Some(values) = object.get("enum") {
        let values = values
            .as_array()
            .ok_or_else(|| anyhow!("enum must be an array"))?;
        anyhow::ensure!(!values.is_empty(), "enum must not be empty");
    }
    if !object.contains_key("type")
        && !object.contains_key("anyOf")
        && !object.contains_key("enum")
        && !object.contains_key("const")
    {
        bail!("schema node requires type, anyOf, enum, or const");
    }
    Ok(())
}

fn validate_type_keyword(kind: &Value) -> Result<()> {
    let mut kinds = Vec::new();
    match kind {
        Value::String(kind) => kinds.push(kind.as_str()),
        Value::Array(values) => {
            anyhow::ensure!(!values.is_empty(), "type array must not be empty");
            for value in values {
                kinds.push(
                    value
                        .as_str()
                        .ok_or_else(|| anyhow!("type array entries must be strings"))?,
                );
            }
        }
        _ => bail!("type must be a string or string array"),
    }
    let mut seen = HashSet::new();
    for kind in kinds {
        anyhow::ensure!(
            matches!(
                kind,
                "string" | "integer" | "number" | "boolean" | "object" | "array" | "null"
            ),
            "unsupported schema type `{kind}`"
        );
        anyhow::ensure!(seen.insert(kind), "duplicate schema type `{kind}`");
    }
    Ok(())
}

fn validate_required(object: &Map<String, Value>, properties: &Map<String, Value>) -> Result<()> {
    let Some(required) = object.get("required") else {
        return Ok(());
    };
    let required = required
        .as_array()
        .ok_or_else(|| anyhow!("required must be an array"))?;
    let mut seen = HashSet::new();
    for value in required {
        let name = value
            .as_str()
            .ok_or_else(|| anyhow!("required entries must be strings"))?;
        anyhow::ensure!(
            properties.contains_key(name),
            "required property `{name}` is not declared"
        );
        anyhow::ensure!(seen.insert(name), "duplicate required property `{name}`");
    }
    Ok(())
}

fn validate_parameter_name(name: &str) -> Result<()> {
    anyhow::ensure!(
        !name.is_empty() && name.len() <= 256 && !name.chars().any(char::is_control),
        "invalid parameter name"
    );
    Ok(())
}

fn type_contains(kind: Option<&Value>, expected: &str) -> bool {
    match kind {
        Some(Value::String(kind)) => kind == expected,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(expected)),
        _ => false,
    }
}

fn build_qwen_tool_grammar(
    tools: &[ToolDefinition],
    options: &ToolConstraintOptions,
    output_schema: Option<&Value>,
) -> Result<String> {
    let selected_indexes = match &options.choice {
        ToolChoiceConstraint::Function(name) => vec![tools
            .iter()
            .position(|tool| tool.name == *name)
            .ok_or_else(|| anyhow!("tool_choice references unknown function `{name}`"))?],
        ToolChoiceConstraint::Auto | ToolChoiceConstraint::Required => (0..tools.len()).collect(),
    };
    let allows_multiple = options.allow_parallel_calls
        && !matches!(&options.choice, ToolChoiceConstraint::Function(_));

    let mut grammar = String::new();
    match (&options.choice, output_schema) {
        (ToolChoiceConstraint::Auto, Some(_)) => {
            grammar.push_str("start: structured_output | first_call more_calls ws\n")
        }
        (ToolChoiceConstraint::Auto, None) => {
            grammar.push_str("start: first_call more_calls ws | TEXT\n")
        }
        (ToolChoiceConstraint::Required | ToolChoiceConstraint::Function(_), _) => {
            grammar.push_str("start: first_call more_calls ws\n");
        }
    }
    append_structured_output_rule(&mut grammar, output_schema)?;
    let first_text = if output_schema.is_some() {
        "WS?"
    } else {
        "TEXT"
    };
    grammar.push_str(&format!(
        "first_call: first_head function_dispatch\nfirst_head[lazy]: {first_text} \"<tool_call>\"\n"
    ));
    if allows_multiple {
        grammar.push_str(
            "more_calls: next_call*\n\
             next_call: ws \"<tool_call>\" function_dispatch\n",
        );
    } else {
        grammar.push_str("more_calls: \"\"\n");
    }
    grammar.push_str("function_dispatch: ");
    for (position, index) in selected_indexes.into_iter().enumerate() {
        if position > 0 {
            grammar.push_str(" | ");
        }
        grammar.push_str(&format!("ws function_{index}"));
    }
    grammar.push('\n');

    let mut next_tagged_rule = 0_usize;
    for (tool_index, tool) in tools.iter().enumerate() {
        let root = tool.parameters.as_object().expect("schema validated");
        let properties = root["properties"].as_object().expect("schema validated");
        if properties.is_empty() {
            let function_open = lark_literal(&format!("<function={}>", tool.name))?;
            grammar.push_str(&format!(
                "function_{tool_index}: {function_open} ws \"</function>\" ws \"</tool_call>\"\n"
            ));
            continue;
        }
        let required = root
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().expect("schema validated"))
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        let function_open = lark_literal(&format!("<function={}>", tool.name))?;
        grammar.push_str(&format!(
            "function_{tool_index}: {function_open} params_{tool_index}::0x0 ws \"</function>\" ws \"</tool_call>\"\n"
        ));
        grammar.push_str(&format!("params_{tool_index}::_ : "));
        if required.is_empty() {
            grammar.push_str("\"\"");
        } else {
            let mut condition = required
                .iter()
                .map(|name| {
                    let bit = properties
                        .keys()
                        .position(|candidate| candidate == *name)
                        .expect("required property exists");
                    format!("bit_set({bit})")
                })
                .collect::<Vec<_>>();
            condition.sort();
            grammar.push_str(&format!("\"\" %if {}", and_condition(&condition)));
        }
        grammar.push('\n');
        let mut value_rules = Vec::with_capacity(properties.len());
        for (parameter_index, (name, schema)) in properties.iter().enumerate() {
            let open = lark_literal(&format!("<parameter={name}>"))?;
            let value_rule = format!("value_{tool_index}_{parameter_index}");
            grammar.push_str(&format!(
                "    | ws {open} {value_rule} params_{tool_index}::set_bit({parameter_index}) %if bit_clear({parameter_index})\n"
            ));
            value_rules.push((value_rule, schema));
        }
        for (value_rule, schema) in value_rules {
            append_value_rule(&mut grammar, &value_rule, schema, &mut next_tagged_rule)?;
        }
    }
    grammar.push_str(
        "native_line: \"\\r\\n\" | \"\\n\"\n\
         ws: WS?\nWS: /[ \\t\\r\\n]+/\nTEXT: /(.|\\n)*/\n",
    );
    Ok(grammar)
}

fn build_gemma_tool_grammar(
    tools: &[ToolDefinition],
    options: &ToolConstraintOptions,
    output_schema: Option<&Value>,
) -> Result<String> {
    let selected_indexes = selected_tool_indexes(tools, options)?;
    let allows_multiple = options.allow_parallel_calls
        && !matches!(&options.choice, ToolChoiceConstraint::Function(_));
    let mut grammar = String::new();
    match (&options.choice, output_schema) {
        (ToolChoiceConstraint::Auto, Some(_)) => {
            grammar.push_str("start: structured_output | first_call more_calls ws\n")
        }
        (ToolChoiceConstraint::Auto, None) => {
            grammar.push_str("start: first_call more_calls ws | TEXT\n")
        }
        (ToolChoiceConstraint::Required | ToolChoiceConstraint::Function(_), _) => {
            grammar.push_str("start: first_call more_calls ws\n");
        }
    }
    append_structured_output_rule(&mut grammar, output_schema)?;
    let first_text = if output_schema.is_some() {
        "WS?"
    } else {
        "TEXT"
    };
    grammar.push_str(&format!(
        "first_call: first_head function_dispatch\nfirst_head[lazy]: {first_text} \"<|tool_call>\"\n"
    ));
    if allows_multiple {
        grammar.push_str(
            "more_calls: next_call*\n\
             next_call: ws \"<|tool_call>\" function_dispatch\n",
        );
    } else {
        grammar.push_str("more_calls: \"\"\n");
    }
    grammar.push_str("function_dispatch: ");
    for (position, index) in selected_indexes.into_iter().enumerate() {
        if position > 0 {
            grammar.push_str(" | ");
        }
        grammar.push_str(&format!("ws function_{index}"));
    }
    grammar.push('\n');

    let mut next_rule = 0_usize;
    for (tool_index, tool) in tools.iter().enumerate() {
        let arguments_rule = format!("gemma_args_{tool_index}");
        grammar.push_str(&format!(
            "function_{tool_index}: {} {arguments_rule} ws \"<tool_call|>\"\n",
            lark_literal(&format!("call:{}", tool.name))?
        ));
        append_gemma_schema_rule(
            &mut grammar,
            &arguments_rule,
            &tool.parameters,
            &mut next_rule,
        )?;
    }
    grammar.push_str("ws: WS?\nWS: /[ \\t\\r\\n]+/\nTEXT: /(.|\\n)*/\n");
    Ok(grammar)
}

fn build_glm_tool_grammar(
    tools: &[ToolDefinition],
    options: &ToolConstraintOptions,
    output_schema: Option<&Value>,
) -> Result<String> {
    let selected_indexes = selected_tool_indexes(tools, options)?;
    let allows_multiple = options.allow_parallel_calls
        && !matches!(&options.choice, ToolChoiceConstraint::Function(_));
    let mut grammar = String::new();
    match (&options.choice, output_schema) {
        (ToolChoiceConstraint::Auto, Some(_)) => {
            grammar.push_str("start: structured_output | first_call more_calls\n")
        }
        (ToolChoiceConstraint::Auto, None) => {
            grammar.push_str("start: first_call more_calls | TEXT\n")
        }
        (ToolChoiceConstraint::Required | ToolChoiceConstraint::Function(_), _) => {
            grammar.push_str("start: first_call more_calls\n");
        }
    }
    append_structured_output_rule(&mut grammar, output_schema)?;
    let first_text = if output_schema.is_some() {
        "WS?"
    } else {
        "TEXT"
    };
    grammar.push_str(&format!(
        "first_call: first_head function_dispatch\nfirst_head[lazy]: {first_text} \"<tool_call>\"\n"
    ));
    if allows_multiple {
        grammar.push_str(
            "more_calls: next_call*\n\
             next_call: \"<tool_call>\" function_dispatch\n",
        );
    } else {
        grammar.push_str("more_calls: \"\"\n");
    }
    grammar.push_str("function_dispatch: ");
    for (position, index) in selected_indexes.into_iter().enumerate() {
        if position > 0 {
            grammar.push_str(" | ");
        }
        grammar.push_str(&format!("function_{index}"));
    }
    grammar.push('\n');

    let mut next_tagged_rule = 0_usize;
    for (tool_index, tool) in tools.iter().enumerate() {
        let root = tool.parameters.as_object().expect("schema validated");
        let properties = root["properties"].as_object().expect("schema validated");
        let required = root
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().expect("schema validated"))
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        grammar.push_str(&format!(
            "function_{tool_index}: {} glm_params_{tool_index}::0x0 \"</tool_call>\"\n",
            lark_literal(&tool.name)?
        ));
        grammar.push_str(&format!("glm_params_{tool_index}::_ : "));
        if required.is_empty() {
            grammar.push_str("\"\"");
        } else {
            let mut conditions = required
                .iter()
                .map(|name| {
                    let bit = properties
                        .keys()
                        .position(|candidate| candidate == *name)
                        .expect("required property exists");
                    format!("bit_set({bit})")
                })
                .collect::<Vec<_>>();
            conditions.sort();
            grammar.push_str(&format!("\"\" %if {}", and_condition(&conditions)));
        }
        grammar.push('\n');

        let mut value_rules = Vec::with_capacity(properties.len());
        for (argument_index, (name, schema)) in properties.iter().enumerate() {
            let key = lark_literal(&format!("<arg_key>{name}</arg_key>"))?;
            let value_rule = format!("glm_value_{tool_index}_{argument_index}");
            grammar.push_str(&format!(
                "    | {key} \"<arg_value>\" {value_rule} glm_params_{tool_index}::set_bit({argument_index}) %if bit_clear({argument_index})\n"
            ));
            value_rules.push((value_rule, schema));
        }
        for (value_rule, schema) in value_rules {
            append_tagged_value_rule(
                &mut grammar,
                &value_rule,
                schema,
                "</arg_value>",
                false,
                &mut next_tagged_rule,
            )?;
        }
    }
    grammar.push_str("ws: WS?\nWS: /[ \\t\\r\\n]+/\nTEXT: /(.|\\n)*/\n");
    Ok(grammar)
}

fn build_llama_tool_grammar(
    tools: &[ToolDefinition],
    options: &ToolConstraintOptions,
    output_schema: Option<&Value>,
) -> Result<String> {
    let selected_indexes = selected_tool_indexes(tools, options)?;
    let mut grammar = String::new();
    match (&options.choice, output_schema) {
        (ToolChoiceConstraint::Auto, Some(_)) => {
            grammar.push_str("start: structured_output | ws function_dispatch ws\n")
        }
        (ToolChoiceConstraint::Auto, None) => {
            grammar.push_str("start: ws function_dispatch ws | PLAIN_TEXT\n")
        }
        (ToolChoiceConstraint::Required | ToolChoiceConstraint::Function(_), _) => {
            grammar.push_str("start: ws function_dispatch ws\n")
        }
    }
    append_structured_output_rule(&mut grammar, output_schema)?;
    grammar.push_str("function_dispatch: ");
    for (position, index) in selected_indexes.into_iter().enumerate() {
        if position > 0 {
            grammar.push_str(" | ");
        }
        grammar.push_str(&format!("function_{index}"));
    }
    grammar.push('\n');

    for (index, tool) in tools.iter().enumerate() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"const": tool.name},
                "parameters": tool.parameters,
            },
            "required": ["name", "parameters"],
            "additionalProperties": false,
        });
        grammar.push_str(&format!(
            "function_{index}: %json {}\n",
            serde_json::to_string(&schema)?
        ));
    }
    grammar.push_str(
        "ws: WS?\n\
         WS: /[ \\t\\r\\n]+/\n\
         PLAIN_TEXT: /[ \\t\\r\\n]*([^{ \\t\\r\\n](.|\\n)*)?/\n",
    );
    Ok(grammar)
}

fn build_minicpm5_tool_grammar(
    tools: &[ToolDefinition],
    options: &ToolConstraintOptions,
    output_schema: Option<&Value>,
) -> Result<String> {
    let selected_indexes = selected_tool_indexes(tools, options)?;
    let allows_multiple = options.allow_parallel_calls
        && !matches!(&options.choice, ToolChoiceConstraint::Function(_));
    let mut grammar = String::new();
    match (&options.choice, output_schema) {
        (ToolChoiceConstraint::Auto, Some(_)) => {
            grammar.push_str("start: structured_output | first_call more_calls ws\n")
        }
        (ToolChoiceConstraint::Auto, None) => {
            grammar.push_str("start: first_call more_calls ws | TEXT\n")
        }
        (ToolChoiceConstraint::Required | ToolChoiceConstraint::Function(_), _) => {
            grammar.push_str("start: first_call more_calls ws\n");
        }
    }
    append_structured_output_rule(&mut grammar, output_schema)?;
    let first_text = if output_schema.is_some() {
        "WS?"
    } else {
        "TEXT"
    };
    grammar.push_str(&format!(
        "first_call: first_head function_dispatch\nfirst_head[lazy]: {first_text} "
    ));
    grammar.push_str(&lark_literal("<function name=\"")?);
    grammar.push('\n');
    if allows_multiple {
        grammar.push_str("more_calls: next_call*\nnext_call: ws ");
        grammar.push_str(&lark_literal("<function name=\"")?);
        grammar.push_str(" function_dispatch\n");
    } else {
        grammar.push_str("more_calls: \"\"\n");
    }
    grammar.push_str("function_dispatch: ");
    for (position, index) in selected_indexes.into_iter().enumerate() {
        if position > 0 {
            grammar.push_str(" | ");
        }
        grammar.push_str(&format!("function_{index}"));
    }
    grammar.push('\n');

    let mut next_tagged_rule = 0_usize;
    for (tool_index, tool) in tools.iter().enumerate() {
        let root = tool.parameters.as_object().expect("schema validated");
        let properties = root["properties"].as_object().expect("schema validated");
        if properties.is_empty() {
            grammar.push_str(&format!(
                "function_{tool_index}: {} ws \"</function>\"\n",
                lark_literal(&format!("{}\">", tool.name))?
            ));
            continue;
        }
        let required = root
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().expect("schema validated"))
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        grammar.push_str(&format!(
            "function_{tool_index}: {} minicpm5_params_{tool_index}::0x0 ws \"</function>\"\n",
            lark_literal(&format!("{}\">", tool.name))?
        ));
        grammar.push_str(&format!("minicpm5_params_{tool_index}::_ : "));
        if required.is_empty() {
            grammar.push_str("\"\"");
        } else {
            let mut conditions = required
                .iter()
                .map(|name| {
                    let bit = properties
                        .keys()
                        .position(|candidate| candidate == *name)
                        .expect("required property exists");
                    format!("bit_set({bit})")
                })
                .collect::<Vec<_>>();
            conditions.sort();
            grammar.push_str(&format!("\"\" %if {}", and_condition(&conditions)));
        }
        grammar.push('\n');

        let mut value_rules = Vec::with_capacity(properties.len());
        for (parameter_index, (name, schema)) in properties.iter().enumerate() {
            let open = lark_literal(&format!("<param name=\"{name}\">"))?;
            let value_rule = format!("minicpm5_value_{tool_index}_{parameter_index}");
            grammar.push_str(&format!(
                "    | ws {open} {value_rule} minicpm5_params_{tool_index}::set_bit({parameter_index}) %if bit_clear({parameter_index})\n"
            ));
            value_rules.push((value_rule, schema));
        }
        for (value_rule, schema) in value_rules {
            append_minicpm5_value_rule(&mut grammar, &value_rule, schema, &mut next_tagged_rule)?;
        }
    }
    grammar.push_str("ws: WS?\nWS: /[ \\t\\r\\n]+/\nTEXT: /(.|\\n)*/\n");
    Ok(grammar)
}

fn selected_tool_indexes(
    tools: &[ToolDefinition],
    options: &ToolConstraintOptions,
) -> Result<Vec<usize>> {
    match &options.choice {
        ToolChoiceConstraint::Function(name) => Ok(vec![tools
            .iter()
            .position(|tool| tool.name == *name)
            .ok_or_else(|| anyhow!("tool_choice references unknown function `{name}`"))?]),
        ToolChoiceConstraint::Auto | ToolChoiceConstraint::Required => {
            Ok((0..tools.len()).collect())
        }
    }
}

fn append_structured_output_rule(
    grammar: &mut String,
    output_schema: Option<&Value>,
) -> Result<()> {
    if let Some(schema) = output_schema {
        grammar.push_str(&format!(
            "structured_output: %json {}\n",
            serde_json::to_string(schema)?
        ));
    }
    Ok(())
}

fn append_gemma_schema_rule(
    grammar: &mut String,
    rule: &str,
    schema: &Value,
    next_rule: &mut usize,
) -> Result<()> {
    let object = schema.as_object().expect("schema validated");
    if let Some(branches) = object.get("anyOf").and_then(Value::as_array) {
        let mut child_rules = Vec::with_capacity(branches.len());
        for branch in branches {
            let child = fresh_gemma_rule(next_rule);
            append_gemma_schema_rule(grammar, &child, branch, next_rule)?;
            child_rules.push(child);
        }
        grammar.push_str(&format!("{rule}: {}\n", child_rules.join(" | ")));
        return Ok(());
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        append_gemma_constants(grammar, rule, values)?;
        return Ok(());
    }
    if let Some(value) = object.get("const") {
        append_gemma_constants(grammar, rule, std::slice::from_ref(value))?;
        return Ok(());
    }

    let kinds = match object.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds
            .iter()
            .map(|kind| kind.as_str().expect("schema validated"))
            .collect(),
        _ => unreachable!("schema validated"),
    };
    if kinds.len() > 1 {
        let mut child_rules = Vec::with_capacity(kinds.len());
        for kind in kinds {
            let child = fresh_gemma_rule(next_rule);
            let mut branch = object.clone();
            branch.insert("type".to_owned(), Value::String(kind.to_owned()));
            append_gemma_schema_rule(grammar, &child, &Value::Object(branch), next_rule)?;
            child_rules.push(child);
        }
        grammar.push_str(&format!("{rule}: {}\n", child_rules.join(" | ")));
        return Ok(());
    }

    match kinds[0] {
        "string" => {
            let body_rule = fresh_gemma_rule(next_rule);
            let delimiter = lark_literal(GEMMA_STRING_DELIMITER)?;
            grammar.push_str(&format!(
                "{rule}: {delimiter} {body_rule}\n{body_rule}[lazy]: /(.|\\n)*/ {delimiter}\n"
            ));
        }
        "integer" => grammar.push_str(&format!("{rule}: /-?(0|[1-9][0-9]*)/\n")),
        "number" => grammar.push_str(&format!(
            "{rule}: /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\n"
        )),
        "boolean" => grammar.push_str(&format!("{rule}: \"true\" | \"false\"\n")),
        "null" => grammar.push_str(&format!("{rule}: \"null\"\n")),
        "array" => {
            let item_rule = fresh_gemma_rule(next_rule);
            append_gemma_schema_rule(
                grammar,
                &item_rule,
                object.get("items").expect("schema validated"),
                next_rule,
            )?;
            grammar.push_str(&format!(
                "{rule}: \"[\" ws ({item_rule} (ws \",\" ws {item_rule})*)? ws \"]\"\n"
            ));
        }
        "object" => append_gemma_object_rule(grammar, rule, object, next_rule)?,
        _ => unreachable!("schema validated"),
    }
    Ok(())
}

fn append_gemma_constants(grammar: &mut String, rule: &str, values: &[Value]) -> Result<()> {
    grammar.push_str(&format!("{rule}: "));
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            grammar.push_str(" | ");
        }
        match value {
            Value::String(value) => grammar.push_str(&format!(
                "{} {} {}",
                lark_literal(GEMMA_STRING_DELIMITER)?,
                lark_literal(value)?,
                lark_literal(GEMMA_STRING_DELIMITER)?
            )),
            _ => grammar.push_str(&lark_literal(&serde_json::to_string(value)?)?),
        }
    }
    grammar.push('\n');
    Ok(())
}

fn append_gemma_object_rule(
    grammar: &mut String,
    rule: &str,
    schema: &Map<String, Value>,
    next_rule: &mut usize,
) -> Result<()> {
    let properties = schema["properties"].as_object().expect("schema validated");
    if properties.is_empty() {
        grammar.push_str(&format!("{rule}: \"{{\" ws \"}}\"\n"));
        return Ok(());
    }
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| value.as_str().expect("schema validated"))
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    let first_rule = fresh_gemma_rule(next_rule);
    let tail_rule = fresh_gemma_rule(next_rule);
    grammar.push_str(&format!(
        "{rule}: \"{{\" ws {first_rule}::0x0 ws \"}}\"\n{first_rule}::_ : "
    ));
    if required.is_empty() {
        grammar.push_str("\"\"");
    }
    let mut value_rules = Vec::with_capacity(properties.len());
    for (index, (name, child_schema)) in properties.iter().enumerate() {
        if required.is_empty() || index > 0 {
            grammar.push_str(" | ");
        }
        let value_rule = fresh_gemma_rule(next_rule);
        grammar.push_str(&format!(
            "{} ws {value_rule} {tail_rule}::set_bit({index})",
            lark_literal(&format!("{name}:"))?
        ));
        value_rules.push((value_rule, child_schema));
    }
    grammar.push('\n');
    let conditions = required
        .iter()
        .map(|name| {
            let bit = properties
                .keys()
                .position(|candidate| candidate == *name)
                .expect("required property exists");
            format!("bit_set({bit})")
        })
        .collect::<Vec<_>>();
    grammar.push_str(&format!(
        "{tail_rule}::_ : \"\" %if {}\n",
        and_condition(&conditions)
    ));
    for (index, (name, _)) in properties.iter().enumerate() {
        grammar.push_str(&format!(
            "    | ws \",\" ws {} ws {} {tail_rule}::set_bit({index}) %if bit_clear({index})\n",
            lark_literal(&format!("{name}:"))?,
            value_rules[index].0,
        ));
    }
    for (value_rule, child_schema) in value_rules {
        append_gemma_schema_rule(grammar, &value_rule, child_schema, next_rule)?;
    }
    Ok(())
}

fn fresh_gemma_rule(next_rule: &mut usize) -> String {
    let rule = format!("gemma_value_{}", *next_rule);
    *next_rule += 1;
    rule
}

fn and_condition(conditions: &[String]) -> String {
    match conditions {
        [only] => only.clone(),
        [first, rest @ ..] => format!("and({first}, {})", and_condition(rest)),
        [] => "true".to_string(),
    }
}

fn append_value_rule(
    grammar: &mut String,
    rule: &str,
    schema: &Value,
    next_rule: &mut usize,
) -> Result<()> {
    append_tagged_value_rule(grammar, rule, schema, "</parameter>", true, next_rule)
}

fn append_tagged_value_rule(
    grammar: &mut String,
    rule: &str,
    schema: &Value,
    close_marker: &str,
    allow_native_line_padding: bool,
    next_rule: &mut usize,
) -> Result<()> {
    let object = schema.as_object().expect("schema validated");
    if let Some(branches) = object.get("anyOf").and_then(Value::as_array) {
        let mut child_rules = Vec::with_capacity(branches.len());
        for branch in branches {
            let child = fresh_tagged_rule(next_rule);
            append_tagged_value_rule(
                grammar,
                &child,
                branch,
                close_marker,
                allow_native_line_padding,
                next_rule,
            )?;
            child_rules.push(child);
        }
        grammar.push_str(&format!("{rule}: {}\n", child_rules.join(" | ")));
        return Ok(());
    }
    if let Some(kinds) = object.get("type").and_then(Value::as_array) {
        if kinds.len() > 1 {
            let mut child_rules = Vec::with_capacity(kinds.len());
            for kind in kinds {
                let child = fresh_tagged_rule(next_rule);
                let mut branch = object.clone();
                branch.insert("type".to_owned(), kind.clone());
                append_tagged_value_rule(
                    grammar,
                    &child,
                    &Value::Object(branch),
                    close_marker,
                    allow_native_line_padding,
                    next_rule,
                )?;
                child_rules.push(child);
            }
            grammar.push_str(&format!("{rule}: {}\n", child_rules.join(" | ")));
            return Ok(());
        }
    }
    let close = lark_literal(close_marker)?;
    if schema_is_string_only(schema) {
        if let Some(values) = raw_string_constants(schema)? {
            grammar.push_str(&format!("{rule}: "));
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    grammar.push_str(" | ");
                }
                if allow_native_line_padding {
                    grammar.push_str(&format!(
                        "native_line? {} native_line? {close}",
                        lark_literal(value)?
                    ));
                } else {
                    grammar.push_str(&format!("{} {close}", lark_literal(value)?));
                }
            }
            grammar.push('\n');
        } else {
            grammar.push_str(&format!("{rule}[lazy]: /(.|\\n)*/ {close}\n"));
        }
    } else {
        if allow_native_line_padding {
            grammar.push_str(&format!(
                "{rule}: native_line? %json {} native_line? {close}\n",
                serde_json::to_string(schema)?
            ));
        } else {
            grammar.push_str(&format!(
                "{rule}: %json {} {close}\n",
                serde_json::to_string(schema)?
            ));
        }
    }
    Ok(())
}

fn append_minicpm5_value_rule(
    grammar: &mut String,
    rule: &str,
    schema: &Value,
    next_rule: &mut usize,
) -> Result<()> {
    let object = schema.as_object().expect("schema validated");
    if let Some(branches) = object.get("anyOf").and_then(Value::as_array) {
        let mut child_rules = Vec::with_capacity(branches.len());
        for branch in branches {
            let child = fresh_tagged_rule(next_rule);
            append_minicpm5_value_rule(grammar, &child, branch, next_rule)?;
            child_rules.push(child);
        }
        grammar.push_str(&format!("{rule}: {}\n", child_rules.join(" | ")));
        return Ok(());
    }
    if let Some(kinds) = object.get("type").and_then(Value::as_array) {
        if kinds.len() > 1 {
            let mut child_rules = Vec::with_capacity(kinds.len());
            for kind in kinds {
                let child = fresh_tagged_rule(next_rule);
                let mut branch = object.clone();
                branch.insert("type".to_owned(), kind.clone());
                append_minicpm5_value_rule(grammar, &child, &Value::Object(branch), next_rule)?;
                child_rules.push(child);
            }
            grammar.push_str(&format!("{rule}: {}\n", child_rules.join(" | ")));
            return Ok(());
        }
    }

    let close = lark_literal("</param>")?;
    if schema_is_string_only(schema) {
        if let Some(values) = raw_string_constants(schema)? {
            grammar.push_str(&format!("{rule}: "));
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    grammar.push_str(" | ");
                }
                let encoded = if value
                    .chars()
                    .any(|character| matches!(character, '<' | '&' | '\r' | '\n'))
                {
                    anyhow::ensure!(
                        !value.contains("]]>"),
                        "MiniCPM5 string constant contains the CDATA terminator"
                    );
                    format!("<![CDATA[{value}]]>")
                } else {
                    value.clone()
                };
                grammar.push_str(&format!("{} {close}", lark_literal(&encoded)?));
            }
            grammar.push('\n');
        } else {
            let cdata_rule = fresh_tagged_rule(next_rule);
            grammar.push_str(&format!(
                "{rule}: /[^<\\r\\n&]*/ {close} | \"<![CDATA[\" {cdata_rule}\n\
                 {cdata_rule}[lazy]: /(.|\\n)*/ \"]]>\" {close}\n"
            ));
        }
    } else {
        grammar.push_str(&format!(
            "{rule}: %json {} {close}\n",
            serde_json::to_string(schema)?
        ));
    }
    Ok(())
}

fn fresh_tagged_rule(next_rule: &mut usize) -> String {
    let rule = format!("tagged_value_{}", *next_rule);
    *next_rule += 1;
    rule
}

pub fn schema_is_string_only(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    if let Some(any_of) = object.get("anyOf").and_then(Value::as_array) {
        return !any_of.is_empty() && any_of.iter().all(schema_is_string_only);
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        return !values.is_empty() && values.iter().all(Value::is_string);
    }
    if let Some(value) = object.get("const") {
        return value.is_string();
    }
    matches!(object.get("type"), Some(Value::String(kind)) if kind == "string")
        || matches!(object.get("type"), Some(Value::Array(kinds)) if !kinds.is_empty() && kinds.iter().all(|kind| kind.as_str() == Some("string")))
}

pub fn schema_accepts_string(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    if let Some(any_of) = object.get("anyOf").and_then(Value::as_array) {
        return any_of.iter().any(schema_accepts_string);
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        return values.iter().any(Value::is_string);
    }
    if let Some(value) = object.get("const") {
        return value.is_string();
    }
    matches!(object.get("type"), Some(Value::String(kind)) if kind == "string")
        || matches!(object.get("type"), Some(Value::Array(kinds)) if kinds.iter().any(|kind| kind.as_str() == Some("string")))
}

fn raw_string_constants(schema: &Value) -> Result<Option<Vec<String>>> {
    let object = schema.as_object().expect("schema validated");
    if let Some(any_of) = object.get("anyOf").and_then(Value::as_array) {
        let mut result = Vec::new();
        for branch in any_of {
            let Some(values) = raw_string_constants(branch)? else {
                return Ok(None);
            };
            result.extend(values);
        }
        result.sort();
        result.dedup();
        return Ok(Some(result));
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        return Ok(Some(
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| anyhow!("string enum contains non-string value"))
                })
                .collect::<Result<Vec<_>>>()?,
        ));
    }
    if let Some(value) = object.get("const") {
        return Ok(Some(vec![value
            .as_str()
            .ok_or_else(|| anyhow!("string const contains non-string value"))?
            .to_owned()]));
    }
    Ok(None)
}

fn lark_literal(value: &str) -> Result<String> {
    serde_json::to_string(value).context("escape Lark string literal")
}

/// Validate a generated value against IronMLX's supported JSON Schema subset.
pub fn validate_schema_value(schema: &Value, value: &Value) -> Result<()> {
    let object = schema
        .as_object()
        .expect("schema validated before generation");
    if let Some(any_of) = object.get("anyOf").and_then(Value::as_array) {
        if !any_of
            .iter()
            .any(|branch| validate_schema_value(branch, value).is_ok())
        {
            bail!("value does not match any anyOf branch");
        }
    }
    if let Some(expected) = object.get("const") {
        anyhow::ensure!(value == expected, "value does not match const");
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        anyhow::ensure!(values.contains(value), "value is not in enum");
    }
    if let Some(kind) = object.get("type") {
        let matches = match kind {
            Value::String(kind) => value_matches_type(value, kind),
            Value::Array(kinds) => kinds
                .iter()
                .filter_map(Value::as_str)
                .any(|kind| value_matches_type(value, kind)),
            _ => false,
        };
        anyhow::ensure!(matches, "value does not match schema type");
    }
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        let value_object = value
            .as_object()
            .ok_or_else(|| anyhow!("expected object value"))?;
        if let Some(required) = object.get("required").and_then(Value::as_array) {
            for name in required {
                let name = name.as_str().expect("schema validated");
                anyhow::ensure!(value_object.contains_key(name), "missing required `{name}`");
            }
        }
        if object.get("additionalProperties") == Some(&Value::Bool(false)) {
            for name in value_object.keys() {
                anyhow::ensure!(
                    properties.contains_key(name),
                    "undeclared property `{name}`"
                );
            }
        }
        for (name, child_schema) in properties {
            if let Some(child) = value_object.get(name) {
                validate_schema_value(child_schema, child)
                    .with_context(|| format!("property `{name}`"))?;
            }
        }
    }
    if let Some(items) = object.get("items") {
        let array = value
            .as_array()
            .ok_or_else(|| anyhow!("expected array value"))?;
        for (index, item) in array.iter().enumerate() {
            validate_schema_value(items, item).with_context(|| format!("item {index}"))?;
        }
    }
    Ok(())
}

fn value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => false,
    }
}

pub fn schema_uses_raw_string(schema: &Value) -> bool {
    schema_is_string_only(schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tool() -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["city"],
                "additionalProperties": false
            }
        }))
        .unwrap()
    }

    fn forecast_tool() -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "get_forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "units": {"type": "string", "enum": ["metric", "imperial"]}
                        },
                        "required": ["units"],
                        "additionalProperties": false
                    }
                },
                "required": ["coordinates", "options"],
                "additionalProperties": false
            }
        }))
        .unwrap()
    }

    fn ping_tool() -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "ping",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }
        }))
        .unwrap()
    }

    fn consume_bytes(session: &mut ConstraintSession, bytes: &[u8]) -> Result<()> {
        for (index, &byte) in bytes.iter().enumerate() {
            let mask = session.compute_mask()?;
            anyhow::ensure!(
                mask.is_allowed(u32::from(byte)),
                "byte {byte:#x} at offset {index} is masked"
            );
            session.commit_token(u32::from(byte))?;
        }
        Ok(())
    }

    fn gemma_tokens(value: &str) -> Vec<u32> {
        const SPECIALS: [(&str, u32); 5] = [
            ("<|tool_call>", 256),
            ("<tool_call|>", 257),
            ("<|\"|>", 258),
            ("<|channel>", 259),
            ("<channel|>", 260),
        ];
        let mut tokens = Vec::new();
        let mut remaining = value;
        while !remaining.is_empty() {
            if let Some((marker, token)) = SPECIALS
                .iter()
                .find(|(marker, _)| remaining.starts_with(marker))
            {
                tokens.push(*token);
                remaining = &remaining[marker.len()..];
            } else {
                let ch = remaining.chars().next().expect("remaining non-empty");
                assert!(ch.is_ascii(), "test helper only supports ASCII text");
                tokens.push(u32::from(ch as u8));
                remaining = &remaining[ch.len_utf8()..];
            }
        }
        tokens
    }

    #[test]
    fn qwen_grammar_accepts_text_and_valid_required_tool_call() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut text = plan.start_session().unwrap();
        consume_bytes(&mut text, b"ordinary answer").unwrap();
        assert!(text.is_accepting().unwrap());

        let mut tool = plan.start_session().unwrap();
        consume_bytes(
            &mut tool,
            b"thinking\n<tool_call>\n<function=get_weather>\n<parameter=days>2</parameter>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>",
        )
        .unwrap();
        assert!(tool.is_accepting().unwrap());
    }

    #[test]
    fn required_choice_rejects_plain_text_and_accepts_one_or_more_calls() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let options = ToolConstraintOptions {
            choice: ToolChoiceConstraint::Required,
            allow_parallel_calls: true,
        };
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &options)
            .unwrap();

        let mut text = plan.start_session().unwrap();
        consume_bytes(&mut text, b"ordinary answer").unwrap();
        assert!(!text.is_accepting().unwrap());

        let call = b"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>";
        let mut one = plan.start_session().unwrap();
        consume_bytes(&mut one, call).unwrap();
        assert!(one.is_accepting().unwrap());

        let mut two = plan.start_session().unwrap();
        consume_bytes(&mut two, call).unwrap();
        consume_bytes(&mut two, call).unwrap();
        assert!(two.is_accepting().unwrap());
    }

    #[test]
    fn disabled_parallel_calls_allows_zero_or_one_call() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let options = ToolConstraintOptions {
            choice: ToolChoiceConstraint::Auto,
            allow_parallel_calls: false,
        };
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &options)
            .unwrap();
        let call = b"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>";

        let mut text = plan.start_session().unwrap();
        consume_bytes(&mut text, b"ordinary answer").unwrap();
        assert!(text.is_accepting().unwrap());

        let mut one = plan.start_session().unwrap();
        consume_bytes(&mut one, call).unwrap();
        assert!(one.is_accepting().unwrap());
        let accepted = one
            .validate_tokens(&call.iter().map(|byte| u32::from(*byte)).collect::<Vec<_>>())
            .unwrap();
        assert!(accepted < call.len());
    }

    #[test]
    fn forced_function_allows_exactly_one_selected_call() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let options = ToolConstraintOptions {
            choice: ToolChoiceConstraint::Function("get_forecast".into()),
            allow_parallel_calls: true,
        };
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool(), forecast_tool()], &options)
            .unwrap();
        let weather = b"<tool_call><function=get_weather>";
        let session = plan.start_session().unwrap();
        assert!(
            session
                .validate_tokens(
                    &weather
                        .iter()
                        .map(|byte| u32::from(*byte))
                        .collect::<Vec<_>>()
                )
                .unwrap()
                < weather.len()
        );

        let forecast = b"<tool_call><function=get_forecast><parameter=coordinates>[35.6,139.7]</parameter><parameter=options>{\"units\":\"metric\"}</parameter></function></tool_call>";
        let mut selected = plan.start_session().unwrap();
        consume_bytes(&mut selected, forecast).unwrap();
        assert!(selected.is_accepting().unwrap());
        let accepted = selected
            .validate_tokens(
                &forecast
                    .iter()
                    .map(|byte| u32::from(*byte))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(accepted < forecast.len());
    }

    #[test]
    fn qwen_grammar_masks_unknown_function_and_required_close() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, b"<tool_call><function=").unwrap();
        let mask = session.compute_mask().unwrap();
        assert!(mask.is_allowed(u32::from(b'g')));
        assert!(!mask.is_allowed(u32::from(b'x')));

        let mut missing = plan.start_session().unwrap();
        consume_bytes(&mut missing, b"<tool_call><function=get_weather>").unwrap();
        let accepted = missing
            .validate_tokens(
                b"</function>"
                    .iter()
                    .map(|b| u32::from(*b))
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap();
        assert!(accepted < "</function>".len());
    }

    #[test]
    fn llama_grammar_discriminates_text_and_enforces_single_json_call() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let auto = tokenizer
            .compile_llama_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut text = auto.start_session().unwrap();
        consume_bytes(&mut text, b"  ordinary answer").unwrap();
        assert!(text.is_accepting().unwrap());

        let valid = br#" {"name":"get_weather","parameters":{"city":"Tokyo","days":2}} "#;
        let mut call = auto.start_session().unwrap();
        consume_bytes(&mut call, valid).unwrap();
        assert!(call.is_accepting().unwrap());

        let malformed_text = b"  {not a tool call";
        let malformed_tokens = malformed_text
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let session = auto.start_session().unwrap();
        assert!(session.validate_tokens(&malformed_tokens).unwrap() < malformed_tokens.len());

        let missing_required = br#"{"name":"get_weather","parameters":{"days":2}}"#;
        let missing_tokens = missing_required
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let session = auto.start_session().unwrap();
        assert!(session.validate_tokens(&missing_tokens).unwrap() < missing_tokens.len());

        let required = tokenizer
            .compile_llama_tools(
                &[weather_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: true,
                },
            )
            .unwrap();
        let plain_tokens = b"ordinary answer"
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let session = required.start_session().unwrap();
        assert_eq!(session.validate_tokens(&plain_tokens).unwrap(), 0);

        let twice = [valid.as_slice(), valid.as_slice()].concat();
        let twice_tokens = twice.iter().copied().map(u32::from).collect::<Vec<_>>();
        let session = required.start_session().unwrap();
        assert!(session.validate_tokens(&twice_tokens).unwrap() < twice_tokens.len());
    }

    #[test]
    fn llama_grammar_forces_function_and_nested_parameter_schema() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let forced = tokenizer
            .compile_llama_tools(
                &[weather_tool(), forecast_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Function("get_forecast".into()),
                    allow_parallel_calls: true,
                },
            )
            .unwrap();
        let valid = br#"{"name":"get_forecast","parameters":{"coordinates":[35.6,139.7],"options":{"units":"metric"}}}"#;
        let mut session = forced.start_session().unwrap();
        consume_bytes(&mut session, valid).unwrap();
        assert!(session.is_accepting().unwrap());

        let invalid = br#"{"name":"get_forecast","parameters":{"coordinates":[35.6,139.7],"options":{"units":"kelvin"}}}"#;
        let tokens = invalid.iter().copied().map(u32::from).collect::<Vec<_>>();
        let session = forced.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());

        let weather = br#"{"name":"get_weather","parameters":{"city":"Tokyo"}}"#;
        let tokens = weather.iter().copied().map(u32::from).collect::<Vec<_>>();
        let session = forced.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());
    }

    #[test]
    fn minicpm5_grammar_accepts_plain_cdata_and_typed_arguments() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let auto = tokenizer
            .compile_minicpm5_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();

        let mut text = auto.start_session().unwrap();
        consume_bytes(&mut text, b"ordinary answer").unwrap();
        assert!(text.is_accepting().unwrap());

        let mut direct = auto.start_session().unwrap();
        consume_bytes(
            &mut direct,
            b"<function name=\"get_weather\"><param name=\"days\">2</param><param name=\"city\">Tokyo</param></function>",
        )
        .unwrap();
        assert!(direct.is_accepting().unwrap());

        let mut cdata = auto.start_session().unwrap();
        consume_bytes(
            &mut cdata,
            b"<function name=\"get_weather\"><param name=\"city\"><![CDATA[Tokyo\n<&]]></param></function>",
        )
        .unwrap();
        assert!(cdata.is_accepting().unwrap());
    }

    #[test]
    fn minicpm5_grammar_enforces_required_parallel_and_forced_function_semantics() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let required = tokenizer
            .compile_minicpm5_tools(
                &[weather_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: true,
                },
            )
            .unwrap();
        let mut plain = required.start_session().unwrap();
        consume_bytes(&mut plain, b"ordinary answer").unwrap();
        assert!(!plain.is_accepting().unwrap());

        let call = b"<function name=\"get_weather\"><param name=\"city\">Tokyo</param></function>";
        let mut parallel = required.start_session().unwrap();
        consume_bytes(&mut parallel, call).unwrap();
        consume_bytes(&mut parallel, b"\n").unwrap();
        consume_bytes(&mut parallel, call).unwrap();
        assert!(parallel.is_accepting().unwrap());

        let serial = tokenizer
            .compile_minicpm5_tools(
                &[weather_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: false,
                },
            )
            .unwrap();
        let mut one = serial.start_session().unwrap();
        consume_bytes(&mut one, call).unwrap();
        assert!(one.is_accepting().unwrap());
        let second = call.iter().copied().map(u32::from).collect::<Vec<_>>();
        assert!(one.validate_tokens(&second).unwrap() < second.len());

        let forced = tokenizer
            .compile_minicpm5_tools(
                &[weather_tool(), forecast_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Function("get_forecast".into()),
                    allow_parallel_calls: true,
                },
            )
            .unwrap();
        let valid = b"<function name=\"get_forecast\"><param name=\"coordinates\">[35.6,139.7]</param><param name=\"options\">{\"units\":\"metric\"}</param></function>";
        let mut selected = forced.start_session().unwrap();
        consume_bytes(&mut selected, valid).unwrap();
        assert!(selected.is_accepting().unwrap());

        let weather = call.iter().copied().map(u32::from).collect::<Vec<_>>();
        assert!(
            forced
                .start_session()
                .unwrap()
                .validate_tokens(&weather)
                .unwrap()
                < weather.len()
        );
    }

    #[test]
    fn tagged_grammars_accept_parameterless_functions_without_parametric_state() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let options = ToolConstraintOptions {
            choice: ToolChoiceConstraint::Required,
            allow_parallel_calls: false,
        };

        let qwen = tokenizer
            .compile_qwen_tools(&[ping_tool()], &options)
            .unwrap();
        let mut qwen_session = qwen.start_session().unwrap();
        consume_bytes(
            &mut qwen_session,
            b"<tool_call><function=ping></function></tool_call>",
        )
        .unwrap();
        assert!(qwen_session.is_accepting().unwrap());

        let minicpm5 = tokenizer
            .compile_minicpm5_tools(&[ping_tool()], &options)
            .unwrap();
        let mut minicpm5_session = minicpm5.start_session().unwrap();
        consume_bytes(
            &mut minicpm5_session,
            b"<function name=\"ping\"></function>",
        )
        .unwrap();
        assert!(minicpm5_session.is_accepting().unwrap());
    }

    #[test]
    fn schema_validation_is_recursive_and_rejects_unknown_keywords() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [1, 2]}
                }
            },
            "required": ["items"],
            "additionalProperties": false
        });
        let mut nodes = 0;
        validate_schema_node(&schema, 0, &mut nodes).unwrap();
        validate_schema_value(&schema, &serde_json::json!({"items": [1, 2]})).unwrap();
        assert!(validate_schema_value(&schema, &serde_json::json!({"items": [3]})).is_err());

        let bad = serde_json::json!({"type": "string", "pattern": "x"});
        let mut nodes = 0;
        assert!(validate_schema_node(&bad, 0, &mut nodes).is_err());
    }

    #[test]
    fn structured_output_schema_validation_enforces_root_and_strict_contracts() {
        let supported = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": ["number", "null"]}
            },
            "required": ["answer", "confidence"],
            "additionalProperties": false
        });
        validate_json_output_schema(&supported, true).unwrap();

        let non_object = serde_json::json!({"type": "string"});
        assert!(validate_json_output_schema(&non_object, false).is_err());

        let missing_property = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer"],
            "additionalProperties": false
        });
        assert!(validate_json_output_schema(&missing_property, true).is_err());

        let open_object = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        });
        assert!(validate_json_output_schema(&open_object, true).is_err());
    }

    #[test]
    fn standalone_structured_output_enforces_schema_during_decoding() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"enum": ["sunny", "rainy"]},
                "days": {"type": "integer"}
            },
            "required": ["answer", "days"],
            "additionalProperties": false
        });
        let plan = tokenizer.compile_json_output(&schema).unwrap();

        let valid = br#"{"answer":"sunny","days":2}"#;
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, valid).unwrap();
        assert!(session.is_accepting().unwrap());

        let invalid = br#"{"answer":"cloudy","days":2}"#;
        let tokens = invalid.iter().copied().map(u32::from).collect::<Vec<_>>();
        assert!(
            plan.start_session()
                .unwrap()
                .validate_tokens(&tokens)
                .unwrap()
                < tokens.len()
        );
    }

    #[test]
    fn auto_tool_grammars_accept_native_calls_or_structured_final_answers() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"const": "sunny"}},
            "required": ["answer"],
            "additionalProperties": false
        });
        let structured = br#"{"answer":"sunny"}"#;
        let options = ToolConstraintOptions::default();
        let byte_tokenizer = ConstraintTokenizer::byte_level().unwrap();

        let qwen = byte_tokenizer
            .compile_qwen_tools_with_output(&[weather_tool()], &options, Some(&schema))
            .unwrap();
        let mut qwen_json = qwen.start_session().unwrap();
        consume_bytes(&mut qwen_json, structured).unwrap();
        assert!(qwen_json.is_accepting().unwrap());
        let mut qwen_tool = qwen.start_session().unwrap();
        consume_bytes(
            &mut qwen_tool,
            b"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>",
        )
        .unwrap();
        assert!(qwen_tool.is_accepting().unwrap());
        let mut qwen_plain = qwen.start_session().unwrap();
        assert!(!qwen_plain
            .compute_mask()
            .unwrap()
            .is_allowed(u32::from(b'o')));
        let plain_result = consume_bytes(&mut qwen_plain, b"ordinary answer");
        assert!(plain_result.is_err() || !qwen_plain.is_accepting().unwrap());

        let glm = byte_tokenizer
            .compile_glm_tools_with_output(&[weather_tool()], &options, Some(&schema))
            .unwrap();
        let mut glm_json = glm.start_session().unwrap();
        consume_bytes(&mut glm_json, structured).unwrap();
        assert!(glm_json.is_accepting().unwrap());
        let mut glm_tool = glm.start_session().unwrap();
        consume_bytes(
            &mut glm_tool,
            b"<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
        )
        .unwrap();
        assert!(glm_tool.is_accepting().unwrap());

        let llama = byte_tokenizer
            .compile_llama_tools_with_output(&[weather_tool()], &options, Some(&schema))
            .unwrap();
        let mut llama_json = llama.start_session().unwrap();
        consume_bytes(&mut llama_json, structured).unwrap();
        assert!(llama_json.is_accepting().unwrap());
        let mut llama_tool = llama.start_session().unwrap();
        consume_bytes(
            &mut llama_tool,
            br#"{"name":"get_weather","parameters":{"city":"Tokyo"}}"#,
        )
        .unwrap();
        assert!(llama_tool.is_accepting().unwrap());

        let minicpm5 = byte_tokenizer
            .compile_minicpm5_tools_with_output(&[weather_tool()], &options, Some(&schema))
            .unwrap();
        let mut minicpm5_json = minicpm5.start_session().unwrap();
        consume_bytes(&mut minicpm5_json, structured).unwrap();
        assert!(minicpm5_json.is_accepting().unwrap());
        let mut minicpm5_tool = minicpm5.start_session().unwrap();
        consume_bytes(
            &mut minicpm5_tool,
            b"<function name=\"get_weather\"><param name=\"city\">Tokyo</param></function>",
        )
        .unwrap();
        assert!(minicpm5_tool.is_accepting().unwrap());

        let gemma_tokenizer = ConstraintTokenizer::byte_level_gemma().unwrap();
        let gemma = gemma_tokenizer
            .compile_gemma_tools_with_output(&[weather_tool()], &options, Some(&schema))
            .unwrap();
        let mut gemma_json = gemma.start_session().unwrap();
        gemma_json
            .commit_tokens(&gemma_tokens(std::str::from_utf8(structured).unwrap()))
            .unwrap();
        assert!(gemma_json.is_accepting().unwrap());
        let mut gemma_tool = gemma.start_session().unwrap();
        gemma_tool
            .commit_tokens(&gemma_tokens(
                "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
            ))
            .unwrap();
        assert!(gemma_tool.is_accepting().unwrap());
    }

    #[test]
    fn required_tool_choice_does_not_admit_structured_final_output() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        });
        let plan = tokenizer
            .compile_qwen_tools_with_output(
                &[weather_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: true,
                },
                Some(&schema),
            )
            .unwrap();
        let structured =
            br#"{"answer":"sunny"}"#.iter().copied().map(u32::from).collect::<Vec<_>>();
        let mut session = plan.start_session().unwrap();
        assert!(session.commit_tokens(&structured).is_err());
    }

    #[test]
    fn grammar_uses_parametric_uniqueness_state() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        assert!(plan.grammar_source().contains("params_0::set_bit(0)"));
        assert!(plan.grammar_source().contains("%if bit_clear(1)"));
    }

    #[test]
    fn speculative_masks_reject_invalid_draft_without_advancing_state() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, b"<tool_call><function=").unwrap();

        let masks = session
            .speculative_masks(&[u32::from(b'x'), u32::from(b'x')])
            .unwrap();
        assert_eq!(masks.len(), 3);
        assert!(!masks[0].is_allowed(u32::from(b'x')));
        assert!(masks[0].is_allowed(u32::from(b'g')));
        assert!(masks[1].is_allowed(u32::from(b'g')));
        assert!(masks[2].is_allowed(u32::from(b'g')));
    }

    #[test]
    fn speculative_validation_preserves_eos_after_accepting_output() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string", "const": "done"}
            },
            "required": ["answer"],
            "additionalProperties": false
        });
        let plan = tokenizer.compile_json_output(&schema).unwrap();
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, br#"{"answer":"done"}"#).unwrap();
        assert!(session.is_accepting().unwrap());

        let mut resolved = vec![256];
        session
            .truncate_invalid_speculative_bonus(&mut resolved)
            .unwrap();
        assert_eq!(resolved, vec![256]);
        session.commit_tokens(&resolved).unwrap();
    }

    #[test]
    fn additive_mask_forces_allowed_token_argmax() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, b"<tool_call><function=").unwrap();
        let mask = session.compute_mask().unwrap();
        let mut values = vec![0.0_f32; 257];
        values[b'x' as usize] = 100.0;
        values[b'g' as usize] = 1.0;
        let logits: Array = (&values[..], &[257_i32][..]).try_into().unwrap();
        let masked = apply_token_mask(&logits, &mask).unwrap();
        let token: u32 = mlx::ops::reduction::argmax(&masked, -1, false)
            .unwrap()
            .item()
            .unwrap();
        assert_eq!(token, u32::from(b'g'));
    }

    #[test]
    fn additive_masks_reject_model_vocab_padding_without_reading_past_the_trie() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, b"<tool_call><function=").unwrap();
        let mask = session.compute_mask().unwrap();
        let padded_vocab = mask.len() + 17;
        let mut values = vec![0.0_f32; padded_vocab];
        values[padded_vocab - 1] = 100.0;
        values[b'g' as usize] = 1.0;
        let logits: Array = (&values[..], &[padded_vocab as i32][..])
            .try_into()
            .unwrap();
        let masked = apply_token_mask(&logits, &mask).unwrap();
        let token: u32 = mlx::ops::reduction::argmax(&masked, -1, false)
            .unwrap()
            .item()
            .unwrap();
        assert_eq!(token, u32::from(b'g'));
    }

    #[test]
    fn diffusion_masks_apply_prefix_conditioned_rows_and_vocab_padding() {
        let mut first = SimpleVob::alloc(4);
        first.allow_token(1);
        let mut second = SimpleVob::alloc(4);
        second.allow_token(2);
        let logits: Array = (
            &[0.0_f32, 5.0, 100.0, 0.0, 200.0, 100.0, 0.0, 5.0, 0.0, 200.0][..],
            &[1_i32, 2_i32, 5_i32][..],
        )
            .try_into()
            .unwrap();
        let masked = apply_diffusion_token_masks(&logits, &[first, second]).unwrap();
        let selected = mlx::ops::reduction::argmax(&masked, -1, false)
            .unwrap()
            .to_vec::<u32>()
            .unwrap();
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn matcher_rollback_restores_the_previous_function_name_state() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, b"<tool_call><function=get_").unwrap();
        consume_bytes(&mut session, b"weather").unwrap();
        session.rollback("weather".len()).unwrap();

        let mask = session.compute_mask().unwrap();
        assert!(mask.is_allowed(u32::from(b'w')));
        assert!(!mask.is_allowed(u32::from(b'x')));
        consume_bytes(
            &mut session,
            b"weather><parameter=city>Tokyo</parameter></function></tool_call>",
        )
        .unwrap();
        assert!(session.is_accepting().unwrap());
    }

    #[test]
    fn grammar_accepts_multiple_calls_and_enforces_nested_json_schema() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_qwen_tools(
                &[weather_tool(), forecast_tool()],
                &ToolConstraintOptions::default(),
            )
            .unwrap();
        let valid = b"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>\n<tool_call><function=get_forecast><parameter=coordinates>[35.6,139.7]</parameter><parameter=options>{\"units\":\"metric\"}</parameter></function></tool_call>";
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, valid).unwrap();
        assert!(session.is_accepting().unwrap());

        let invalid = b"<tool_call><function=get_forecast><parameter=coordinates>[35.6,139.7]</parameter><parameter=options>{\"units\":\"kelvin\"}</parameter>";
        let invalid_tokens = invalid
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let session = plan.start_session().unwrap();
        assert!(session.validate_tokens(&invalid_tokens).unwrap() < invalid_tokens.len());
    }

    #[test]
    fn gemma_grammar_enforces_native_syntax_and_required_arguments() {
        let tokenizer = ConstraintTokenizer::byte_level_gemma().unwrap();
        let plan = tokenizer
            .compile_gemma_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut speculative = plan.start_session().unwrap();
        speculative
            .commit_tokens(&gemma_tokens(
                "<|tool_call>call:get_weather{city:<|\"|><|\"|>}",
            ))
            .unwrap();
        let mut resolved = vec![257, u32::from(b'X')];
        speculative
            .truncate_invalid_speculative_bonus(&mut resolved)
            .unwrap();
        assert_eq!(resolved, vec![257]);

        let valid = b"<|tool_call>call:get_weather{days:2,city:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let mut session = plan.start_session().unwrap();
        session
            .commit_tokens(&gemma_tokens(std::str::from_utf8(valid).unwrap()))
            .unwrap();
        assert!(session.is_accepting().unwrap());

        let missing_required = b"<|tool_call>call:get_weather{days:2}<tool_call|>";
        let tokens = gemma_tokens(std::str::from_utf8(missing_required).unwrap());
        let session = plan.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());

        let with_thinking = concat!(
            "<|channel>thought\nI should use the weather tool.<channel|>",
            "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>"
        );
        let mut session = plan.start_session().unwrap();
        session.commit_tokens(&gemma_tokens(with_thinking)).unwrap();
        assert!(session.is_accepting().unwrap());
    }

    #[test]
    fn gemma_grammar_enforces_parallel_and_forced_function_semantics() {
        let tokenizer = ConstraintTokenizer::byte_level_gemma().unwrap();
        let plan = tokenizer
            .compile_gemma_tools(
                &[weather_tool(), forecast_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Function("get_weather".into()),
                    allow_parallel_calls: true,
                },
            )
            .unwrap();
        let valid = b"<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let mut session = plan.start_session().unwrap();
        session
            .commit_tokens(&gemma_tokens(std::str::from_utf8(valid).unwrap()))
            .unwrap();
        assert!(session.is_accepting().unwrap());

        let forecast = b"<|tool_call>call:get_forecast{";
        let tokens = gemma_tokens(std::str::from_utf8(forecast).unwrap());
        let session = plan.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());

        let serial = tokenizer
            .compile_gemma_tools(
                &[weather_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: false,
                },
            )
            .unwrap();
        let twice = gemma_tokens(&format!(
            "{}{}",
            std::str::from_utf8(valid).unwrap(),
            std::str::from_utf8(valid).unwrap()
        ));
        let session = serial.start_session().unwrap();
        assert!(session.validate_tokens(&twice).unwrap() < twice.len());

        let parallel = tokenizer
            .compile_gemma_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut session = parallel.start_session().unwrap();
        session.commit_tokens(&twice).unwrap();
        assert!(session.is_accepting().unwrap());
    }

    #[test]
    fn gemma_grammar_enforces_nested_objects_arrays_and_string_enums() {
        let tokenizer = ConstraintTokenizer::byte_level_gemma().unwrap();
        let plan = tokenizer
            .compile_gemma_tools(&[forecast_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let valid = concat!(
            "<|tool_call>call:get_forecast{coordinates:[35.6,139.7],",
            "options:{units:<|\"|>metric<|\"|>}}<tool_call|>"
        );
        let mut session = plan.start_session().unwrap();
        session.commit_tokens(&gemma_tokens(valid)).unwrap();
        assert!(session.is_accepting().unwrap());

        let invalid = valid.replace("metric", "kelvin");
        let tokens = gemma_tokens(&invalid);
        let session = plan.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());
    }

    #[test]
    fn glm_grammar_enforces_native_syntax_and_required_arguments() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_glm_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let valid = concat!(
            "<tool_call>get_weather",
            "<arg_key>days</arg_key><arg_value>2</arg_value>",
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>",
            "</tool_call>"
        );
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, valid.as_bytes()).unwrap();
        assert!(session.is_accepting().unwrap());

        let missing_required = concat!(
            "<tool_call>get_weather",
            "<arg_key>days</arg_key><arg_value>2</arg_value>",
            "</tool_call>"
        );
        let tokens = missing_required.bytes().map(u32::from).collect::<Vec<_>>();
        let session = plan.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());

        let unknown = b"<tool_call>unknown";
        let tokens = unknown.iter().copied().map(u32::from).collect::<Vec<_>>();
        let session = plan.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());
    }

    #[test]
    fn completed_glm_constraint_accepts_eos_without_consuming_it_as_grammar_text() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_glm_tools(
                &[weather_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: false,
                },
            )
            .unwrap();
        let mut incomplete = plan.start_session().unwrap();
        assert!(incomplete.commit_token(256).is_err());

        let valid = concat!(
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>",
            "</tool_call>"
        );
        let mut complete = plan.start_session().unwrap();
        consume_bytes(&mut complete, valid.as_bytes()).unwrap();
        assert!(complete.is_accepting().unwrap());
        assert!(complete.compute_mask().unwrap().is_allowed(256));
        complete.commit_token(256).unwrap();
        assert!(complete.is_accepting().unwrap());
    }

    #[test]
    fn glm_grammar_enforces_parallel_and_forced_function_semantics() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let weather = concat!(
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>",
            "</tool_call>"
        );
        let forced = tokenizer
            .compile_glm_tools(
                &[weather_tool(), forecast_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Function("get_weather".into()),
                    allow_parallel_calls: true,
                },
            )
            .unwrap();
        let mut session = forced.start_session().unwrap();
        consume_bytes(&mut session, weather.as_bytes()).unwrap();
        assert!(session.is_accepting().unwrap());

        let forecast = b"<tool_call>get_forecast";
        let tokens = forecast.iter().copied().map(u32::from).collect::<Vec<_>>();
        let session = forced.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());

        let twice = format!("{weather}{weather}");
        let serial = tokenizer
            .compile_glm_tools(
                &[weather_tool()],
                &ToolConstraintOptions {
                    choice: ToolChoiceConstraint::Required,
                    allow_parallel_calls: false,
                },
            )
            .unwrap();
        let session = serial.start_session().unwrap();
        let tokens = twice.bytes().map(u32::from).collect::<Vec<_>>();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());

        let parallel = tokenizer
            .compile_glm_tools(&[weather_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let mut session = parallel.start_session().unwrap();
        session.commit_tokens(&tokens).unwrap();
        assert!(session.is_accepting().unwrap());
    }

    #[test]
    fn glm_grammar_enforces_nested_json_and_string_enums() {
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let plan = tokenizer
            .compile_glm_tools(&[forecast_tool()], &ToolConstraintOptions::default())
            .unwrap();
        let valid = concat!(
            "<tool_call>get_forecast",
            "<arg_key>coordinates</arg_key><arg_value>[35.6,139.7]</arg_value>",
            "<arg_key>options</arg_key><arg_value>{\"units\":\"metric\"}</arg_value>",
            "</tool_call>"
        );
        let mut session = plan.start_session().unwrap();
        consume_bytes(&mut session, valid.as_bytes()).unwrap();
        assert!(session.is_accepting().unwrap());

        let invalid = valid.replace("metric", "kelvin");
        let tokens = invalid.bytes().map(u32::from).collect::<Vec<_>>();
        let session = plan.start_session().unwrap();
        assert!(session.validate_tokens(&tokens).unwrap() < tokens.len());
    }
}
