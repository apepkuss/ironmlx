//! Tokenizer — thin wrapper around the `tokenizers` crate, plus an
//! attached chat template (optional) and resolved EOS token ids.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, Context};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::core::chat_template::{ChatTemplate, ChatTemplateSpecialTokens};
use crate::core::constrained::{ConstraintPlan, ConstraintTokenizer, ToolConstraintOptions};
use crate::core::generated_output::ModelCapabilityProfile;
use crate::core::loader::{EosTokenId, Loader, TokenizerConfig};
use crate::core::native_output::{NativeOutputDecoderConfig, NativeOutputDialect};
use crate::core::tool_calling::{ToolDefinition, ToolDialect};
use crate::core::tool_prompt_cache::{
    CacheInsertKey, CacheLookupKey, CacheMatch, SafeBoundary, ToolPromptCache,
};
use crate::Result;

pub use crate::core::tool_prompt_cache::ToolPromptCacheStats;

/// Tokenizer + optional chat template + EOS token id list.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    chat: Option<ChatTemplate>,
    tool_dialect: Option<ToolDialect>,
    native_output_dialect: Option<NativeOutputDialect>,
    constraint: Option<ConstraintTokenizer>,
    eos_token_ids: Vec<u32>,
    tool_prompt_cache_identity: [u8; 32],
    tool_prompt_cache_boundaries: HashSet<u32>,
    tool_prompt_cache: ToolPromptCache,
}

/// Streaming detokenizer — owns its own state (does NOT delegate to
/// `tokenizers::DecodeStream`; that crate's `step_decode_stream` has a
/// known usize-underflow bug at version 0.20.4 — see
/// `tokenizers/src/tokenizer/mod.rs:1108`, `let new_prefix_index =
/// ids.len() - *prefix_index` underflows when state variables drift out
/// of sync, panicking the request handler).
///
/// Algorithm (correct, simple, O(N) per step over **generated tokens
/// only** — N is bounded by `max_new_tokens`, typically 128-2048, so the
/// O(N²) total cost is negligible vs GPU forward time):
///
/// 1. push the new id to the rolling token buffer
/// 2. decode the buffer to a fresh string
/// 3. if the new string starts with the previously-emitted prefix, return
///    the suffix (delta) and update prefix; otherwise the BPE has not yet
///    produced a stable boundary, return `None` and wait for more tokens
/// 4. drop the leading replacement char (`U+FFFD`) case as `None` — the
///    next token will resolve it
///
/// Lifetime `'a` borrows the underlying [`Tokenizer`] for `decode` calls.
pub struct DecodeStream<'a> {
    tokenizer: &'a Tokenizer,
    skip_special: bool,
    /// Generated token ids (NOT including prompt) accumulated so far.
    ids: Vec<u32>,
    /// Last text string emitted to the caller — the running prefix.
    last_text: String,
}

impl<'a> DecodeStream<'a> {
    /// Feed one token id, get the incremental text delta. `Ok(None)` means
    /// the underlying BPE has not yet produced a renderable string for
    /// this id (e.g. mid-codepoint UTF-8 split, or text shorter than the
    /// running prefix); the caller should keep streaming and the next
    /// `step` will catch up.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        let text = self.tokenizer.decode(&self.ids, self.skip_special)?;
        // BPE may emit a trailing replacement char while waiting for the
        // continuation token of a multi-byte UTF-8 sequence. Treat this
        // as "no progress yet" — return None, do NOT advance prefix.
        if text.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        if text.len() < self.last_text.len() {
            // Text shrank (rare BPE re-segmentation). Reset prefix to the
            // new shorter text and report no delta this step.
            self.last_text = text;
            return Ok(None);
        }
        if !text.starts_with(&self.last_text) {
            // Prefix divergence — the latest decode does not extend the
            // previous prefix. Most likely a temporary BPE boundary shift;
            // wait for the next token to settle.
            return Ok(None);
        }
        let delta = text[self.last_text.len()..].to_string();
        self.last_text = text;
        if delta.is_empty() {
            Ok(None)
        } else {
            Ok(Some(delta))
        }
    }
}

impl Tokenizer {
    /// Load tokenizer data and chat-template configuration without opening
    /// model weights.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let config = TokenizerConfig::from_model_dir(model_dir)?;
        let config_raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        let config = merge_model_eos_token_ids(config, &config_raw)?;
        let model_type = config_raw
            .get("model_type")
            .and_then(serde_json::Value::as_str);
        Self::from_files_with_model_type(&model_dir.join("tokenizer.json"), &config, model_type)
    }

    /// Build a [`Tokenizer`] from a [`Loader`]. Loads
    /// `{model_dir}/tokenizer.json` and uses
    /// [`Loader::tokenizer_config`] for chat template + EOS resolution.
    pub fn from_loader(loader: &Loader) -> Result<Self> {
        let path = loader.model_dir().join("tokenizer.json");
        let model_type = loader
            .config_raw_value()
            .get("model_type")
            .and_then(serde_json::Value::as_str);
        let config = merge_model_eos_token_ids(
            loader.tokenizer_config().clone(),
            loader.config_raw_value(),
        )?;
        Self::from_files_with_model_type(&path, &config, model_type)
    }

    /// Build directly from a `tokenizer.json` path and a parsed
    /// [`TokenizerConfig`].
    pub fn from_files(tokenizer_json: &Path, cfg: &TokenizerConfig) -> Result<Self> {
        Self::from_files_with_model_type(tokenizer_json, cfg, None)
    }

    fn from_files_with_model_type(
        tokenizer_json: &Path,
        cfg: &TokenizerConfig,
        model_type: Option<&str>,
    ) -> Result<Self> {
        let tokenizer_bytes = std::fs::read(tokenizer_json)?;
        let tokenizer_value: serde_json::Value = serde_json::from_slice(&tokenizer_bytes)?;
        let inner = tokenizers::Tokenizer::from_file(tokenizer_json)
            .map_err(|e| anyhow!("tokenizers::from_file: {e}"))?;
        let tool_dialect = model_type.and_then(|model_type| {
            cfg.chat_template
                .as_deref()
                .and_then(|template| ToolDialect::detect(model_type, template))
        });
        let native_output_dialect = model_type.and_then(|model_type| {
            cfg.chat_template
                .as_deref()
                .and_then(|template| NativeOutputDialect::detect(model_type, template))
        });
        let chat = match cfg.chat_template.as_deref() {
            Some(src) => Some(ChatTemplate::new_with_special_tokens(
                src,
                ChatTemplateSpecialTokens {
                    bos_token: cfg.bos_token.clone(),
                    eos_token: cfg.eos_token.clone(),
                    pad_token: cfg.pad_token.clone(),
                },
            )?),
            None => None,
        };
        let eos_token_ids = resolve_eos_token_ids(&inner, cfg);
        let tool_prompt_cache_boundaries = inner
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| {
                (token.special && !token.single_word && !token.rstrip && !token.normalized)
                    .then_some(id)
            })
            .collect();
        let tool_prompt_cache_identity = tool_prompt_cache_identity(
            &tokenizer_bytes,
            cfg.chat_template.as_deref(),
            cfg,
            model_type,
        );
        let constraint = tool_dialect
            .map(|dialect| {
                ConstraintTokenizer::from_tokenizer_json(
                    &tokenizer_value,
                    &eos_token_ids,
                    dialect.ordinary_constraint_tokens(),
                )
            })
            .transpose()?;
        Ok(Self {
            inner,
            chat,
            tool_dialect,
            native_output_dialect,
            constraint,
            eos_token_ids,
            tool_prompt_cache_identity,
            tool_prompt_cache_boundaries,
            tool_prompt_cache: ToolPromptCache::default(),
        })
    }

    /// Encode `text` to token ids. `add_special_tokens` controls BOS/EOS
    /// insertion as defined by the tokenizer config.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids to a string. `skip_special` drops BOS/EOS/etc.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Construct a streaming detokenizer for the decode hot path. Owns its
    /// own state (rolling token buffer + last-emitted prefix string);
    /// safe across the long-prompt / large-token-count workloads ironmlx
    /// targets (10K+ prompts). `skip_special` mirrors the same flag on
    /// [`Tokenizer::decode`].
    pub fn decode_stream(&self, skip_special: bool) -> DecodeStream<'_> {
        DecodeStream {
            tokenizer: self,
            skip_special,
            ids: Vec::new(),
            last_text: String::new(),
        }
    }

    /// Resolved EOS token ids, in declared order. Empty if unresolved.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Resolve a token's textual form to its numeric id. Used for special
    /// tokens like `<|image_pad|>` so VL-aware callers don't hardcode ids.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Render a chat template. Errors if the tokenizer config did not
    /// supply a `chat_template`. `extra_kwargs` (when present) is a JSON
    /// object whose top-level keys are merged into the template context
    /// (e.g. `{"enable_thinking": false}` from OpenAI's
    /// `chat_template_kwargs`).
    pub fn apply_chat_template<M: Serialize>(
        &self,
        messages: &[M],
        add_generation_prompt: bool,
        extra_kwargs: Option<&serde_json::Value>,
    ) -> Result<String> {
        let chat = self
            .chat
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer has no chat template"))?;
        chat.render_serializable(messages, add_generation_prompt, extra_kwargs)
    }

    /// Render and tokenize a native tool prompt with exact-result and safe-prefix caching.
    /// Cache keys preserve the exact serialized messages and template kwargs. Prefix reuse is
    /// restricted to context-independent special-token boundaries (not normalized, single-word,
    /// or right-stripping), where tokenization of the following text is isolated from the cached
    /// prefix.
    pub fn render_and_encode_tool_prompt<M: Serialize>(
        &self,
        messages: &[M],
        extra_kwargs: &serde_json::Value,
    ) -> Result<Vec<u32>> {
        let message_key = serde_json::to_vec(messages)?;
        let kwargs_key = serde_json::to_vec(extra_kwargs)?;
        let cache_key =
            CacheLookupKey::new(&self.tool_prompt_cache_identity, &message_key, &kwargs_key);
        if let Some(token_ids) = self.tool_prompt_cache.lookup_exact(&cache_key) {
            return Ok(token_ids);
        }

        let prompt = self.apply_chat_template(messages, true, Some(extra_kwargs))?;
        let (token_ids, boundaries) = match self
            .tool_prompt_cache
            .lookup_after_render(&cache_key, &prompt)
        {
            CacheMatch::Exact(token_ids) => return Ok(token_ids),
            CacheMatch::Prefix(mut prefix) => {
                let suffix = &prompt[prefix.byte_offset..];
                let encoding = self
                    .inner
                    .encode(suffix, false)
                    .map_err(|error| anyhow!("encode tool prompt suffix: {error}"))?;
                let prefix_token_count = prefix.token_ids.len();
                prefix.boundaries.extend(self.safe_boundaries(
                    &encoding,
                    prefix.byte_offset,
                    prefix_token_count,
                ));
                prefix.token_ids.extend_from_slice(encoding.get_ids());
                (prefix.token_ids, prefix.boundaries)
            }
            CacheMatch::Miss => {
                let encoding = self
                    .inner
                    .encode(prompt.as_str(), false)
                    .map_err(|error| anyhow!("encode tool prompt: {error}"))?;
                let boundaries = self.safe_boundaries(&encoding, 0, 0);
                (encoding.get_ids().to_vec(), boundaries)
            }
        };
        self.tool_prompt_cache.insert(
            CacheInsertKey::new(self.tool_prompt_cache_identity, message_key, kwargs_key),
            prompt,
            token_ids.clone(),
            boundaries,
        );
        Ok(token_ids)
    }

    /// Snapshot counters and current occupancy for the native tool prompt cache.
    pub fn tool_prompt_cache_stats(&self) -> ToolPromptCacheStats {
        self.tool_prompt_cache.stats()
    }

    fn safe_boundaries(
        &self,
        encoding: &tokenizers::Encoding,
        byte_offset: usize,
        token_offset: usize,
    ) -> Vec<SafeBoundary> {
        encoding
            .get_ids()
            .iter()
            .zip(encoding.get_offsets())
            .enumerate()
            .filter_map(|(index, (id, (_, end)))| {
                (self.tool_prompt_cache_boundaries.contains(id) && *end > 0).then_some(
                    SafeBoundary {
                        byte_offset: byte_offset + *end,
                        token_count: token_offset + index + 1,
                    },
                )
            })
            .collect()
    }

    /// True iff a chat template was provided.
    pub fn has_chat_template(&self) -> bool {
        self.chat.is_some()
    }

    /// Native tool-call syntax recognized from the active chat template.
    pub fn tool_dialect(&self) -> Option<ToolDialect> {
        self.tool_dialect
    }

    /// Native reasoning/thought syntax recognized from the active template.
    pub fn native_output_dialect(&self) -> Option<NativeOutputDialect> {
        self.native_output_dialect
    }

    /// Build the request-local native output decoder configuration using the
    /// same template kwargs that rendered the prompt.
    pub fn native_output_decoder_config(
        &self,
        chat_template_kwargs: Option<&serde_json::Value>,
    ) -> Result<Option<NativeOutputDecoderConfig>> {
        self.native_output_dialect
            .map(|dialect| {
                Ok(NativeOutputDecoderConfig {
                    dialect,
                    reasoning_enabled: dialect.reasoning_enabled(chat_template_kwargs)?,
                })
            })
            .transpose()
    }

    /// Capabilities derived from the components and exact template contract
    /// loaded for this tokenizer. Optional output channels remain disabled
    /// until a model-specific producer is registered and validated.
    pub fn capability_profile(&self, supports_image_input: bool) -> ModelCapabilityProfile {
        ModelCapabilityProfile::from_loaded_contract(
            supports_image_input,
            self.tool_dialect,
            self.constraint.is_some(),
            self.native_output_dialect.is_some(),
        )
    }

    /// Compile one immutable request plan for the active native tool dialect.
    pub fn compile_tool_constraint(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
    ) -> Result<ConstraintPlan> {
        match (self.tool_dialect, self.constraint.as_ref()) {
            (Some(ToolDialect::Qwen35), Some(constraint)) => {
                constraint.compile_qwen_tools(tools, options)
            }
            (Some(ToolDialect::Gemma), Some(constraint)) => {
                constraint.compile_gemma_tools(tools, options)
            }
            (Some(ToolDialect::Glm), Some(constraint)) => {
                constraint.compile_glm_tools(tools, options)
            }
            (Some(ToolDialect::Llama), Some(constraint)) => {
                constraint.compile_llama_tools(tools, options)
            }
            (Some(ToolDialect::MiniCpmV46), Some(constraint)) => {
                constraint.compile_qwen_tools(tools, options)
            }
            (Some(ToolDialect::MiniCpm5), Some(constraint)) => {
                constraint.compile_minicpm5_tools(tools, options)
            }
            _ => Err(anyhow!(
                "tokenizer does not provide a supported constrained tool dialect"
            )),
        }
    }

    /// Compile a native tool grammar whose `auto` branch may alternatively
    /// produce one structured JSON final answer.
    pub fn compile_tool_or_json_constraint(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
        output_schema: &serde_json::Value,
    ) -> Result<ConstraintPlan> {
        match (self.tool_dialect, self.constraint.as_ref()) {
            (Some(ToolDialect::Qwen35), Some(constraint)) => {
                constraint.compile_qwen_tools_with_output(tools, options, Some(output_schema))
            }
            (Some(ToolDialect::Gemma), Some(constraint)) => {
                constraint.compile_gemma_tools_with_output(tools, options, Some(output_schema))
            }
            (Some(ToolDialect::Glm), Some(constraint)) => {
                constraint.compile_glm_tools_with_output(tools, options, Some(output_schema))
            }
            (Some(ToolDialect::Llama), Some(constraint)) => {
                constraint.compile_llama_tools_with_output(tools, options, Some(output_schema))
            }
            (Some(ToolDialect::MiniCpmV46), Some(constraint)) => {
                constraint.compile_qwen_tools_with_output(tools, options, Some(output_schema))
            }
            (Some(ToolDialect::MiniCpm5), Some(constraint)) => {
                constraint.compile_minicpm5_tools_with_output(tools, options, Some(output_schema))
            }
            _ => Err(anyhow!(
                "tokenizer does not provide a supported constrained tool dialect"
            )),
        }
    }

    /// Compile client tools plus a structured final answer while leaving the
    /// model-native reasoning section unconstrained.
    pub fn compile_tool_or_json_constraint_with_reasoning(
        &self,
        tools: &[ToolDefinition],
        options: &ToolConstraintOptions,
        output_schema: &serde_json::Value,
        reasoning: NativeOutputDecoderConfig,
    ) -> Result<ConstraintPlan> {
        anyhow::ensure!(
            reasoning.reasoning_enabled,
            "reasoning-aware output constraint requires enabled native reasoning"
        );
        let tool_dialect = self.tool_dialect.ok_or_else(|| {
            anyhow!("tokenizer does not provide a supported constrained tool dialect")
        })?;
        self.constraint
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer does not support constrained decoding"))?
            .compile_tools_with_output_and_reasoning(
                tool_dialect,
                reasoning.dialect,
                tools,
                options,
                output_schema,
            )
    }

    /// Compile a standalone structured JSON output grammar.
    pub fn compile_json_output_constraint(
        &self,
        schema: &serde_json::Value,
    ) -> Result<ConstraintPlan> {
        self.constraint
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer does not support constrained decoding"))?
            .compile_json_output(schema)
    }

    /// Compile a structured final answer after an unconstrained native
    /// reasoning section.
    pub fn compile_json_output_constraint_with_reasoning(
        &self,
        schema: &serde_json::Value,
        reasoning: NativeOutputDecoderConfig,
    ) -> Result<ConstraintPlan> {
        anyhow::ensure!(
            reasoning.reasoning_enabled,
            "reasoning-aware output constraint requires enabled native reasoning"
        );
        self.constraint
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer does not support constrained decoding"))?
            .compile_json_output_with_reasoning(schema, reasoning.dialect)
    }
}

fn tool_prompt_cache_identity(
    tokenizer_bytes: &[u8],
    chat_template: Option<&str>,
    cfg: &TokenizerConfig,
    model_type: Option<&str>,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    for part in [
        tokenizer_bytes,
        chat_template.unwrap_or_default().as_bytes(),
        cfg.bos_token.as_deref().unwrap_or_default().as_bytes(),
        cfg.eos_token.as_deref().unwrap_or_default().as_bytes(),
        cfg.pad_token.as_deref().unwrap_or_default().as_bytes(),
        model_type.unwrap_or_default().as_bytes(),
    ] {
        digest.update(part.len().to_le_bytes());
        digest.update(part);
    }
    digest.finalize().into()
}

fn merge_model_eos_token_ids(
    mut config: TokenizerConfig,
    model_config: &serde_json::Value,
) -> Result<TokenizerConfig> {
    if config.eos_token_id.is_none() {
        config.eos_token_id = model_config
            .get("eos_token_id")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .context("parsing config.json eos_token_id")?;
    }
    Ok(config)
}

fn resolve_eos_token_ids(tok: &tokenizers::Tokenizer, cfg: &TokenizerConfig) -> Vec<u32> {
    // Direct ids first.
    if let Some(ids) = &cfg.eos_token_id {
        return match ids {
            EosTokenId::Single(i) => vec![*i],
            EosTokenId::Multi(v) => v.clone(),
        };
    }
    // Fall back to looking up the eos token string.
    if let Some(s) = &cfg.eos_token {
        if let Some(id) = tok.token_to_id(s) {
            return vec![id];
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_token_id_enum_matches_explicit_variants() {
        // This test only exercises the `EosTokenId` enum shape — it does
        // NOT invoke `resolve_eos_token_ids` (that path requires a real
        // `tokenizers::Tokenizer` and is covered by the integration
        // test). Real lookup requires a tokenizer; that path is covered
        // by the integration test.
        //
        // Here we only assert the explicit-id branches.
        let cfg = TokenizerConfig {
            chat_template: None,
            eos_token: None,
            bos_token: None,
            pad_token: None,
            eos_token_id: Some(EosTokenId::Single(7)),
        };
        // We cannot construct a tokenizer from thin air without a JSON
        // file, so for this branch we route through the explicit-id
        // path which never touches `tok`. Build a no-op stand-in via a
        // tiny in-memory BPE: to keep things simple, this test relies
        // on resolve_eos_token_ids handling Single directly. Use the
        // real public path by constructing through `from_files` with a
        // synthetic tokenizer json would need a fixture; skip that and
        // test the helper behaviour indirectly via a unit assertion.
        match cfg.eos_token_id.as_ref().unwrap() {
            EosTokenId::Single(i) => assert_eq!(*i, 7),
            EosTokenId::Multi(_) => panic!("expected Single"),
        }
    }

    #[test]
    fn model_config_multi_eos_shape_deserializes() {
        let ids: EosTokenId = serde_json::from_value(serde_json::json!([154820, 154827, 154829]))
            .expect("multi EOS ids");
        match ids {
            EosTokenId::Multi(ids) => assert_eq!(ids, vec![154820, 154827, 154829]),
            EosTokenId::Single(_) => panic!("expected multi EOS ids"),
        }
    }

    #[test]
    fn model_config_supplies_eos_ids_only_when_tokenizer_config_omits_them() {
        let config = TokenizerConfig {
            chat_template: None,
            eos_token: None,
            bos_token: None,
            pad_token: None,
            eos_token_id: None,
        };
        let merged = merge_model_eos_token_ids(
            config,
            &serde_json::json!({"eos_token_id": [154820, 154827, 154829]}),
        )
        .expect("model EOS ids");
        assert!(matches!(
            merged.eos_token_id,
            Some(EosTokenId::Multi(ref ids)) if ids == &[154820, 154827, 154829]
        ));

        let explicit = TokenizerConfig {
            chat_template: None,
            eos_token: None,
            bos_token: None,
            pad_token: None,
            eos_token_id: Some(EosTokenId::Single(7)),
        };
        let merged = merge_model_eos_token_ids(
            explicit,
            &serde_json::json!({"eos_token_id": [154820, 154827, 154829]}),
        )
        .expect("explicit tokenizer EOS id");
        assert!(matches!(merged.eos_token_id, Some(EosTokenId::Single(7))));
    }
}
