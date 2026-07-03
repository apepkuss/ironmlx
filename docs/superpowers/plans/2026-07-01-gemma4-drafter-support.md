# Gemma4 Drafter Support

> Required sub-skill note: follow test-driven-development before production code, use verification-before-completion before claiming completion.

## Goal

Add production-grade support for Gemma4 assistant/drafter checkpoints such as `gemma-4-E4B-it-qat-assistant-4bit` and `gemma-4-12B-it-assistant-4bit`, treating them as Gemma4 shared-KV assistant models rather than Qwen-style MTP heads.

## Constraints

- Keep Qwen MTP behavior unchanged.
- Do not add compatibility shims for unsupported architectures.
- Validate Gemma4 base/drafter pairs before runtime loading.
- Preserve existing Gemma4 text and VL paths.
- Run the repository-required Rust checks after Rust edits.

## Implementation Plan

1. Add failing tests for Gemma4 assistant config parsing, loader sanitization, model-manager pair validation, CLI/serve gating, and App local model classification.
2. Extend Gemma4 config handling with explicit assistant config types:
   - accept `gemma4_assistant` and `gemma4_unified_assistant`;
   - allow assistant text configs whose `num_kv_shared_layers == num_hidden_layers`;
   - validate drafter backbone hidden size, vocabulary, layer kinds, and tied embedding assumptions.
3. Add a Gemma4 drafter loader path:
   - keep assistant-specific tensors such as `pre_projection`, `post_projection`, `masked_embedding`, and `model.*`;
   - remove tied `lm_head` tensors when applicable;
   - cast ordered embedding token ordering to integer dtype.
4. Expose Gemma4 main-model shared K/V states:
   - return hidden states plus the latest `sliding_attention` and `full_attention` shared K/V entries;
   - support text and existing VL embedding paths without changing plain generation semantics.
5. Implement the Gemma4 assistant model:
   - pre-project concatenated target embedding and verified hidden state;
   - run drafter layers with external shared K/V only;
   - build bidirectional full/sliding drafter masks;
   - project back to backbone hidden size and produce tied logits.
6. Add a Gemma4 speculative generation stream:
   - draft greedily from assistant logits;
   - verify with the main model;
   - roll back/replay target cache on mismatch;
   - reuse existing speculative statistics where applicable.
7. Wire public entry points:
   - CLI `generate --mtp-model-dir` accepts Gemma4 base plus Gemma4 assistant;
   - `serve` and model manager validate Gemma4 drafter pairs;
   - app scanner marks Gemma4 assistant checkpoints as MTP/drafter artifacts and exposes matching compatibility signatures.
8. Verify:
   - targeted Rust unit tests;
   - Swift app scanner tests if Swift tooling is available;
   - `cargo fmt`;
   - `cargo +nightly fmt --all -- --check`;
   - `cargo +nightly clippy --all-features --workspace -- -D warnings`;
   - `cargo build --release`.
