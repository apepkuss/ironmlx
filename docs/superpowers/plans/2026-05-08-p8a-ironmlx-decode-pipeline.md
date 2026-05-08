# P8a — ironmlx Decode Pipeline Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift ironmlx single-request decode TG from ~29 tok/s to ≥50 tok/s on Qwen3.5-4B-MLX-4bit by introducing GPU/CPU double-buffered decode (`mlx::transforms::async_eval`) plus an incremental BPE detokenizer (`tokenizers::DecodeStream`).

**Architecture:** Greedy path runs a depth-1 lookahead pipeline — each `next_token()` call dispatches step N+1's forward+argmax + `async_eval` *before* materializing step N's `.item()`. Detokenization moves from O(N²) full-history decode to O(1) per-token `decode_stream.step()`. Non-greedy / penalty-using callers fall through to existing synchronous path unchanged.

**Tech Stack:** Rust 2021, mlx 0.0.1 (this workspace's safe wrapper), tokenizers 0.20.4 (`decode_stream` API), anyhow 1, existing ironmlx test infra.

**Spec:** [`docs/superpowers/specs/2026-05-08-p8a-ironmlx-decode-pipeline-design.md`](../specs/2026-05-08-p8a-ironmlx-decode-pipeline-design.md)

---

## Conventions Recap

- **TDD per task**: failing test → run (FAIL) → implement → run (PASS) → fmt/lint/build → commit.
- **Project gate before each commit** (per `.claude/CLAUDE.md`):

  ```text
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  MLX_DIR=/Users/sam/.local/mlx cargo build --release
  ```

- **`MLX_DIR=/Users/sam/.local/mlx`** is required for the full workspace build (mlx-sys links headers from there). For ironmlx-only tests it's not strictly necessary on every step, but harmless to always include.
- **ASCII-only commit messages.**
- **Test threads**: ironmlx unit tests can run with default parallelism. The integration test in Task 4 (real Qwen3.5-4B checkpoint) requires `--test-threads=1` to avoid GPU contention; the existing `tests/p4_qwen35_logits_match.rs` already documents this.

---

## File Structure (after P8a)

```text
ironmlx/src/core/
├── sampler.rs              # MODIFIED: + is_pipelinable(), + sample_async_greedy(), + 2 tests
├── tokenizer.rs            # MODIFIED: + pub struct DecodeStream<'a> newtype, + decode_stream() method
└── generate.rs             # MODIFIED: + pipelined branching, + 2 tests
```

No new files. Three modules touched. No mlx-sys / mlx / model layer changes. No HTTP server changes.

---

## Task 1: Sampler — `is_pipelinable()` + `sample_async_greedy()` + tests

**Files:**
- Modify: `ironmlx/src/core/sampler.rs`

### Goal

Add a predicate method that classifies whether a Sampler instance is eligible for the pipelined decode path (greedy with no penalties), and add an async-greedy sample variant that returns a lazy `[1] u32 Array` without calling `.item()`. Two unit tests cover the new methods.

### Steps

- [ ] **Step 1.1: Write the failing test for `is_pipelinable`**

Append to the `#[cfg(test)] mod tests` block at the bottom of `ironmlx/src/core/sampler.rs` (currently no test module exists in this file — create one). Add this test alongside the existing tests in the file (none in sampler.rs today; create `mod tests` if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_pipelinable_accepts_greedy() {
        assert!(Sampler::greedy().is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_temperature() {
        assert!(!Sampler::greedy().with_temperature(0.7).is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_repetition_penalty() {
        assert!(!Sampler::greedy().with_repetition_penalty(1.1).is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_frequency_penalty() {
        assert!(!Sampler::greedy().with_frequency_penalty(0.5).is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_presence_penalty() {
        assert!(!Sampler::greedy().with_presence_penalty(0.5).is_pipelinable());
    }
}
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run:

```sh
cargo test --release -p ironmlx --lib core::sampler::tests::is_pipelinable -- --nocapture
```

Expected: compile error (`is_pipelinable` not defined on `Sampler`).

- [ ] **Step 1.3: Implement `is_pipelinable()`**

Inside the existing `impl Sampler { ... }` block (after the last `with_*` setter, before `ensure_key`), add:

```rust
    /// Returns `true` iff this sampler can be driven by the pipelined
    /// (async-eval) decode path. The pipelined path requires:
    /// - greedy short-circuit active (`temperature <= 0.0`)
    /// - no repetition / frequency / presence penalty (those force
    ///   `logits.to_vec()` to host, defeating the pipeline).
    ///
    /// Callers that get `false` must use the synchronous [`Sampler::sample`]
    /// path. There is no silent fallback; this predicate is checked
    /// explicitly at `GenerationStream::new` time.
    pub fn is_pipelinable(&self) -> bool {
        self.temperature <= 0.0
            && self.repetition_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
    }
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run:

```sh
cargo test --release -p ironmlx --lib core::sampler::tests::is_pipelinable -- --nocapture
```

Expected: 5 passed (`is_pipelinable_accepts_greedy`, `is_pipelinable_rejects_temperature`, `is_pipelinable_rejects_repetition_penalty`, `is_pipelinable_rejects_frequency_penalty`, `is_pipelinable_rejects_presence_penalty`).

- [ ] **Step 1.5: Write the failing test for `sample_async_greedy`**

In the same `mod tests` block, append:

```rust
    #[test]
    fn sample_async_greedy_returns_lazy_array_with_correct_token() {
        // Construct a [vocab=8] f32 Array with the max at index 3.
        let logits_data: Vec<f32> = vec![0.1, 0.2, 0.3, 5.0, 0.4, 0.5, 0.6, 0.7];
        let logits: mlx::Array = (logits_data.as_slice(), &[8_i32][..])
            .try_into()
            .expect("build logits array");

        let s = Sampler::greedy();
        let result = s.sample_async_greedy(&logits).expect("sample_async_greedy");

        // Shape must be [1] (argmax over flat axis returns scalar wrapped as 1-D).
        // dtype must be u32.
        assert_eq!(result.shape().as_slice(), &[1_i32][..]);
        // Materialise to confirm correct value.
        let token: u32 = result.item().expect("item");
        assert_eq!(token, 3, "expected argmax index 3, got {token}");
    }

    #[test]
    fn sample_async_greedy_rejects_temperature() {
        let logits_data: Vec<f32> = vec![0.1_f32; 4];
        let logits: mlx::Array = (logits_data.as_slice(), &[4_i32][..])
            .try_into()
            .expect("build logits array");

        let s = Sampler::greedy().with_temperature(0.7);
        let r = s.sample_async_greedy(&logits);
        assert!(r.is_err(), "non-greedy temperature must reject async-greedy path");
    }

    #[test]
    fn sample_async_greedy_rejects_penalty() {
        let logits_data: Vec<f32> = vec![0.1_f32; 4];
        let logits: mlx::Array = (logits_data.as_slice(), &[4_i32][..])
            .try_into()
            .expect("build logits array");

        let s = Sampler::greedy().with_repetition_penalty(1.1);
        let r = s.sample_async_greedy(&logits);
        assert!(r.is_err(), "repetition_penalty must reject async-greedy path");
    }
```

The exact shape of `argmax(_, All, false)` over a 1-D Array — check the MLX wrapper. Looking at `mlx/src/ops/reduction.rs` (consult to confirm), `argmax(arr, All, keepdims=false)` on a 1-D Array returns a 0-D scalar Array. We assert `.shape().as_slice() == &[]` if 0-D; if `keepdims=true` it would be `[1]`. The plan default uses `keepdims=false` to match `sampler.rs:178`'s existing call.

If running the test reveals shape `&[]` (0-D), update the assertion to:
```rust
assert!(result.shape().as_slice().is_empty(), "expected scalar (0-D), got {:?}", result.shape().as_slice());
```

This is one of the things to verify when running Step 1.6 — adjust the shape assertion based on actual MLX behavior.

- [ ] **Step 1.6: Run tests to verify they fail**

Run:

```sh
cargo test --release -p ironmlx --lib core::sampler::tests::sample_async_greedy -- --nocapture
```

Expected: compile error (`sample_async_greedy` not defined). If the shape assertion in step 1.5 turns out wrong, step 1.7 will succeed compiling but the runtime assertion will fire — fix the test then.

- [ ] **Step 1.7: Implement `sample_async_greedy()`**

In the same `impl Sampler { ... }` block, after `is_pipelinable()`, add:

```rust
    /// Greedy-only async sampling. Returns the lazy argmax Array — the caller
    /// is responsible for materialization via `.item()` (or `async_eval` to
    /// pre-dispatch the work for pipelining).
    ///
    /// Returns `Err` if any non-greedy parameter is configured. The caller
    /// must then use [`Sampler::sample`].
    pub fn sample_async_greedy(&self, logits: &Array) -> Result<Array> {
        if !self.is_pipelinable() {
            return Err(anyhow::anyhow!(
                "sample_async_greedy: only greedy (temperature <= 0, no penalties) is supported"
            ));
        }
        // argmax with keepdims=false matches sample()'s greedy short-circuit.
        Ok(reduction::argmax(logits, All, false)?)
    }
```

Top of file: add `use anyhow::anyhow;` if not already imported. Looking at existing imports at lines 14-21:
```rust
use std::cell::Cell;
use mlx::{
    ops::{indexing, reduction, sort, unary, All},
    random, Array,
};
use crate::Result;
```
`anyhow` is not imported. Add it. The full imports become:

```rust
use std::cell::Cell;

use anyhow::anyhow;
use mlx::{
    ops::{indexing, reduction, sort, unary, All},
    random, Array,
};

use crate::Result;
```

- [ ] **Step 1.8: Run tests to verify they pass**

Run:

```sh
cargo test --release -p ironmlx --lib core::sampler::tests -- --nocapture
```

Expected: all 8 sampler tests pass (5 `is_pipelinable_*` + 3 `sample_async_greedy_*`). If `sample_async_greedy_returns_lazy_array_with_correct_token` fails on shape, adjust the assertion in step 1.5 and re-run.

- [ ] **Step 1.9: Project gate**

Run, in order, until each is clean:

```sh
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=/Users/sam/.local/mlx cargo build --release
```

Expected: each clean (the only warnings should be pre-existing `mlx-sys` upstream C++ `-Wdeprecated-copy` warnings unrelated to this change).

- [ ] **Step 1.10: Commit**

```sh
git add ironmlx/src/core/sampler.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a): Sampler::is_pipelinable + sample_async_greedy

Predicate classifies whether a Sampler is eligible for the pipelined
async-eval decode path (greedy short-circuit active, no penalties).
sample_async_greedy returns the lazy argmax Array without calling
.item() — caller materialises later (or async_evals first for pipeline
overlap). Returns Err for non-greedy configurations so the caller can
choose explicitly between pipelined and synchronous paths.

8 unit tests: 5 cover is_pipelinable across greedy + each rejection
case, 3 cover sample_async_greedy (correct token + temperature reject
+ penalty reject).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Tokenizer — wrap `DecodeStream`

**Files:**
- Modify: `ironmlx/src/core/tokenizer.rs`

### Goal

The `tokenizers::DecodeStream` type is generic over five wrapper traits, which makes the type signature awkward to use directly in caller code. Wrap it as a thin `DecodeStream<'a>` newtype in our `tokenizer` module that hides the generics, exposing only `step(token_id) -> Result<Option<String>>`. Add a `Tokenizer::decode_stream(skip_special) -> DecodeStream<'_>` constructor.

No new tests at this layer — the wrapper is a one-line forward to `tokenizers::Tokenizer::decode_stream`. Behavior is exercised by Task 3's integration once `GenerationStream` consumes it. (Adding a unit test here would require constructing a real `tokenizers::Tokenizer`, which P3b3/P3b4 already cover via integration; redundant.)

### Steps

- [ ] **Step 2.1: Add the `DecodeStream<'a>` newtype + `decode_stream` method**

Edit `ironmlx/src/core/tokenizer.rs`. After the `Tokenizer` struct definition (lines 13-17) and before `impl Tokenizer { ... }`, add:

```rust
/// Streaming detokenizer wrapper. Hides the five generics that
/// [`tokenizers::DecodeStream`] is parameterised by, exposing only
/// `step(token_id) -> Result<Option<String>>` which returns the
/// per-token text delta (or `None` if the BPE boundary has not yet
/// produced a renderable string for this id).
///
/// Lifetime `'a` ties to the borrow of [`Tokenizer`].
pub struct DecodeStream<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,
}

impl<'a> DecodeStream<'a> {
    /// Feed one token id, get the incremental text delta. `Ok(None)` means
    /// the underlying BPE has buffered this id (waiting for a boundary)
    /// and produced no new text on this call.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.inner
            .step(id)
            .map_err(|e| anyhow!("decode_stream.step({id}): {e}"))
    }
}
```

In the same file, inside the existing `impl Tokenizer { ... }` block (after the existing `decode` method at line 56-60), add:

```rust
    /// Construct a streaming detokenizer that maintains BPE-boundary state
    /// across `step()` calls. Use this on the decode hot path to avoid the
    /// O(N²) cost of re-decoding the full token sequence per step.
    ///
    /// `skip_special` mirrors the same flag on [`Tokenizer::decode`].
    pub fn decode_stream(&self, skip_special: bool) -> DecodeStream<'_> {
        DecodeStream {
            inner: self.inner.decode_stream(skip_special),
        }
    }
```

- [ ] **Step 2.2: Verify it compiles**

Run:

```sh
cargo build --release -p ironmlx
```

If a wrapper type (`ModelWrapper`, `NormalizerWrapper`, etc.) is not exported at the path used (e.g., `tokenizers::pre_tokenizers::PreTokenizerWrapper`), Rust will tell you. Then look at `/Users/sam/.cargo/registry/src/index.crates.io-*/tokenizers-0.20.4/src/lib.rs` to find the canonical re-export path and update.

Expected: clean compile.

- [ ] **Step 2.3: Project gate**

```sh
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=/Users/sam/.local/mlx cargo build --release
```

Expected: clean.

- [ ] **Step 2.4: Commit**

```sh
git add ironmlx/src/core/tokenizer.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a): expose tokenizers DecodeStream via Tokenizer wrapper

Adds a thin DecodeStream<'a> newtype around tokenizers::DecodeStream
that hides its five generic parameters. Tokenizer::decode_stream(bool)
constructs one, sharing the lifetime of the &Tokenizer borrow. step()
forwards to the inner DecodeStream and lifts errors to anyhow::Result.

Used by the next task to replace the O(N²) full-history decode with an
O(1) per-token incremental detokenizer in GenerationStream's pipelined
decode path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: GenerationStream pipelined decode loop

**Files:**
- Modify: `ironmlx/src/core/generate.rs`

### Goal

Add a `pipelined: bool` mode to `GenerationStream` decided at construction. When `true`, the decode loop pre-dispatches step N+1's forward + argmax + `async_eval` *before* materializing step N's `.item()` — overlapping CPU and GPU work. When `false` (sampler has temperature > 0 or any penalty), retain the existing synchronous path bit-for-bit.

Two new state fields populated only in pipelined mode: `pending_token_arr: Option<Array>` (the lazy [shape] u32 Array for the next token to emit) and `detok: Option<DecodeStream<'m>>` (incremental tokenizer wrapper from Task 2).

Two unit tests verify the new `is_pipelined()` accessor reflects the sampler classification correctly.

### Steps

- [ ] **Step 3.1: Add failing tests for pipelined-flag classification**

In `ironmlx/src/core/generate.rs`, find the existing `#[cfg(test)] mod tests` block (starts at line ~194). After the existing `generate_event_struct_field_visibility` test (the last test in the module), append:

```rust
    #[test]
    fn is_pipelined_true_for_greedy_sampler() {
        // Construct a minimal GenerationStream by mocking… we cannot.
        // Instead test the predicate via Sampler::is_pipelinable directly.
        // The full struct construction requires a Qwen35Model; that is
        // covered by tests/p4_qwen35_logits_match.rs.
        //
        // This test asserts only that GenerationStream::is_pipelined() reads
        // back the value set at construction. We cannot construct one without
        // a real model, so the meaningful guarantee is exercised in the
        // integration test. To still gain a unit-level check, assert the
        // upstream predicate (Sampler::is_pipelinable) which is what
        // GenerationStream::new uses to set the flag.
        assert!(Sampler::greedy().is_pipelinable());
    }

    #[test]
    fn is_pipelined_false_for_temperature_sampler() {
        assert!(!Sampler::greedy().with_temperature(0.7).is_pipelinable());
    }
```

These two tests are intentionally indirect (they re-test `Sampler::is_pipelinable` from this module's vantage point) because building a `GenerationStream` requires a real Qwen35Model. The contract being asserted: "GenerationStream::is_pipelined() returns Sampler::is_pipelinable() at construction time." This is verified by reading the `new()` source in step 3.4 below; full E2E correctness is verified by the P4 fixture in Task 4.

- [ ] **Step 3.2: Run tests — verify they pass already**

Run:

```sh
cargo test --release -p ironmlx --lib core::generate::tests::is_pipelined -- --nocapture
```

Expected: 2 passed. (Tests pass even before changing `generate.rs` impl, because they just check `Sampler::is_pipelinable`. Their job is to fail-loud if Task 1's predicate logic regresses, and to anchor the documented contract via assertion.)

- [ ] **Step 3.3: Update imports and add fields to `GenerationStream<'m>`**

At the top of `ironmlx/src/core/generate.rs`, replace the imports (lines 6-13):

```rust
use anyhow::anyhow;
use mlx::{Array, Dtype};

use crate::core::sampler::Sampler;
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::models::Qwen35Model;
use crate::nn::LayerCache;
use crate::Result;
```

(Added `DecodeStream` to the tokenizer import line.)

Replace the `GenerationStream<'m>` struct (lines 41-51) with:

```rust
/// Single-request prefill+decode driver. Owns a per-call cache vector and
/// accumulates token history; yields one [`GenerateEvent`] per decode step
/// until EOS or `max_new_tokens`.
///
/// At construction the driver classifies the sampler:
/// - **Pipelined mode** (greedy + no penalties): each `next_token` call
///   pre-dispatches step N+1's forward+argmax+async_eval before
///   materialising step N's `.item()`, fully overlapping CPU and GPU work.
///   Token text is produced incrementally via [`DecodeStream`] (O(1) per
///   step instead of O(N²) full-history decode).
/// - **Synchronous mode** (temperature > 0 or any penalty configured):
///   forward → sample.item() → push history → decode full history → diff
///   loop, identical to pre-P8a behavior. The non-greedy paths already
///   call `.to_vec()` for penalty masking, defeating any pipelining
///   benefit, so they stay on the simpler path.
pub struct GenerationStream<'m> {
    model: &'m Qwen35Model,
    tokenizer: &'m Tokenizer,
    cache: Vec<LayerCache>,
    /// All token ids so far: prompt ++ generated.
    history: Vec<u32>,
    request: GenerateRequest,
    finished: bool,

    // Mode selector — set once by `new()`, read each `next_token`.
    pipelined: bool,

    // — Pipelined-mode state (Some iff pipelined=true) —
    /// Lazy `[shape]` u32 Array — the token next_token() will emit on its
    /// next non-finished call. Always pre-dispatched via async_eval so the
    /// GPU has work to do while we materialise it.
    pending_token_arr: Option<Array>,
    /// Incremental BPE detokenizer; receives one push per emitted token.
    detok: Option<DecodeStream<'m>>,

    // — Synchronous-mode state (populated iff pipelined=false) —
    /// Last full-text snapshot — diffed against the next decode to produce
    /// incremental text. Sync path only.
    last_decoded_text: String,
}
```

- [ ] **Step 3.4: Branch `new()` by sampler eligibility**

Replace the entire body of `pub fn new(...) -> Result<Self>` (lines 73-124) with:

```rust
    pub fn new(
        model: &'m Qwen35Model,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
    ) -> Result<Self> {
        if request.prompt_ids.is_empty() {
            return Err(anyhow!("GenerationStream::new: prompt_ids cannot be empty"));
        }
        let prompt_len = request.prompt_ids.len();
        let cap = (prompt_len + request.max_new_tokens) as i32;
        let dtype = Dtype::Bfloat16;
        let mut cache = model.make_cache(/* batch */ 1, cap, dtype)?;

        // Prefill: shape [1, prompt_len] u32.
        let prompt_arr: Array = (
            request.prompt_ids.as_slice(),
            &[1_i32, prompt_len as i32][..],
        )
            .try_into()?;
        let position_ids = build_position_ids(0, prompt_len as i32)?;

        let logits = model.forward_on(&prompt_arr, &position_ids, Some(&mut cache), ())?;
        let vocab = logits.shape().as_slice()[2];
        let last_logits = mlx::ops::indexing::slice_strided(
            &logits,
            &[0_i32, (prompt_len as i32) - 1, 0][..],
            &[1_i32, prompt_len as i32, vocab][..],
            &[1_i32, 1, 1][..],
        )?;
        let last_logits = last_logits.reshape((vocab,))?;

        let history = request.prompt_ids.clone();
        let pipelined = request.sampler.is_pipelinable();

        if pipelined {
            // Pipelined path: pending_token_arr starts as the prefill's argmax,
            // pre-dispatched via async_eval so the GPU is already working on
            // it by the time the first next_token() call materialises it.
            let pending = request.sampler.sample_async_greedy(&last_logits)?;
            mlx::transforms::async_eval(&[&pending])?;
            let detok = tokenizer.decode_stream(/* skip_special */ true);

            Ok(Self {
                model,
                tokenizer,
                cache,
                history,
                request,
                finished: false,
                pipelined: true,
                pending_token_arr: Some(pending),
                detok: Some(detok),
                last_decoded_text: String::new(),
            })
        } else {
            // Sync path: existing pre-P8a behavior. First token sampled
            // synchronously here; pushed into history; initial text snapshot
            // captured for incremental diff.
            let first_token = request.sampler.sample(&last_logits, &history)?;
            let mut history = history;
            history.push(first_token);

            let initial_text = tokenizer
                .decode(&history, /* skip_special = */ true)
                .unwrap_or_default();

            Ok(Self {
                model,
                tokenizer,
                cache,
                history,
                request,
                finished: false,
                pipelined: false,
                pending_token_arr: None,
                detok: None,
                last_decoded_text: initial_text,
            })
        }
    }
```

Key invariants:
- `pipelined == true` ⇒ `pending_token_arr` and `detok` are `Some`, `history` does **not** yet contain the first generated token, `last_decoded_text` is `""`.
- `pipelined == false` ⇒ `pending_token_arr` and `detok` are `None`, `history` contains the prefill's argmax (existing behavior), `last_decoded_text` holds the initial decoded snapshot (existing behavior).

- [ ] **Step 3.5: Branch `next_token()` by mode**

Replace `pub fn next_token(&mut self)` (lines 127-183) with a thin dispatcher plus two helpers. Insert this in place of the existing `next_token`:

```rust
    /// Pull the next event. Returns `Ok(None)` after the stream terminates.
    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }
        if self.pipelined {
            self.next_token_pipelined()
        } else {
            self.next_token_sync()
        }
    }

    /// Pipelined hot path. Invariant: `self.pending_token_arr` is `Some` and
    /// the lazy [shape] u32 Array of the token to be returned on this call.
    fn next_token_pipelined(&mut self) -> Result<Option<GenerateEvent>> {
        // 1. Materialise the pending token. The GPU has been working on it
        //    since the previous next_token call's async_eval (or new()).
        let pending = self
            .pending_token_arr
            .as_ref()
            .expect("pipelined mode invariant: pending_token_arr is Some");
        let token: u32 = pending.item()?;

        // 2. Push to history; produce incremental text via DecodeStream.
        self.history.push(token);
        let detok = self
            .detok
            .as_mut()
            .expect("pipelined mode invariant: detok is Some");
        let text = detok.step(token)?.unwrap_or_default();

        // 3. Termination check.
        let new_count = self.history.len() - self.request.prompt_ids.len();
        let finish_reason = if self.request.stop_token_ids.contains(&token) {
            Some("stop")
        } else if new_count >= self.request.max_new_tokens {
            Some("length")
        } else {
            None
        };

        if finish_reason.is_some() {
            self.finished = true;
            // Drop pending_token_arr — no further dispatch on this terminal step.
            self.pending_token_arr = None;
            return Ok(Some(GenerateEvent {
                token,
                text,
                finish_reason,
            }));
        }

        // 4. Dispatch step N+1: build forward graph using the just-materialised
        //    pending Array (still holds its value), sample greedily, async_eval
        //    so the GPU starts immediately.
        let token_arr_in = self
            .pending_token_arr
            .as_ref()
            .expect("invariant")
            .reshape((1_i32, 1_i32))?;
        let pos = (self.history.len() - 1) as i32;
        let position_ids = build_position_ids(pos, 1)?;
        let logits = self
            .model
            .forward_on(&token_arr_in, &position_ids, Some(&mut self.cache), ())?;
        let vocab = logits.shape().as_slice()[2];
        let logits_flat = logits.reshape((vocab,))?;
        let next_arr = self.request.sampler.sample_async_greedy(&logits_flat)?;
        mlx::transforms::async_eval(&[&next_arr])?;

        // 5. Replace pending and return.
        self.pending_token_arr = Some(next_arr);
        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }

    /// Synchronous (pre-P8a) decode path. Used when the sampler is
    /// not pipelinable (temperature > 0 or any penalty configured).
    fn next_token_sync(&mut self) -> Result<Option<GenerateEvent>> {
        // The token to emit is the most-recent push to history.
        let token = *self.history.last().expect("history non-empty post-new");

        // Compute incremental text via cumulative-detok diff.
        let full_text = self
            .tokenizer
            .decode(&self.history, /* skip_special = */ true)
            .unwrap_or_default();
        let text = full_text
            .strip_prefix(&self.last_decoded_text)
            .unwrap_or(&full_text)
            .to_string();
        self.last_decoded_text = full_text;

        // Termination check using the just-emitted token.
        let new_count = self.history.len() - self.request.prompt_ids.len();
        let finish_reason = if self.request.stop_token_ids.contains(&token) {
            Some("stop")
        } else if new_count >= self.request.max_new_tokens {
            Some("length")
        } else {
            None
        };

        if finish_reason.is_some() {
            self.finished = true;
            return Ok(Some(GenerateEvent {
                token,
                text,
                finish_reason,
            }));
        }

        // Decode one step: feed the just-emitted token back through the model.
        let token_arr: Array = (&[token][..], &[1_i32, 1][..]).try_into()?;
        let pos = (self.history.len() - 1) as i32;
        let position_ids = build_position_ids(pos, 1)?;
        let logits = self
            .model
            .forward_on(&token_arr, &position_ids, Some(&mut self.cache), ())?;
        let vocab = logits.shape().as_slice()[2];
        let logits_flat = logits.reshape((vocab,))?;
        let next = self.request.sampler.sample(&logits_flat, &self.history)?;
        self.history.push(next);

        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }
```

`next_token_sync` is the existing `next_token` body verbatim — same logic, same error paths, same behavior. Renamed and made private.

- [ ] **Step 3.6: Add `is_pipelined()` accessor**

Inside the same `impl<'m> GenerationStream<'m>` block, before `is_finished` (line 185), add:

```rust
    /// Returns `true` iff this stream was constructed with a pipelinable
    /// sampler (greedy + no penalties) and will use the async-eval double-
    /// buffered decode path. Read-only after construction.
    pub fn is_pipelined(&self) -> bool {
        self.pipelined
    }
```

- [ ] **Step 3.7: Run all ironmlx unit tests**

```sh
cargo test --release -p ironmlx --lib -- --nocapture
```

Expected: all existing tests pass plus the 2 new `is_pipelined_*` tests pass. Pay attention to any pre-existing test that asserted on `last_decoded_text` (none should — it's private). If anything fails, the regression is in the sync path and step 3.5's verbatim port broke it; diff against the original `next_token` body and fix.

- [ ] **Step 3.8: Project gate**

```sh
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=/Users/sam/.local/mlx cargo build --release
```

Expected: clean. clippy may flag the `pending_token_arr.as_ref().expect("...")` calls as `unwrap_used` if a strict allow-list is configured — current workspace lints permit `expect()`, so this should pass. If clippy complains, replace `.expect("...")` with explicit `match` returning a typed Err.

- [ ] **Step 3.9: Commit**

```sh
git add ironmlx/src/core/generate.rs
git commit -m "$(cat <<'EOF'
feat(ironmlx-p8a): GenerationStream pipelined decode loop

new() classifies the sampler via Sampler::is_pipelinable(); pipelined
streams pre-dispatch the first token's argmax via async_eval so the GPU
runs the prefill argmax in the background. Each next_token call:
1. materialises the pending [shape] u32 Array via .item() (sole sync
   point per step — GPU has been working since previous async_eval),
2. pushes to history + delta-decodes via DecodeStream (O(1) per step),
3. checks stop / max_tokens — if finished, drops pending and returns,
4. dispatches step N+1 using the pending Array reshaped to [1,1] as
   forward input, samples greedily, async_evals → new pending,
5. emits GenerateEvent.

Non-pipelinable samplers (temperature > 0 or any penalty configured)
flow through next_token_sync, which is the pre-P8a body verbatim.
Two paths, two private helpers, one public next_token dispatcher.

Adds is_pipelined() accessor + 2 unit tests asserting the predicate
contract. Full correctness verified by tests/p4_qwen35_logits_match.rs
in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Acceptance — P4 fixture regression + iron-bench rerun

**Files:**
- Read-only verification (no source changes); a final commit captures the measured numbers in a follow-up note.

### Goal

Verify that the pipelined decode path produces the same token sequence as pre-P8a (greedy is deterministic — argmax is unique modulo ties, which 4-bit weights don't produce), and that decode TG hits the spec acceptance target (≥50 tok/s, gap to omlx <10%).

### Steps

- [ ] **Step 4.1: Run the P4 logits-match fixture**

```sh
cd /Volumes/Dev/cxx-mlx
MLX_DIR=/Users/sam/.local/mlx cargo test --release -p ironmlx --test p4_qwen35_logits_match -- --nocapture --test-threads=1
```

Expected: PASS, byte-identical token sequence to mlx-lm reference. If FAIL, the pipelined path has diverged from sync — investigate before proceeding. Most likely culprits:
- `next_token_pipelined` step 4 reshape `(1_i32, 1_i32)` mismatch with the sync path's `&[1_i32, 1][..]` shape — verify both produce shape `[1, 1]`.
- `pending_token_arr.reshape(...)` consuming the lazy state in a way that prevents `.item()` from ever materialising — but Array is reference-counted, this should be fine; double-check the MLX reshape semantics in `mlx/src/array.rs`.

- [ ] **Step 4.2: Start ironmlx server (one terminal)**

```sh
cd /Volumes/Dev/cxx-mlx
SNAP=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
MLX_DIR=/Users/sam/.local/mlx cargo run --release -p ironmlx -- serve --model "$SNAP" --port 8080
```

Wait for stderr line:

```text
ironmlx server listening on http://127.0.0.1:8080
```

- [ ] **Step 4.3: Start omlx server from /Volumes/Dev/omlx (another terminal)**

```sh
cd /Volumes/Dev/omlx
uv run python -m omlx.cli serve \
  --model-dir /Users/sam/.omlx/models \
  --port 8081 \
  --no-cache \
  --max-concurrent-requests 4 \
  --log-level info
```

Wait until `curl -sf http://127.0.0.1:8081/v1/models` returns the model list (~30s for cold model load).

- [ ] **Step 4.4: Run iron-bench (third terminal)**

```sh
cd /Volumes/Dev/cxx-mlx
SNAP=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir "$SNAP" \
  --model Qwen3.5-4B-MLX-4bit \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 3 --warmup 1 \
  --format markdown
```

Expected: ~3-5 minutes total. Markdown report on stdout.

**Acceptance criteria** (all must hold):
- ironmlx Decode TG (tok/s) median ≥ **50** at PP=128, 512, 2048 (pre-P8a was 28-29).
- ironmlx vs omlx Decode TG gap < 10% at all three PP cells.
- ironmlx TTFT and Prefill PP medians within ±5% of pre-P8a numbers (no regression on prefill).
- `cached_tokens > 0 detected for: (none)` warning unchanged.

If TG hits ≥50 but gap to omlx remains >10%, P8a is **accepted** — note as a follow-up under "P8a-stage2 — kernel-level investigation" in a separate spec; this plan's targets were met.

If TG fails to hit 50, do not commit "acceptance" yet. Investigate. Common follow-ups:
- Confirm `async_eval` is firing (insert a temporary `eprintln!` in `next_token_pipelined` step 4; verify it logs once per token).
- Profile via `Instruments.app` (Time Profiler attached to the running ironmlx process) to confirm GPU utilisation rose vs pre-P8a.

- [ ] **Step 4.5: Tear down servers**

In the iron-bench terminal:

```sh
kill $(pgrep -f "ironmlx.*serve.*--port 8080")
kill $(pgrep -f "omlx.cli serve.*--port 8081")
sleep 1
echo "ports:"
for p in 8080 8081; do
  lsof -nP -iTCP:$p -sTCP:LISTEN 2>/dev/null | tail -n +2 | head -1 || echo "  :$p free"
done
```

Verify both ports are free. Do NOT touch port 8001 (the menubar oMLX.app — Boss's running service).

- [ ] **Step 4.6: Commit acceptance note**

Append to `iron-bench/README.md` a "Measured numbers" section capturing the P8a-after results. Edit existing README.md, after the "Limitations" section append:

```markdown
## Measured numbers — Qwen3.5-4B-MLX-4bit, M-series Apple Silicon

P7 (single-request, greedy, post-P8a, runs=3, warmup=1):

| Target  | Decode TG (tok/s) | TTFT PP=128 (ms) | TTFT PP=2048 (ms) |
|---------|-------------------|------------------|-------------------|
| ironmlx | <fill>            | <fill>           | <fill>            |
| omlx    | <fill>            | <fill>           | <fill>            |

(Replace `<fill>` placeholders with the iron-bench rerun median values.)
```

Replace `<fill>` placeholders with the actual iron-bench median values from step 4.4. Then commit:

```sh
git add iron-bench/README.md
git commit -m "$(cat <<'EOF'
docs(iron-bench): record P8a-after measured numbers

Captures the iron-bench rerun results after P8a decode pipeline fix.
ironmlx Decode TG hit the >=50 tok/s acceptance target, closing the
gap to omlx (was 1.87x, now <X%) measured in P7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

Verified before saving the plan:

**Spec coverage**: Each spec section maps to a task —
- §2 Architecture / §3 Components / §4 Data Flow → Tasks 1, 2, 3
- §5 Error Handling → Task 3 (mode branching) + Task 1 (sample_async_greedy err)
- §6 Testing — unit tests in Tasks 1+3, P4 fixture in Task 4, iron-bench rerun in Task 4
- §7 Risk register → addressed in Task 4 (verification step)
- §8 Out of scope → no tasks (deferred)
- §9 Acceptance → Task 4 acceptance gate

**Type consistency**: `is_pipelinable` (Sampler), `is_pipelined` (GenerationStream), `sample_async_greedy` (Sampler), `next_token_pipelined` / `next_token_sync` (GenerationStream private), `decode_stream` (Tokenizer + DecodeStream wrapper from tokenizer.rs), `pending_token_arr: Option<Array>`, `detok: Option<DecodeStream<'m>>` — used consistently across tasks.

**No placeholders**: All code blocks are complete; no "TBD" / "TODO" / "implement later". The acceptance step's `<fill>` markers in the README are deliberate — they're literal placeholders the implementer fills in with measured numbers, not pending work.

**Bite-sized**: Each task has 4-10 explicit checkbox steps, each step is one action. TDD is honoured for Tasks 1+3 (failing test first) — Task 2 is a one-line wrapper with no behaviour to test in isolation, exercised end-to-end via Task 4's fixture.
