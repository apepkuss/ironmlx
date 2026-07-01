# Gemma4 Long-Context Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and fix the Gemma4 12B target-model long-context parity divergence that collapses drafter acceptance around 20K+ tokens.

**Architecture:** Keep drafter policy out of this phase. Add repeatable, ignored real-model parity diagnostics that compare IronMLX against mlx-lm/mlx-vlm reference artifacts at fixed prompt lengths, then make the smallest production fix once a single root cause is confirmed. Any permanent diagnostic API must stay hidden/test-oriented and must not affect normal generation.

**Tech Stack:** Rust, MLX C++ bindings, mlx-lm/mlx-vlm Python reference scripts, `.npy`/JSON fixtures, ignored cargo integration tests.

## Global Constraints

- Always answer Boss in Chinese outside code and docs.
- Do not add compatibility code.
- Do not change Gemma4 drafter policy while investigating target-model parity.
- Do not attempt production fixes before a reproducible failing parity test exists.
- When Rust production code changes, run `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, and `cargo build --release`.
- Preserve existing Qwen and Gemma4 short-context behavior.

---

### Task 1: Reference Fixture Generator

**Files:**
- Create: `ironmlx/tests/fixtures/gemma4_long_context/README.md`
- Create: `ironmlx/tests/fixtures/gemma4_long_context/gen_reference.py`

**Interfaces:**
- Consumes: `GEMMA4_LONG_CONTEXT_MODEL=/path/to/mlx-community/gemma-4-12B-it-4bit/snapshot`
- Produces: `case_<tokens>_input_ids.npy`, `case_<tokens>_expected.json`, `case_<tokens>_expected_layer_last_hiddens.npy`, and `case_<tokens>_expected_layer0_stage_last_hiddens.npy`

- [ ] **Step 1: Write the fixture README**

Document the command:

```bash
GEMMA4_LONG_CONTEXT_MODEL=$HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/<snapshot> \
  uv run python ironmlx/tests/fixtures/gemma4_long_context/gen_reference.py --tokens 18000 --tokens 19900 --tokens 20000 --tokens 24000
```

Document that fixtures are intentionally not checked in because they are model-specific and large.

- [ ] **Step 2: Implement `gen_reference.py`**

The script must:

```text
1. Load the reference model with mlx-lm/mlx-vlm.
2. Build deterministic repeated-line prompts trimmed to requested tokenizer lengths.
3. Save input ids as `.npy`.
4. Run greedy generation for at least 8 tokens and save token ids plus decoded text.
5. Save last-token top-k logits after the prompt.
6. Append the first generated token and save the after-append top-k logits.
7. Save layer-last and layer0-stage traces for the after-append state.
```

- [ ] **Step 3: Run the generator once**

Run for `--tokens 18000 --tokens 19900 --tokens 20000 --tokens 24000`.

Expected: each case has one `.json` and three `.npy` files.

- [ ] **Step 4: Commit Task 1**

```bash
git add ironmlx/tests/fixtures/gemma4_long_context/README.md ironmlx/tests/fixtures/gemma4_long_context/gen_reference.py
git commit -m "test(gemma4): add long-context reference fixtures"
```

### Task 2: IronMLX Long-Context Parity Harness

**Files:**
- Create: `ironmlx/tests/gemma4_long_context_parity.rs`
- Modify only if necessary: `ironmlx/src/models/gemma4/model.rs`
- Modify only if necessary: `ironmlx/src/models/gemma4/text_model.rs`

**Interfaces:**
- Consumes: `GEMMA4_LONG_CONTEXT_MODEL=/path/to/snapshot`
- Consumes: fixtures from `ironmlx/tests/fixtures/gemma4_long_context`
- Produces: ignored cargo tests that print top-k, greedy-token, layer, and stage deltas.

- [ ] **Step 1: Write the failing ignored test first**

Add an ignored test named `gemma4_12b_long_context_20000_after_append_matches_reference`. It must:

```text
1. Skip with stderr if `GEMMA4_LONG_CONTEXT_MODEL` is unset.
2. Load `case_20000_input_ids.npy` and `case_20000_expected.json`.
3. Run IronMLX prompt prefill, append the reference first token, and compute next logits.
4. Assert that IronMLX greedy token after append equals reference greedy token.
```

- [ ] **Step 2: Verify RED**

Run:

```bash
GEMMA4_LONG_CONTEXT_MODEL=$HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/<snapshot> \
  cargo test --release --ignored -p ironmlx gemma4_12b_long_context_20000_after_append_matches_reference -- --test-threads=1
```

Expected: FAIL with greedy mismatch at 20K.

- [ ] **Step 3: Add layer/stage reporting to the same ignored test**

The test should print:

```text
prompt_top_k
after_append_top_k
layer_00..layer_47 max_abs
stage_00..stage_09 max_abs
```

If existing hidden no-cache trace APIs cannot inspect the after-append cached state, add the narrowest hidden helper needed for test diagnostics. The helper must not alter normal forward behavior.

- [ ] **Step 4: Add non-regression passing cases**

Add ignored tests for 18K and 19.9K. They should use the same helper and either assert exact greedy parity or print the measured top-k margin if reference and IronMLX disagree.

- [ ] **Step 5: Commit Task 2**

```bash
git add ironmlx/tests/gemma4_long_context_parity.rs ironmlx/src/models/gemma4/model.rs ironmlx/src/models/gemma4/text_model.rs
git commit -m "test(gemma4): add long-context target parity harness"
```

### Task 3: Root-Cause Isolation

**Files:**
- Modify only the file proven by Task 2 diagnostics. Expected candidates:
  - `ironmlx/src/models/gemma4/attention.rs`
  - `ironmlx/src/models/gemma4/rope.rs`
  - `ironmlx/src/models/gemma4/ops.rs`
  - `ironmlx/src/nn/rms_norm.rs`
  - `ironmlx/src/nn/linear.rs`
  - `ironmlx/src/nn/embedding.rs`

**Interfaces:**
- Consumes: Task 2 failing test and trace output.
- Produces: one production fix with a failing-then-passing parity test.

- [ ] **Step 1: State one hypothesis**

Record one hypothesis before changing code, for example:

```text
Hypothesis: IronMLX applies X in dtype/order Y while reference applies X in dtype/order Z, causing the first meaningful divergence at layer/stage N.
```

- [ ] **Step 2: Test the hypothesis minimally**

Use temporary diagnostic edits only if needed. Revert temporary edits before production commit.

- [ ] **Step 3: Write or tighten the failing test**

The test must fail for the confirmed root cause before production code changes.

- [ ] **Step 4: Implement the minimal production fix**

Change only the proven component. Do not bundle drafter policy changes or unrelated cleanup.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
GEMMA4_LONG_CONTEXT_MODEL=$HOME/.ironmlx/models/models--mlx-community--gemma-4-12B-it-4bit/snapshots/<snapshot> \
  cargo test --release --ignored -p ironmlx gemma4_12b_long_context_20000_after_append_matches_reference -- --test-threads=1
```

Expected: PASS.

- [ ] **Step 6: Run required Rust checks**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

- [ ] **Step 7: Commit Task 3**

```bash
git add ironmlx/src/models/gemma4 ironmlx/src/nn ironmlx/tests/gemma4_long_context_parity.rs
git commit -m "fix(gemma4): restore long-context target parity"
```

### Task 4: Post-Fix Drafter Retest

**Files:**
- Create: `docs/benchmarks/gemma4-drafter-long-context/<timestamp>/summary.md`
- Create: `docs/benchmarks/gemma4-drafter-long-context/<timestamp>/*.json`

**Interfaces:**
- Consumes: fixed target model and existing Gemma4 drafter support.
- Produces: base vs drafter metrics for 12B at 18K, 20K, and 24K.

- [ ] **Step 1: Run base and drafter benchmarks**

Run base and `d=2` first. Add `d=3` only if `d=2` no longer collapses.

- [ ] **Step 2: Summarize acceptance and latency**

Record TTFT, decode ms, E2E ms, TPS, rollback count, and acceptance.

- [ ] **Step 3: Commit benchmark summary**

```bash
git add docs/benchmarks/gemma4-drafter-long-context
git commit -m "docs(gemma4): record long-context drafter retest"
```
