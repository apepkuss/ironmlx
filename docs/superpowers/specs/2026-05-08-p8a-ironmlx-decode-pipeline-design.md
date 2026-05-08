# P8a — ironmlx Decode Pipeline Fix (Design Spec)

**Goal**: lift ironmlx single-request decode TG from ~29 tok/s to ≥50 tok/s on Qwen3.5-4B-MLX-4bit, closing the gap to omlx (~54 tok/s) measured in P7.

**Approach**: introduce a GPU/CPU double-buffered decode loop using `mlx::transforms::async_eval`, plus an incremental BPE detokenizer using tokenizers' built-in `decode_stream`. Greedy path only (matches benchmark + production temp=0 use cases).

**Scope**: changes confined to `ironmlx/src/core/{generate.rs, sampler.rs}`. No mlx-sys / mlx layer changes. No HTTP/server changes. No model layer changes.

---

## 1. Motivation

P7 iron-bench (commits `a6a577e..e1f72b8`) measured the following on Qwen3.5-4B-MLX-4bit, single request, greedy decode, both engines reading the same checkpoint:

| Metric | ironmlx | omlx | Gap |
|---|---|---|---|
| Decode TG (tok/s) | 28-29 | 54-55 | omlx 1.87× |
| Prefill PP=2048 (tok/s) | 239 | 292 | omlx 1.22× |
| TPOT (ms/tok) | 35 | 19 | omlx 1.86× |

Decode dominates production latency (TG ≫ PP for typical max_tokens=128). A read-only investigation of ironmlx's hot path identified two root causes:

**H1 — Synchronous `.item()` in sampler (HIGH)**
[`ironmlx/src/core/sampler.rs:179`](../../../ironmlx/src/core/sampler.rs#L179) and [`:208`](../../../ironmlx/src/core/sampler.rs#L208) call `.item::<u32>()` on the sampled token Array per decode step. `.item()` blocks the calling thread until the Metal command buffer completes. The next forward pass cannot be dispatched until `.item()` returns. Result: zero GPU/CPU overlap.

mlx-lm's equivalent (`mlx_lm/generate.py:460`) calls `mx.async_eval(next_y, next_logprobs)` to dispatch step N+1's GPU work *before* blocking on step N's `.item()` (`generate.py:466`). GPU runs N+1 while CPU is reading N. Full overlap.

**H6 — O(N²) tokenizer decode (MEDIUM)**
[`ironmlx/src/core/generate.rs:138`](../../../ironmlx/src/core/generate.rs#L138) calls `tokenizer.decode(&self.history, ...)` on the full accumulated sequence per step. For an N-token decode the cumulative tokenizer CPU work is O(N²).

mlx-lm's `stream_generate` uses an incremental detokenizer (`detokenizer.add_token(token)`) that maintains rolling state — O(1) per step.

mlx-sys + mlx already expose `async_eval` (binding `async_eval_many` in `mlx-sys/shim/src/stream.cc:109`, safe wrapper `mlx::transforms::async_eval` in `mlx/src/transforms.rs:58`, tested under `mlx/tests/p2a_async.rs` + `mlx/tests/p5_7_stream.rs`). No upstream binding work needed.

tokenizers 0.20.4 exposes `Tokenizer::decode_stream(skip_special_tokens) -> DecodeStream` with `step(token_id) -> Result<Option<String>>` returning the per-token text delta. Upstream handles BPE / byte-level / sentencepiece boundary correctness.

H1 + H6 together are sufficient to explain the observed 1.87× gap. Other hypotheses investigated (KV cache layout, attention kernel choice, RoPE / norm placement) ruled out — those paths match mlx-lm.

---

## 2. Architecture

### 2.1 Current decode loop (per `next_token()` call)
```text
┌────────────────────────────────────────────────────────────────┐
│ 1. detokenize history (O(N) — full BPE re-decode)              │
│ 2. forward(token_N)              — dispatched, lazy graph      │
│ 3. argmax → sample.item()        — BLOCKS until GPU done       │
│ 4. push to history; emit event                                 │
└────────────────────────────────────────────────────────────────┘
```
GPU is idle during steps 1 and 4 (CPU work) and during the kernel-launch wait portion of step 3.

### 2.2 P8a decode loop (greedy path)
```text
Invariant: pending_token_arr is a lazy [1] u32 Array — the token
that next_token() will return on the next call.

┌────────────────────────────────────────────────────────────────┐
│ 1. token = pending_token_arr.item()    — single sync per step  │
│ 2. push to history; detok.step(token)  — O(1)                  │
│ 3. stop / max_tokens check; if finished, return finish event   │
│ 4. dispatch step N+1:                                          │
│      token_arr  = pending_token_arr.reshape((1,1))             │
│      logits     = model.forward_on(token_arr, ...)             │
│      next_arr   = sampler.sample_async_greedy(logits)          │
│      async_eval(&[&next_arr])  ← fire-and-forget               │
│ 5. pending_token_arr = next_arr; emit event                    │
└────────────────────────────────────────────────────────────────┘
```
By the time step 1 of call N+1 runs, the GPU has been busy executing step 4's dispatch from call N. The blocking `.item()` waits only on residual command-queue time, not full-step latency.

### 2.3 Mode selection
GenerationStream classifies its sampler at construction:

```text
pipelined = (sampler.temperature <= 0.0)
         && sampler.repetition_penalty.is_none()
         && sampler.frequency_penalty.is_none()
         && sampler.presence_penalty.is_none()
```

(Matches existing `Sampler::sample()` greedy short-circuit at `sampler.rs:177` which also uses `<= 0.0`.)

When `pipelined == true` → use the new path. When `pipelined == false` (any non-greedy parameter present) → fall through to the existing synchronous path unchanged. No tri-state, no silent fallback — explicit branch decided once at `new()`.

The non-pipelined path is left as-is for two reasons:
1. `apply_repetition_penalty` / `apply_freq_presence_penalty` already pull `logits.to_vec()` to CPU (`sampler.rs:219, 238`), forcing sync. Pipelining cannot bypass them without rewriting penalties as MLX scatter ops — out of scope per Boss's feedback ("仅 greedy" scope).
2. `random::categorical` for stochastic sampling already calls `.item()` (`sampler.rs:208`); same constraint as above.

---

## 3. Components

### 3.1 `sampler.rs` — new method `sample_async_greedy`

```rust
impl Sampler {
    /// Greedy-only async sampling. Returns the lazy [1] u32 Array — caller is
    /// responsible for materialization via `.item()` (or `async_eval` to
    /// pre-dispatch the work).
    ///
    /// Returns Err if any non-greedy parameter is configured. The caller must
    /// then use the synchronous `sample()` path.
    pub fn sample_async_greedy(&self, logits: &Array) -> Result<Array> {
        if self.temperature > 0.0
            || self.repetition_penalty.is_some()
            || self.frequency_penalty.is_some()
            || self.presence_penalty.is_some()
        {
            bail!("sample_async_greedy: only greedy (temp=0, no penalties) supported");
        }
        let idx = reduction::argmax(logits, All, false)?;
        Ok(idx) // shape [1], dtype u32, lazy
    }
}
```

- No `.item()` call. No `to_vec()`. Pure MLX op.
- The check exists so the caller can decide pipelined vs sync path *before* allocating any state. Returning an Err in `new()` is cheap and explicit.

### 3.2 `generate.rs` — `GenerationStream<'m>` field additions

```rust
pub struct GenerationStream<'m> {
    model: &'m Qwen35Model,
    cache: KvCache,
    request: GenerateRequest,
    history: Vec<u32>,
    finished: bool,

    // — pipeline state, only populated when `pipelined == true` —
    pipelined: bool,
    pending_token_arr: Option<Array>,            // lazy [1] u32; None when !pipelined
    detok: Option<DecodeStream<'m, /*generics*/>>, // None when !pipelined

    // — legacy state, only populated when `pipelined == false` —
    last_decoded_text: String,                   // for sync path's strip_prefix diff
    tokenizer: &'m Tokenizer,
}
```

- Two state sets, mutually exclusive based on `pipelined`. No state duplication.
- `pending_token_arr: Option<Array>` rather than `Array` because a `pipelined=false` stream legitimately has no pending lazy token.
- `tokenizer: &'m Tokenizer` retained for the sync path. The pipelined path uses `detok` exclusively (which already borrows `&'m Tokenizer` internally).

### 3.3 `generate.rs` — `next_token()` dispatch

```rust
pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
    if self.finished { return Ok(None); }
    if self.pipelined { self.next_token_pipelined() } else { self.next_token_sync() }
}
```

`next_token_sync()` is the existing implementation, renamed.

`next_token_pipelined()` is new (pseudocode in §2.2 above).

### 3.4 `new()` — pipeline initialization

```rust
let pipelined = sampler.is_pipelinable();   // helper on Sampler
let last_logits = /* prefill, slice last position */;

if pipelined {
    let pending = sampler.sample_async_greedy(&last_logits)?;
    mlx::transforms::async_eval(&[&pending])?;          // pre-dispatch
    let detok = tokenizer.decode_stream(/*skip_special*/ true);
    Ok(Self { ..., pipelined: true, pending_token_arr: Some(pending), detok: Some(detok), ... })
} else {
    let first_token = sampler.sample(&last_logits, &history)?;  // existing sync path
    history.push(first_token);
    let initial_text = tokenizer.decode(&history, true).unwrap_or_default();
    Ok(Self { ..., pipelined: false, pending_token_arr: None, detok: None, last_decoded_text: initial_text, ... })
}
```

Note: in the pipelined branch, the first token is **not** pushed to history in `new()` — it's pushed on the first `next_token_pipelined()` call after `.item()` materialization. This matches the loop invariant ("pending_token_arr always represents the token next_token() will emit next").

In the sync branch, the first token **is** pushed in `new()` (existing behavior preserved).

This asymmetry is deliberate. Trying to unify them would force the pipelined path to materialize the first token in `new()` (defeating the whole point of pre-dispatch) or the sync path to defer materialization (changing existing tested behavior). Two separate branches keep both paths optimal.

---

## 4. Data Flow

### 4.1 Pipelined call sequence (max_tokens = 3 example)

```text
new():
  prefill → last_logits → sample_async_greedy → pending = arr_T0  (lazy)
  async_eval(arr_T0)   ← fires GPU work for T0
                       ← GPU computes T0 in background
  return GenerationStream

next_token() #1:
  [step 1] T0 = pending.item()           ← waits on GPU; T0 is materialized
  [step 2] history.push(T0); detok.step(T0) → text_delta_0
  [step 3] not stop; not max
  [step 4] dispatch step 1:
             token_arr_in = pending.reshape((1,1))
             logits = forward(token_arr_in)
             arr_T1 = argmax(logits)        ← lazy
             async_eval(arr_T1)              ← GPU starts T1
  [step 5] pending = arr_T1
  return GenerateEvent { T0, text_delta_0, finish_reason: None }

next_token() #2:
  [step 1] T1 = pending.item()           ← GPU has been working on T1 since #1.step4
  [step 2] history.push(T1); detok.step(T1) → text_delta_1
  ...
  [step 4] dispatch step 2: forward(T1) → arr_T2; async_eval(arr_T2)
  return GenerateEvent { T1, text_delta_1, None }

next_token() #3:
  [step 1] T2 = pending.item()
  [step 2] history.push(T2); detok.step(T2) → text_delta_2
  [step 3] new_count == max_new_tokens → finish_reason = Some("length")
  return GenerateEvent { T2, text_delta_2, Some("length") }
  ↑ does NOT dispatch step 3. arr_T3 work is never wasted.

next_token() #4:
  finished → return None
```

### 4.2 Pipeline depth
The pipeline is depth-1 (one step look-ahead). Depth-2 (look ahead by 2) is theoretically possible but introduces complexity around stop-token-mid-pipeline cleanup with no proven benefit on M-series silicon. Depth-1 is what mlx-lm uses and is sufficient to fully overlap CPU and GPU when CPU work per step is roughly equal to or less than GPU work per step (which is the regime here: ~5ms CPU vs ~25ms GPU per token).

---

## 5. Error Handling

| Scenario | Behavior |
|---|---|
| Non-greedy sampler (temp>0 / penalty) | `new()` detects via `sample_async_greedy` Err → `pipelined = false` → uses synchronous path. No silent fallback; explicit branch. |
| `mlx::transforms::async_eval` returns Err | Propagated via `?`. MLX `async_eval` failures are limited to OOM / Metal driver faults, where stream termination is the correct response. |
| Stop token triggers mid-pipeline | After step 3 sets `finished = true`, control returns *before* step 4. The next dispatch is never issued. No GPU work wasted. |
| `max_new_tokens` boundary | Same as stop token — finish branch returns before step 4 dispatch. |
| `pending.item()` fails (rare; e.g., context cancelled) | Propagated via `?`. GenerationStream becomes unusable; caller drops it; cache + lazy arrays released via MLX refcount. |
| `detok.step(token)` returns `Err` | Soft-degrade with `.unwrap_or_default()` (mirrors existing `tokenizer.decode().unwrap_or_default()` behavior on `generate.rs:139`). The token is still emitted; only the text delta is empty. |
| HTTP client disconnect | Existing `server.rs` cancellation path unchanged. GenerationStream drop releases pending Array + DecodeStream + KvCache. |

---

## 6. Testing

### 6.1 New unit tests

**`ironmlx/src/core/sampler.rs#tests`**
1. `sample_async_greedy_returns_array_shape_1`
   - Construct logits Array of shape [vocab=8] with deterministic max at index 3.
   - `Sampler::greedy().sample_async_greedy(&logits)` → assert shape == [1], dtype == u32.
   - Materialize via `.item::<u32>()` and assert == 3.
2. `sample_async_greedy_rejects_temperature`
   - `Sampler::greedy().with_temperature(0.7).sample_async_greedy(&logits)` → `assert!(result.is_err())`.

**`ironmlx/src/core/generate.rs#tests`** (extends existing module)
3. `is_pipelinable_for_greedy`
   - `Sampler::greedy().is_pipelinable()` → true.
4. `is_pipelinable_for_temperature`
   - `Sampler::greedy().with_temperature(0.7).is_pipelinable()` → false.

`is_pipelinable()` is a public predicate on `Sampler`, used by `GenerationStream::new()` and exposed for testing.

End-to-end behavior is covered by §6.2 below; we do not add a `GenerationStream` unit test because constructing one requires a real `Qwen35Model` (no trait/dyn dispatch — Boss memory).

### 6.2 Integration regression — P4 logits-match fixture

`ironmlx/tests/p4_qwen35_logits_match.rs` (already present, P4) drives ironmlx greedy decode against a known-good token sequence captured from mlx-lm. With `pipelined=true` enabled by default for greedy, this fixture must still produce byte-identical output:

- greedy is deterministic (argmax is unique up to numerical ties, which 4-bit weights don't produce);
- pipelining doesn't change *what* gets sampled, only *when*;
- expected token sequence is identical pre- and post-P8a.

If the fixture fails post-P8a, that's a correctness bug — root-cause and fix before claiming completion.

### 6.3 Performance verification — iron-bench rerun

After all unit tests + project gate pass:

1. Start ironmlx :8080 against `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/.../snapshot/`.
2. Start omlx :8081 from `/Volumes/Dev/omlx` via `uv run python -m omlx.cli serve --model-dir ~/.omlx/models --port 8081 --no-cache`.
3. Run iron-bench:
   ```
   cargo run --release -p iron-bench -- \
     --target ironmlx=http://localhost:8080 \
     --target omlx=http://localhost:8081 \
     --model-dir <snap> --model Qwen3.5-4B-MLX-4bit \
     --prompt-len 128,512,2048 --max-tokens 128 --runs 3 --warmup 1
   ```

**Acceptance**:
- ironmlx Decode TG median ≥ **50 tok/s** at all three PP cells (current 28-29).
- ironmlx vs omlx Decode TG gap < 10% (current 1.87×).
- ironmlx TTFT and PP medians within ±5% of pre-P8a numbers (this change touches decode only; prefill should be unaffected).
- `cached_tokens > 0 detected for: (none)` warning unchanged.

If TG hits ≥ 50 but gap to omlx remains > 10%, consider stage 2 follow-up (P8a-stage2 — kernel-level investigation), but P8a itself is accepted.

### 6.4 Project gate (per commit)

```
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=/Users/sam/.local/mlx cargo build --release
```

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `async_eval(&[&next_arr])` does not include KV cache writes in the dispatched graph | Low | High (would break correctness silently) | MLX builds the graph via reference counting — cache buffer is a dependency of the forward output that next_arr depends on, so it's transitively included. Already verified by `mlx/tests/p2a_async.rs::async_eval_multiple_arrays`. P4 logits-match fixture (§6.2) will catch any divergence. |
| `decode_stream.step()` behavior on stop tokens differs from cumulative-decode | Low | Low (cosmetic — text delta) | Stop-token detection happens before the *next* `step()` call, so the stream state is consistent. tokenizers' DecodeStream is the same impl Python `transformers` and mlx-lm use. |
| `pipelined=true` adds dispatch overhead at very low max_tokens (e.g., max=1) | Very Low | Negligible | At max=1 the loop dispatches one extra step that's never read. Cost ≤ 5ms (kernel launch) — invisible at human latency scale. Worst-case test vector: max_tokens=1 path returns correct stop event. |
| Self-borrow of `DecodeStream<'m>` referencing `&'m Tokenizer` while held inside `GenerationStream<'m>` | Already mitigated | — | `GenerationStream<'m>` already has a `'m` lifetime borrowing `&'m Tokenizer` (see existing struct definition). DecodeStream just attaches to that same lifetime. |
| Pipelined path produces different token sequence than sync path under floating-point edge cases | Very Low | High | `argmax` is deterministic; the only nondeterminism source is execution order in MLX's stream worker, but argmax over a single tensor is invariant to that. P4 fixture catches any divergence — both engines must match mlx-lm's reference token IDs. |
| MLX `async_eval` blocks under M-series Metal driver faults / OOM, hanging the request indefinitely | Low | Medium | MLX's `async_eval` is documented as non-blocking submission; failures surface at the next `.item()`. server.rs request timeout (already configured per `ironmlx serve`) catches indefinite hangs. |

---

## 8. Out of Scope (deferred to future phases)

- **Non-greedy sampler pipelining** — would require pushing `apply_repetition_penalty` / `apply_freq_presence_penalty` from host `Vec<f32>` to MLX scatter ops. Likely needs new `mlx-sys` bindings if `mlx::ops::indexing::scatter` doesn't cover the case. Defer to a future "P8a-stage2 sampler MLX-ification" phase if production traffic shows demand.
- **Multi-request batching / scheduler** — handled by P8b (already planned).
- **Speculative decoding** — handled by P8c (already planned, MTP weights prepared).
- **Prefill optimization** — out of scope; P7 showed prefill gap is small (1.22×) and likely kernel-bound, not orchestration-bound.
- **Detokenizer streaming optimization beyond `decode_stream`** — future phase if profiling shows detok is still a bottleneck.

---

## 9. Acceptance Criteria

- [ ] All four new unit tests pass (sampler×2 + generate×2).
- [ ] `cargo test --release -p ironmlx` runs the full ironmlx test suite to completion. `tests/p4_qwen35_logits_match.rs` passes byte-identical to pre-P8a.
- [ ] iron-bench rerun shows ironmlx Decode TG median ≥ 50 tok/s at all three PP cells.
- [ ] iron-bench rerun shows ironmlx vs omlx Decode TG gap < 10%.
- [ ] iron-bench rerun shows ironmlx TTFT / PP medians within ±5% of pre-P8a (sanity).
- [ ] `cargo +nightly fmt --all -- --check` clean.
- [ ] `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` clean.
- [ ] `MLX_DIR=/Users/sam/.local/mlx cargo build --release` clean.

---

## 10. Implementation Sequencing (preview for writing-plans)

Recommended task split (final version produced by `superpowers:writing-plans`):

1. **Sampler `sample_async_greedy` + `is_pipelinable` predicate + 2 unit tests** — adds new code, no existing path touched.
2. **GenerationStream pipelined fields + branching `new()` + branching `next_token()` + 2 unit tests** — splits dispatch by mode, both paths separately tested.
3. **P4 fixture regression check** — run `tests/p4_qwen35_logits_match.rs`; root-cause any divergence.
4. **iron-bench rerun + acceptance gate** — start servers, rerun benchmark, confirm targets met, capture numbers in commit message / follow-up doc.

Each task is one commit, with the project gate (fmt + clippy + build) running clean before commit per CLAUDE.md.

Estimated effort: ~1 day (6-8 hours focused work) for an engineer familiar with the codebase.
