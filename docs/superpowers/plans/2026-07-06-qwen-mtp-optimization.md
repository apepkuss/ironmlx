# Qwen MTP Production Optimization Plan

> **For Boss:** Execute autonomously from `dev` in isolated worktree `/Users/xin/workspace/ironmlx-backend-qwen-mtp-optimization`, branch `feat/qwen-mtp-optimization`.

**Goal:** Improve Qwen3.5/Qwen3.6 MTP production performance and batching behavior without regressing default TTFT/ITL, then verify with unit tests, targeted regressions, full Rust checks, and A/B benchmark evidence.

**Final status:** Completed on branch `feat/qwen-mtp-optimization`.

**Final implementation notes:**
- Added cost-aware Qwen MTP draft-depth gating so low-value windows can reduce effective draft depth while high-value windows recover toward the configured limit.
- Added lazy logits/sampling attribution improvements and profile accounting without changing greedy semantics.
- Added safe main-cache rollback support for hybrid/non-Full Qwen caches by restoring snapshots and replaying the accepted prefix; Full-only caches continue using direct trim.
- Reworked Qwen batched MTP decode so active rows are drafted/verified in batched windows, with mid-admit reserved rows skipped until their MTP row state is finalized.

**Final verification:**
- `cargo fmt`
- `cargo +nightly fmt --all -- --check`
- `cargo +nightly clippy --all-features --workspace -- -D warnings`
- `cargo build --release`
- `cargo test -p ironmlx --lib mtp`
- Clean A/B benchmark matrix under `docs/benchmarks/qwen-mtp-optimization/2026-07-06-195554-clean/`

**Current root-cause findings:**
- Qwen MTP telemetry was unstable on `dev` until `fix(qwen): stabilize mtp rolling admission`; that fix is included as a prerequisite commit on this branch.
- Existing Qwen MTP adaptive budget is acceptance-only. It can keep drafting when the accepted-token value does not offset draft/verify/projection/sampling/commit cost.
- Existing Qwen batched MTP path is not truly batched. `Scheduler::step_mtp_batch` routes each active row through a temporary single-row scheduler, so `b_max > 1` adds admission concurrency but not batched MTP compute efficiency.
- Existing mismatch commit path restores main/MTP caches and replays accepted prefix. MTPLX shows a capture/trim style commit can avoid this replay when cache implementations support precise truncation.
- Existing `sampling_us` includes MLX lazy graph materialization triggered by `to_vec`. Any optimization must first separate attribution from real wall-time improvement.

## Phase 0: Baseline and Harness

**Files:**
- `ironmlx/src/core/speculative.rs`
- `ironmlx/src/core/scheduler.rs`
- `ironmlx/src/core/server/scheduler_actor.rs`
- existing benchmark scripts under `docs/benchmarks/qwen-mtp-regression/` if reusable

**Tasks:**
1. Run targeted tests after the prerequisite cherry-pick:
   - `cargo test -p ironmlx actor_mtp_mode_prefill_and_step_use_mtp_for_eligible_request --lib`
   - `cargo test -p ironmlx mtp_counters_publish_cumulative_stat_deltas --lib`
2. Inspect cache APIs and Qwen MTP forward APIs before changing behavior:
   - `LayerCache`, `MtpCache`, `Mtp`, `MtpStepOutput`
   - Qwen model hidden/logits projection methods
3. Add focused unit tests before implementation for each policy/cache behavior.

## Phase 1: Cost-Aware Gate/Depth Policy

**Design target:**
- Replace acceptance-only budget adjustment with a cost-aware policy that considers:
  - per-window attempted and accepted draft tokens
  - by-position acceptance
  - draft forward cost
  - verify/projection/sampling cost
  - MTP cache commit/restore or rollback cost
- Preserve conservative defaults:
  - Qwen3.6 with high acceptance should keep effective MTP enabled.
  - Qwen3.5 with low acceptance and high per-window overhead should quickly reduce draft depth or gate MTP for a cooldown.
  - Long prompt fallback/gated behavior must not regress TTFT.

**Initial API sketch:**
```rust
#[derive(Debug, Clone)]
struct MtpDraftPolicyState {
    max_draft_tokens: usize,
    current_budget: usize,
    cooldown_windows: usize,
    ewma_acceptance: f64,
    ewma_cost_per_accepted: f64,
}

impl MtpDraftPolicyState {
    fn budget_for_next_window(&mut self, stats_delta: &MtpSpeculativeStats) -> usize;
}
```

**Tests first:**
- Low acceptance plus high overhead reduces budget to 1 or enters cooldown.
- High acceptance plus low overhead restores toward configured maximum.
- Position-level acceptance can cap depth when later positions are consistently rejected.
- Policy is deterministic and saturating with zero attempts/costs.

## Phase 2: Lazy Logits/Sampling Optimization

**Investigation target:**
- Determine whether `project_hidden_on` can project only necessary rows and whether logits materialization can be reduced without changing greedy semantics.
- Separate profile accounting:
  - projection/materialization time
  - CPU-side token extraction time
- Keep exact greedy output parity. No approximate sampling behavior.

**Candidate fixes:**
- Avoid projecting unneeded verifier rows if any path currently computes beyond `accepted_prefix + correction`.
- Force/evaluate verifier logits before sampling only if this improves attribution or enables overlapped scheduling.
- If MLX APIs cannot reduce wall time safely, leave behavior unchanged and document with a regression test for profile accounting.

**Tests first:**
- Greedy sampled tokens match pre-change behavior for full-accept and mismatch windows.
- Profile fields remain monotonic and do not double count deltas.

## Phase 3: Capture/Trim Commit

**Design target:**
- On mismatch, keep the verified main-model KV state for the accepted prefix and trim only unaccepted suffix instead of restoring and replaying the accepted prefix.
- Apply only when every involved cache type can guarantee exact truncation. No compatibility fallback unless the current cache abstraction already exposes safe capability checks.

**Candidate API sketch:**
```rust
trait TrimToLen {
    fn trim_to_len(&mut self, len: usize) -> Result<()>;
}
```

**Tests first:**
- Mismatch after `k` accepted tokens leaves main/MTP caches at exactly prefix + `k`.
- Full rejection leaves cache at original prefix plus correction token only.
- Full acceptance still reuses temporary MTP cache and does not regress.

## Phase 4: True Qwen Batched MTP Decode

**Design target:**
- Replace per-row temporary scheduler loop in `Scheduler::step_mtp_batch` with actual batched draft/verify for active Qwen rows.
- Preserve streaming event order and per-row finish semantics.
- Support `b_max > 1` with:
  - ragged prompt/history lengths
  - per-row MTP budget decisions
  - row-level fallback when a request is not MTP-eligible

**Implementation outline:**
1. Group active MTP-eligible rows by effective draft budget.
2. Draft next tokens in batched MTP forward calls.
3. Verify proposed windows with batched main-model hidden forward.
4. Resolve accept/reject per row.
5. Commit or trim caches per row.
6. Emit events in existing scheduler order.

**Tests first:**
- `b_max=2` two active Qwen fake rows both draft in one batch step.
- One row mismatch and one row full-accept in the same batch preserve independent token streams.
- `b_max=4` mixed eligible/ineligible rows do not starve baseline decoding.

## Final Verification

Required Rust checks before completion:
- `cargo fmt`
- `cargo +nightly fmt --all -- --check`
- `cargo +nightly clippy --all-features --workspace -- -D warnings`
- `cargo build --release`

Targeted tests:
- Qwen MTP speculative policy tests
- Qwen batched scheduler tests
- Scheduler actor MTP profile tests

Benchmark evidence:
- Clean A/B repeated runs in the same environment:
  - baseline `b_max=1`
  - MTP fixed/default `b_max=1`
  - MTP `b_max=4`
  - optimized adaptive/default
- Cover short and long prompt for Qwen3.5/Qwen3.6, with special attention to TTFT, ITL, e2e, tokens/sec, acceptance, and profile breakdown.
