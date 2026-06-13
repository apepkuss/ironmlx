# TurboQuant Pre-Rotated QK Cost Probe

## Goal

Determine whether the fused TurboQuant pre-rotated decode path has a real `qk`
kernel regression, or whether the current `qk` profile stage is paying for the
lazy materialization of the fused MRoPE + WHT query transform. Keep production
changes only if they improve end-to-end decode throughput.

## Current Evidence

- The fused path removed `q_rotate` profile events and improved 32k/K3V4 decode
  throughput versus the previous simdgroup q-rotate baseline.
- The fused path's `qk` profile stage is higher than the old `qk` stage, but
  lower than the old `q_rotate + qk` combined cost.
- The fused query tensor is produced lazily before attention, so the first
  profiled attention stage may be absorbing upstream transform work.

## Tasks

1. Add a narrow profile-only materialization point around the fused MRoPE + WHT
   decode query transform.
   - Emit a separate JSON event so existing attention-stage summaries do not
     misattribute it as `qk`.
   - Keep the behavior gated behind `IRONMLX_TURBOQUANT_ATTN_PROFILE`.
   - Add a small unit test for the JSON formatter before wiring it into the hot
     path.

2. Run the focused correctness tests for the MRoPE fused transform and
   pre-rotated KV attention path.
   - Confirm the profile-only change does not affect normal production output.

3. Re-run a 32k/K3V4 profile pass.
   - Compare the new fused-transform event with the `qk` event.
   - If `qk` falls back near the previous baseline, treat the earlier high `qk`
     as lazy materialization, not a qk kernel regression.

4. Decide whether to keep any production optimization.
   - If the data shows no real qk regression, keep only useful diagnostics.
   - If a real regression remains, inspect the qk dispatch and implement the
     smallest reliable production fix, then re-benchmark.

5. Run required Rust checks.
   - `cargo fmt`
   - `cargo +nightly fmt --all -- --check`
   - `cargo +nightly clippy --all-features --workspace -- -D warnings`
   - `cargo build --release`

## Acceptance Criteria

- The cost attribution is unambiguous in profile output.
- K3V4 long-context profile results are recorded under
  `docs/benchmarks/turboquant-kv/`.
- Any retained production code is justified by measured end-to-end improvement.
- All required Rust checks pass.
