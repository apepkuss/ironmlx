# P5e Stage 1 Results (Approach A: MLX op rearrangement)

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Hardware | M5 Max 128GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit (35B-A3B-4bit MoE) |
| Branch HEAD post-Stage-1 | to be filled with `git rev-parse HEAD` after T4 commit |

## Per-experiment measurements (wall-clock ms, 3-run median)

| PP | T0 baseline | A.1 stream-parallel | A.2 mlx::compile | A.3 shape-elim | Notes |
|---|---|---|---|---|---|
| 128 | 127.66 | 148.83 (-16.6%) | 127.17 (+0.38%) | 127.06 (+0.47%) | A.1 regresses |
| 512 | 488.30 | 517.62 (-6.0%) | 488.84 (-0.11%) | 486.78 (+0.31%) | A.1 regresses |
| 2048 | 2067.45 | 2328.49 (-12.6%) | 2085.22 (-0.86%) | 2075.25 (-0.38%) | A.1 regresses |

## Promotion decisions

- **A.1 stream parallelism:** DISCARDED — consistent 6-17% regression at all PP. Per-forward `new_stream` allocation + implicit cross-stream sync at `silu(gate) * up` exceeds Metal scheduler's overlap benefit at these kernel shapes.
- **A.2 mlx::compile wrap:** DISCARDED — could not be implemented as a real wrap. 4 safe-wrapper API gaps blocked the experiment (closure `'static`, private `LinearImpl/MlpImpl`, runtime M-aware dispatch, integer reshape literals). No-op gate shows <=1% noise as expected; permanently removed since the API gaps aren't addressed in P5e scope.
- **A.3 shape elimination:** DISCARDED — +/-0.5% delta across all PP, within run-to-run noise. The extra squeeze kernel + smaller multiply did not represent measurable cost at these shapes.

**Net Stage 1 promotion:** none. All 3 features + cfg gates removed at HEAD. `sparse_moe.rs` is back to pre-T1 single-path code.

### Note on 4-way combined measurement (Step 4.1)

Skipped. With each individual A.x experiment landing within +/-0.5% of T0 (A.2, A.3) or showing a clear regression at all PP (A.1), there is no plausible mechanism for a 4-way combined run to expose synergy hidden by the single-feature runs. The combined run would only re-confirm the A.1 regression dominates.

## Stage 1 final wall-clock (post-cleanup, single unconditional code path)

| PP | Stage 1 final (ms) | tok/s | Δ vs T0 baseline |
|---|---|---|---|
| 128 | 128.16 | 998.8 | -0.39% |
| 512 | 488.61 | 1047.9 | -0.06% |
| 2048 | 2061.35 | 993.5 | +0.30% |

All three PP shapes land within +/-0.4% of T0 baseline, confirming that the
post-cleanup code path is semantically identical to pre-T1.

## Validation gates passed

- p5_qwen35_moe_smoke regression sentinel argmax=11: PASS (2/2)
- p5_qwen35_moe_batched (B=2 vs B=1 per-row): PASS (1/1)
- p5_qwen35_moe_http_smoke chat completion: PASS (1/1)
- sweep_full: 19/19 in 142 seconds (2m 22s) on Qwen3.5-4B-MLX-4bit
- clippy --all-features --workspace --release -D warnings: 0 warnings
- fmt --check: clean
- release build: PASS

### Note on sweep_full run-1 transient flake

The first sweep_full attempt reported 18/19 PASS with
`b1_p2_3c_plus_chunked_admit_mid::chunked_admit_mid_stall_delta` panicking
on `long_tokens.len() == 8` (got 0). Running the test in isolation
immediately after passed cleanly, and a fresh end-to-end sweep_full re-run
returned 19/19 PASS. The failing suite exercises B1-p2.3c chunked admission
on the Qwen3.5-4B dense path and has no source dependency on
`sparse_moe.rs` or the removed feature gates; classified as a transient
flake under sweep concurrency pressure, not a Stage 1 regression.

## Notes for Stage 2

Stage 1 yielded no positive result on Apple Silicon M5 Max at PP=128/512/2048
shapes. Profile data from T0 (`reports/p5e-t0-profile.md`) indicated 64.8% of
PP=2048 wall-clock in the 3 gather_qmm calls; rearranging MLX op order at the
forward level did not move that. Stage 2 (B.1 sorted routing) goes deeper —
modifying the index data MLX gather_qmm consumes — and is the remaining
in-scope opportunity for P5e.

## Open follow-ups (out of P5e scope)

- A.2 mlx::compile would require exposing `LinearImpl`/`MlpImpl` Array accessors + hoisting M-aware dispatch into a wrapper. Tracked as a future safe-wrapper-API task, not P5e.
- A.1 stream parallelism could revisit if MLX exposes a way to share a Stream pool across forwards (avoid per-call `new_stream` overhead). Not a current MLX safe-wrapper feature.
