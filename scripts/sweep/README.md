# B1-p2 sweep scripts

Three-tier regression gate for the B1-p2 batched-serving stack.

| Script | Time | Run | Coverage |
| --- | --- | --- | --- |
| `sweep_smoke.sh` | ~5-15 min | every code change | lib tests + 1-2 most-relevant integration suites (`git diff`-driven) |
| `sweep_scoped.sh` | ~30-60 min | every meaningful commit | adjacent suites for the changed area (scheduler / vl / http / decode) |
| `sweep_full.sh` | ~3-4 h | pre-merge / close-out | all 16 suites end-to-end |

All three:
- Require `QWEN35_MODEL` env var (auto-detected from `~/.ironmlx/models/`).
- Require `MLX_DIR` (default `$HOME/.local/mlx`).
- Append results to `/tmp/sweep_*_<timestamp>.log`.
- Use `--release --test-threads=1` to serialise GPU.

## Smoke

```bash
# Auto-pick by git diff vs HEAD~1.
./scripts/sweep/sweep_smoke.sh

# Compare vs a specific base.
./scripts/sweep/sweep_smoke.sh --base main

# Explicit suite list (skips auto-pick).
./scripts/sweep/sweep_smoke.sh --suites b1_p2_4_batched_vl b1_p2_3b_2_scheduler_actor

# Run a specific test inside a suite.
./scripts/sweep/sweep_smoke.sh --suites b1_p2_4_batched_vl::mid_admit_vl_during_text_decode
```

Default fallback (no diff match) is `b1_p2_3b_2_scheduler_actor` — a fast scheduler-actor smoke (~1 min).

## Scoped

```bash
# Auto-pick area(s) by git diff vs HEAD~1.
./scripts/sweep/sweep_scoped.sh

# Force an area.
./scripts/sweep/sweep_scoped.sh --area scheduler    # 10 scheduler suites
./scripts/sweep/sweep_scoped.sh --area vl           # 2 vl suites
./scripts/sweep/sweep_scoped.sh --area http         # 3 http suites
./scripts/sweep/sweep_scoped.sh --area decode       # 3 decode suites
./scripts/sweep/sweep_scoped.sh --area all          # equivalent to sweep_full
```

## Full

```bash
./scripts/sweep/sweep_full.sh
```

Runs all 15 (or 16 if `b1_p2_3c_plus_chunked_admit_mid.rs` exists) suites; logs every suite even on failure for full coverage in the close-out report.

## Diff → area mapping

| Source path pattern (regex) | Smoke suite(s) | Scoped area |
| --- | --- | --- |
| `core/scheduler.rs`, `core/server/scheduler_actor` | `b1_p2_3b_2_scheduler_actor`, `b1_p2_3c_3_continuous_batching` | scheduler |
| `admit_mid`, `AdmitMidHandle` | `b1_p2_4_batched_vl::mid_admit_vl_during_text_decode` | scheduler |
| `core/server/(openai\|anthropic\|chat_format\|mod).rs`, `cli/serve` | `p4_http_smoke` | http |
| `core/generate.rs`, `GenerationStream` | `b1_p2_2_batched_decode` | decode |
| `models/(vision\|qwen3_5/(cross_modal\|image_processor)\|qwen3_5_moe)` | `p6_qwen35_vl_logits_match`, MoE-VL smoke script | vl |
| `core/cache/` | `b1_p2_1_batched_prefill` | decode |
| `models/qwen3_5/(model\|text_model\|config).rs`, `models/qwen3_5_moe/(model\|text_model\|config).rs` | `b1_p2_2_batched_decode`, `b1_p2_4_batched_vl::batched_vl_b2_full_vl_bit_id` | decode |

Add new patterns by editing the `PATTERNS` table inside each script.

## Known sweep-context flakes

These have failed inside `sweep_full.sh` but **pass standalone**. Treat as known flakes; document in close-out reports:

- `b1_p2_3d_admission_queue::iron_bench_c8_with_queue_no_4xx` (S5) — 8 concurrent HTTP workers; sweep-context thermal / kernel-cache state causes 0/8 successes. Standalone gets 4/8+ → passes.
- `b1_p2_3f_cache_cap::admit_long_prompt_pp10k` — PP=10K real-model decode; sometimes hangs in sweep when prior suites left GPU resources in flight. Standalone: ~5 min PASS.

If either fails in `sweep_full.sh`, re-run standalone before treating as a real regression.
