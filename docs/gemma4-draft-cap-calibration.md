# Gemma4 Draft-Cap Observation and Offline Calibration

## Scope

This workflow recommends the production `max_draft_tokens` ceiling for the
Gemma4 assistant-drafter policy from measured `ironmlx-core-bench`
observations. The existing online adaptation remains enabled below each tested
ceiling. Calibration is observation-only: it does not change runtime defaults.

Calibration reports use schema version 2. Scheduler prompt metadata,
`scheduler_batch_width`, per-record `scheduler_requests`, and draft-cap
observation fields are required in benchmark artifacts. Legacy artifacts that
omit these fields are rejected rather than inferred.

Calibration accepts Gemma4 `mtp-text` (`B=1`) and Gemma4 drafter
`scheduler-text`. Scheduler batch width is the number of repeated
`--prompt-file` arguments, not `b_max`; `b_max` remains the scheduler capacity
and must be at least the number of admitted benchmark requests. The benchmark
uses the same batched Gemma4 drafter prefill and decode APIs as the production
scheduler core.

Paged prefix cache and Active KV offload must be disabled. Each calibration
report also requires exact matches for mode, model and drafter paths, device,
IronMLX version, prompt, generation length, prefill settings, KV quantization,
capacity, warmup count, measured-run count, ordered scheduler prompt paths, each
prompt's token count, and admitted batch width. Every input must contain exactly
the declared number of measured records. Greedy token IDs and finish reasons
must match across all tested ceilings for each request position.

## Observation Schema

Every measured record contains bounded `draft_cap_observations` grouped by:

- configured cap and actual min/max draft depth;
- batch width;
- context bucket: `up_to2k`, `up_to8k`, `up_to32k`, `up_to128k`, or
  `above128k`;
- whether rows crossed context buckets.

Each group reports windows, drafted, accepted, and actually committed tokens,
rollbacks, total window time, and stage timing. Candidate output also reports
full-cap, adaptively lowered, and mixed-depth window coverage plus mean actual
draft depth. The table is capped at 256 regimes. Calibration fails if any
windows were dropped, because an incomplete table cannot support a safe
recommendation.

Adaptive-lowered and mixed-depth windows are included under their configured
policy ceiling; excluding them would bias the higher-cap policy toward only its
favorable windows. Mixed-context, malformed, invalid, or zero-time observations
are not used to select a cap, and their coverage is reported.

Each scheduler record contains ordered `scheduler_requests` with per-request
prompt path, TTFT, E2E, generated token IDs, finish reason, and validity. The
top-level generated output remains request 0 for existing benchmark consumers.
`aggregate_generation_tps` reports batch decode throughput; cap selection still
uses committed tokens per measured speculative-window second.

## Collection

Use the same already-rendered raw prompt and all the same flags for every cap.
Run at least three measured records per configured ceiling and collect at least
32 context-homogeneous windows per candidate. For thermal balance, execute
ceilings in ABBA order and pass all resulting JSON files to the calibrator.

```bash
source /Users/xin/.local/mlx/mlx-env.sh
cargo build --release --bin ironmlx-core-bench --bin ironmlx

target/release/ironmlx-core-bench \
  --model "$GEMMA4_MODEL" \
  --mtp-model-dir "$GEMMA4_DRAFTER" \
  --prompt-file "$PROMPT_FILE" \
  --mode mtp-text \
  --mtp-draft-tokens 1 \
  --max-tokens 256 \
  --warmup-runs 1 \
  --runs 3 \
  --out cap1-a.json

target/release/ironmlx-core-bench \
  --model "$GEMMA4_MODEL" \
  --mtp-model-dir "$GEMMA4_DRAFTER" \
  --prompt-file "$PROMPT_FILE" \
  --mode mtp-text \
  --mtp-draft-tokens 2 \
  --max-tokens 256 \
  --warmup-runs 1 \
  --runs 3 \
  --out cap2-b.json
```

Repeat cap 2 and then cap 1 with identical flags to complete ABBA. Calibrate one
ordered prompt workload at a time:

```bash
target/release/ironmlx mtp-draft-cap \
  --input cap1-a.json \
  --input cap2-b.json \
  --input cap2-b2.json \
  --input cap1-a2.json \
  --min-windows 32 \
  --min-records 3 \
  --min-improvement-percent 3 \
  --output gemma4-draft-cap.json
```

For a homogeneous scheduler batch, repeat the same prompt path. For a
heterogeneous batch, pass prompt paths in a stable order and preserve that order
for every cap and ABBA repetition:

```bash
target/release/ironmlx-core-bench \
  --model "$GEMMA4_MODEL" \
  --mtp-model-dir "$GEMMA4_DRAFTER" \
  --prompt-file "$SHORT_PROMPT" \
  --prompt-file "$LONG_PROMPT" \
  --mode scheduler-text \
  --b-max 2 \
  --mtp-draft-tokens 2 \
  --max-tokens 256 \
  --warmup-runs 1 \
  --runs 3 \
  --out cap2-b2.json
```

If heterogeneous requests finish at different times, later observations may
have a smaller actual `batch_width`; those regimes remain separate. Windows
whose rows cross context buckets are reported as mixed coverage and excluded
from recommendation selection.

## Recommendation Rule

The score compares the complete production policy under each configured ceiling
using committed decode tokens per measured speculative-window second:

`committed_tokens / total_window_time`.

`committed_tokens` is captured after stop-token and generation-length
truncation, so the score does not overcount the final speculative window.

At least two caps must satisfy both coverage gates. `best_observed_cap` records
the raw throughput winner. `recommended_cap` is conservative: it selects the
lowest eligible cap within the configured improvement threshold of the winner.
With the default 3% threshold, a higher cap must provide a material gain before
it displaces a lower cap.

Possible statuses are `recommended`, `insufficient_cap_coverage`,
`insufficient_windows`, and `insufficient_records`. A recommendation is evidence
for a specific runtime context only; promoting it to a default requires separate
repeated end-to-end performance and correctness validation.

The in-process scheduler benchmark isolates model-core cap policy. Before
promotion, repeat the winning policy through the production HTTP path and gate
aggregate throughput, ITL p95/p99, TTFT p95, E2E p95, and output identity.
