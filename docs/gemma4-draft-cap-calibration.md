# Gemma4 Draft-Cap Observation and Offline Calibration

## Scope

This workflow recommends the production `max_draft_tokens` ceiling for the
Gemma4 assistant-drafter policy from measured `ironmlx-core-bench`
observations. The existing online adaptation remains enabled below each tested
ceiling. Calibration is observation-only: it does not change runtime defaults.

The current calibration input is deliberately limited to Gemma4 `mtp-text`
(`B=1`). `ironmlx-core-bench` does not yet expose concurrent Gemma4 drafter
scheduler traffic, so batched recommendations are out of scope even though the
scheduler records the same bounded observations internally.

Paged prefix cache and Active KV offload must be disabled. Each calibration
report also requires exact matches for mode, model and drafter paths, device,
IronMLX version, prompt, generation length, prefill settings, KV quantization,
capacity, warmup count, and measured-run count. Every input must contain exactly
the declared number of measured records, and every valid greedy record must
produce the same token IDs and finish reason across all tested ceilings.

## Observation Schema

Every measured record contains bounded `draft_cap_observations` grouped by:

- configured cap and actual min/max draft depth;
- batch width;
- context bucket: `up_to2k`, `up_to8k`, `up_to32k`, `up_to128k`, or
  `above128k`;
- whether rows crossed context buckets.

Each group reports windows, drafted, accepted, and actually committed tokens,
rollbacks, total window time, and stage timing. Candidate output also reports full-cap, adaptively
lowered, and mixed-depth window coverage plus mean actual draft depth. The table
is capped at 256 regimes. Calibration fails if any windows were dropped, because
an incomplete table cannot support a safe recommendation.

Adaptive-lowered and mixed-depth windows are included under their configured
policy ceiling; excluding them would bias the higher-cap policy toward only its
favorable windows. Mixed-context, malformed, invalid, or zero-time observations
are not used to select a cap, and their coverage is reported.

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
prompt/context workload at a time:

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
