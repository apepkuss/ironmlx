# P5f Reference Baseline

| Field | Value |
|---|---|
| Date | 2026-05-19 (HEAD a4249af) + 2026-05-20 sanity (HEAD pre-T1) |
| Hardware | M5 Max 128 GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Branch | ironmlx-p5e-perf |
| Baseline document | reports/p5e-three-way-bench.md (commit fc8e6c6 then trimmed to a4249af) |
| Sanity document | this section below (2026-05-19 b_max=1 sweep, no JSON committed) |

## Why this is the P5f baseline (no new bench needed)

Between HEAD `a4249af` (when the canonical 4-way bench was captured)
and the current P5f starting HEAD, all commits are doc-only:

- `6f4acb8 docs(p5f): spec update — T1 to Option 1`
- `0414de9 docs(p5f): MoE text-only known-path perf design spec`
- `41774b9 docs(p5f): MoE text-only known-path perf implementation plan`

No inference code path changed → ironmlx numbers in
`reports/p5e-three-way-bench.md` § 5 still represent the current
default-b_max=4 behavior. Reuse them as P5f T0.

## Canonical baseline numbers (default `b_max=4`)

From `reports/p5e-three-way-bench.md` § 5 (median across 5 timed runs):

| PP | ironmlx prefill (tok/s) | ironmlx TTFT (ms) | ironmlx decode TG (tok/s) | omlx prefill (tok/s) | omlx+10% target |
|---:|---:|---:|---:|---:|---:|
| 128 | 390 | 329 | 79.6 | 1088 | 1197 |
| 512 | 491 | 1042 | 78.5 | 2623 | 2886 |
| 2048 | 1842 | 1112 | 123.7 | 4227 | 4649 |
| 4096 | 1773 | 2310 | 123.7 | 4419 | 4861 |
| 8192 | 1725 | 4748 | 118.0 | 4261 | 4687 |
| 16384 | 1548 | 10581 | 112.0 | 3669 | 4036 |

## T1 dry-run reference (`--b-max 1` sanity, 2026-05-19)

Captured during P5f brainstorming with `ironmlx serve --b-max 1`,
same iron-bench sweep parameters. Raw JSON not committed (gitignored
per `reports/.gitignore`); summary preserved here.

| PP | ironmlx_bmax1 prefill (tok/s) | TTFT (ms) | decode TG (tok/s) | bmax1/b_max=4 prefill | omlx+10% target |
|---:|---:|---:|---:|---:|---:|
| 128 | 951 | 135 | 125.6 | 2.44× | 1197 (79% of target) |
| 512 | 1577 | 325 | 123.7 | 3.21× | 2886 (55%) |
| 2048 | 1843 | 1111 | 124.9 | 1.00× | 4649 (40%) |
| 4096 | 1833 | 2235 | 122.3 | 1.03× | 4861 (38%) |
| 8192 | 1724 | 4751 | 117.5 | 1.00× | 4687 (37%) |
| 16384 | 1606 | 10200 | 117.0 | 1.04× | 4036 (40%) |

## Reading

- T1 (CLI default `b_max=1`) is expected to deliver the b_max=1 sanity
  numbers in the second table (PP=128/512 prefill 2.44-3.21×). PP=2048+
  shows 1.00-1.04× (within noise) — those routes go through
  GenerationStream not Scheduler, so b_max change does not affect them.
- T2 (GenerationStream single-shot when KV budget allows) targets
  PP=4096-16384 prefill specifically. Expected post-T2: 3000-4500 tok/s
  across the long-prompt range.
- P5f expected close-out: PP=128 ~950 / PP=512 ~1577 / PP=2048 ~1842 /
  PP=4096+ 3000-4500 tok/s. PP=2048 will remain the largest gap to
  omlx+10% target — by design, that gap is the P5g GatedDeltaNet /
  GatedAttention focus.
