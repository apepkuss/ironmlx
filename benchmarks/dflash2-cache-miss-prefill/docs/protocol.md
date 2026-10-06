# DFlash2 cache-miss prefill — measurement protocol

Written 2026-10-06, after diagnosis and before any formal timing. Sample
sizes and criteria below are fixed; there are no re-runs to reach a pass.

## Builds

| arm | binary | source |
| --- | --- | --- |
| B | `binaries/ironmlx-baseline-b9c7b4fb` | dev 6c1774ce, clean |
| C | `binaries/ironmlx-candidate-f007c386` | 6c1774ce + `binaries/ironmlx-candidate-f007c386.source.diff` |

- **Build.** Both binaries: `MLX_DIR=/Users/xin/.local/mlx cargo build --release -p ironmlx`. Identities are in `binaries/*.identity.json` and `*.build.log`.
- **Candidate change.**
  - A cold miss under the prefix cache uses single-graph prefill when single prefill is already qualified (M5 profile setting and M5 affine4 route). A restored prefix keeps the scheduler prefill.
  - The fingerprint prefill tag becomes `scheduler-b1-chunk-v2-cold-single`.
  - Adds a default-off prefill phase timing diagnostic.

## Fixed environment

- **Model.** Snapshots:
  - target `mlx-community--Qwen3.8-27B-4bit/3e6447f0…`;
  - draft `z-lab--Qwen3.8-27B-DFlash2/50307d4c…`.
- **MLX.** `/Users/xin/.local/mlx` (libmlx.a 27f1e5d1…, mlx.metallib adc6967e…), passed with `--mlx-metallib`.
- **Server.** The arguments the App builds for this model with default settings (`scripts/common.py: command`):
  - block 8, draft 4-bit;
  - `--max-cache-cap 32768`, `--prefix-lru-cache-max-bytes 8589934592`;
  - server-default sequences (1) and prefill chunk size (2048).
- **Environment.** App-like: only PATH, HOME, TMPDIR, LANG, LC_ALL, USER and LOGNAME, plus `IRONMLX_LOG_LEVEL=warn`. No DYLD, MLX or experiment variable, and no diagnostic variable in timed sessions.
- **Port and isolation.** Port 18490, one server at a time, fresh server per session. Only the runner's own process group is cleaned up.
- **Fixtures.**
  - `fixtures/short-prompts.json` (sha 138468e1…) and `fixtures/30k-prompts.json` (sha a7ecb63f…), copied from the earlier suites.
  - Requests are streamed chat completions: greedy (temperature 0, top_p 1), thinking off.

## Diagnosis (done before this protocol; `results/diagnose-phases-v1`)

The phase timing on B's code plus the timing diagnostic gave this split.

**Short prompts, cache on (SchedulerB1):**
- `[N-1]` forward 100–170 ms;
- an extra `[1]` forward 35–40 ms;
- two cache saves at 0.1–0.3 ms each;
- first logits 1.5–7.7 ms.

**Short prompts, cache off (single prefill):** one `[N]` forward of 103–140 ms.

**30K:** chunking is identical in both modes (15 × 2048 + 12). Saves cost 0.1–0.4 ms, or 29 ms when a save evicts 17 entries.

The cost of a short cold miss is therefore the B1 `[N-1] + [1]` split. Cache saving is not the cost.

## A. Correctness and cache behaviour (`scripts/scenarios.py`)

Every session enables the token-id and phase diagnostics, so none of them is timed.

- **`--set main`.**
  - Sessions B-on, C-on, B-off and C-off.
  - Chains for short code-1, knowledge-1 and code-3, and for 30K code-1: miss, exact repeat, append (`[P, A, Q2]`), fork (`[P, A, Q3]`), and for 30K a document fork with a different final question.
- **`--set fallback`.**
  - M5 profile off (B, C).
  - Candidate with `IRONMLX_EXPERIMENTAL_DFLASH2_SINGLE_PREFILL=0`.
  - Candidate with a 1 GiB prefix cache.
  - Memory limit 40 GB (B, C).
  - Candidate cancellation: a 30K cold miss abandoned after 3 s, then a short miss and its repeat.

**Required:**
1. **Valid responses.** Every response is valid. Any error in a memory-limit session must be the same in B and C.
2. **Candidate equals no-cache reference.**
   - C-on cold-miss token ids and finish reasons equal B-off's and C-off's for every chain prompt. With the change, a cold miss uses the same prefill as no cache.
   - C-on exact repeats reproduce C-on misses.
3. **Fallbacks unchanged.** With the profile off or single prefill forced off, C equals B's token ids, because both run the B1 path.
4. **Hit capability kept.** For every step, C-on hit tokens ≥ B-on hit tokens.
   - The exact repeat hits the whole prompt in both.
   - Saves and cache entries are reported.
5. **Cancellation.** It leaves the server serving, with valid later requests.

## B. Timing (`scripts/run_timing.py`, `scripts/analyze_timing.py`)

**Rounds.**
- One unscored adaptation pair (B then C), then 6 scored rounds.
- Arm orders: BC, CB, CB, BC, BC, CB.
- Fresh server per session, 10 s between sessions, 1 s between requests.

**Session steps (fixed):**
1. `first`: short code-2 as the process's first request (max_tokens 4096).
2. Two unrelated warmups (max_tokens 64, not scored).
3. Warm cold misses: short code-1, code-3, knowledge-1, knowledge-2 and knowledge-3 (max_tokens 4096). A miss with hit tokens > 0, or without a recorded miss, is invalid.
4. Exact repeats of code-1 and knowledge-1 (hit path).
5. Append for code-1, `[P, A, Q2]`, with the arm's own answer A.
6. 30K code-1 cold miss (max_tokens 1024), then its exact repeat.

**Per request:**
- client TTFT, Decode (completion tokens − 1 over the streamed span) and E2E;
- server prefill time and prefix-cache counter deltas;
- footprint peak (50 ms sampler).

The session lifecycle peak is the maximum of all samples.

**Analysis.**
- Per task and arm, take the median over the 6 rounds.
- Ratio C/B per task; overall value is the geometric mean over tasks.
- 95% interval by paired-round bootstrap: resample rounds, 10000 draws, seed 20261006.

**Criteria:**

| | metric | pass when |
| --- | --- | --- |
| primary | warm cold-miss TTFT (5 short tasks) | point ≤ 0.95 and upper bound < 1.0 |
| non-regression | warm cold-miss Decode | lower bound ≥ 0.97 |
| non-regression | warm cold-miss E2E | upper bound ≤ 1.03 |
| non-regression | hit TTFT (2 exact repeats) | point ≤ 1.10, and hit tokens equal between arms |
| non-regression | 30K cold-miss TTFT | upper bound ≤ 1.03 |
| non-regression | 30K repeat TTFT | point ≤ 1.10 |
| non-regression | memory: median session lifecycle peak and median 30K-miss request peak | C/B ≤ 1.02 each |

- **Hit TTFT margin.** Hit TTFT is about 10 ms, so 1 ms is 10%; that is why the hit criteria use 1.10.
- **Descriptive only.** Process-first TTFT and append hit tokens/TTFT are reported with intervals, without a criterion.
- **Failures.** A failed or unresolved criterion is reported as such. Failed runs are kept.

The historical short-prompt TTFT failure of the M5 productization is not
re-judged by this work.
