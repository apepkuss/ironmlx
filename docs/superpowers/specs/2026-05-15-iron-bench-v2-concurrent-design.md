# iron-bench v2 — Closed-loop concurrent benchmarking

**Date:** 2026-05-15
**Project:** `iron-bench` (workspace member of cxx-mlx, decoupled from ironmlx)
**Predecessor:** iron-bench v1 (single-request sequential, shipped P7 commit `7c...` per README line 88-89: "Single-request only. Multi-request concurrency comes in v2 once ironmlx P8b ships the batched scheduler.")
**Successor:** iron-bench v3 (open-loop Poisson + fairness metrics — deferred until ironmlx 3d admission queue ships)

---

## §1 Goals

1. Add **closed-loop concurrent benchmarking** to iron-bench. N concurrent workers, each fires HTTP request → awaits response → repeats. Run for `--duration <s>` (typically 30-60s). Aggregates per-request metrics across all workers + all cells.

2. **iron-bench v1 (sequential) mode preserved as default.** v2 mode opt-in via `--concurrent N`. CLI parser dispatches on flag presence: sequential when absent, concurrent when `--concurrent N > 1`.

3. **New metrics** specifically for multi-request workloads:
   - **TTFT distribution**: p50 / p95 / p99 across all worker-emitted requests
   - **ITL distribution** (Inter-Token Latency, per-request mean → p50/p95/p99 across requests)
   - **Aggregate throughput**: total emitted tokens / total wall-clock seconds (tokens/s); total completed requests / wall-clock (req/s)
   - **Per-worker throughput**: same but scoped to one worker (validates worker balance — should be ~uniform under closed-loop)
   - **Worker utilization**: % time a worker spent in-flight vs idle (idle = waiting for HTTP response; closed-loop should approach 100% under saturation)

4. **Multi-target comparison preserved** — `--target name=URL` repeatable flag from v1 works unchanged in v2. Each target gets independent worker pool of size N.

5. **iron-bench v2 unblocks ironmlx 3c-3 aggregate-throughput perf comparison** vs. omlx (CLI), mlx-lm-server (PyPI), vllm-mlx (production multi-request baseline), llama.cpp server (single-batch slot baseline).

## §2 Non-goals

- **Open-loop (Poisson arrival).** Closed-loop is sufficient for ironmlx 3c-3 throughput comparison (the most important near-term need). Open-loop measures SLA-sensitive metrics under arrival burstiness; defers to iron-bench v3 alongside ironmlx 3d (admission queue) where backpressure semantics matter.
- **Fairness metrics (Jain's index, per-tenant quotas).** Closed-loop with N identical workers doesn't reveal queue fairness; needs open-loop + multi-tenant client identity. Defers to v3.
- **Latency under target QPS.** "Hold target rate X req/s and measure p99" is an open-loop concept. v2 closed-loop instead measures "at N concurrent in-flight, what's the resulting QPS + p99".
- **Distributed multi-host load generation.** iron-bench v2 runs from one machine. Multi-host coordination (k8s job, locust master/worker) is v4+ if ever needed.
- **Request-content variation.** v2 keeps v1's synthetic-prompt construction (length-controlled with per-request nonce). Real-traffic replay is v4+.
- **Streaming protocol changes.** v2 keeps SSE-only consumption. WebSocket / gRPC streaming would require new client code; out of scope.

## §3 Background

### 3.1 Current iron-bench structure (v1)

- `main.rs` — CLI parser (clap), tokenizer load, `for prompt_len { for target { run_cell }} `nested loop, dispatch to report formatter
- `runner.rs` — `run_cell(client, target_name, target_url, model, pp, tg, warmup, runs, tokenizer)` runs `warmup` discarded runs then `runs` timed runs sequentially, builds `Vec<RunOutcome>`
- `client.rs` — `run_chat_completion(client, url, model, prompt, tg)` issues streaming POST + parses SSE; returns `RequestResult { timings, server_prompt_tokens, server_completion_tokens, chunk_count, finish_reason, ... }`
- `prompt.rs` — `synthesize_prompt(tokenizer, target_len, nonce) -> Result<(String, usize)>` constructs prompts of exact target token length using nonce-prefixed filler
- `report.rs` — `render_markdown / render_csv / render_json` reduce `Vec<CellResult>` to formatted output

Total: 1140 LOC across 5 files. Tokio runtime is `#[tokio::main]` (multi-thread by default). All async types are `Send + Sync` already (uses `reqwest::Client`).

### 3.2 Why closed-loop first

Closed-loop is the industry default for LLM serving benchmarks:
- **vLLM** `benchmark_serving.py` — closed-loop (N concurrent prompts shuffled from dataset)
- **SGLang** `bench_serving.py` — same pattern
- **wrk, hey, ghz** (general HTTP) — closed-loop default; `--rps` for open-loop is opt-in

Closed-loop properties:
- ✅ Server saturation guaranteed at N → measures upper bound on throughput
- ✅ Backpressure self-regulating (slow server → fewer requests in flight, never blows up)
- ✅ Simple correctness (one outcome per worker iteration, no queue management)
- ❌ Doesn't model real arrival burstiness (open-loop's strength)
- ❌ Doesn't expose server's queue depth or fairness under overload

For ironmlx 3c-3 (the immediate need), closed-loop is the right tool: we want to know "at N concurrent client load, what total tokens/s + p99 TTFT does the rolling decode loop deliver". Open-loop's "at λ req/s, what p99" is the wrong question for ironmlx — we don't yet have admission queue (3d), so high λ would just produce `Err("scheduler full")` not interesting metrics.

### 3.3 Aggregation semantics

Closed-loop's natural unit is **completed request**, not **timed run**. v2 changes the reporter to aggregate over all completed requests across all workers within a cell:

```
v1: Cell { runs: [Run { ttft, tg_tps, ... } × N_runs] } → median(ttft), p95(tg_tps), ...
v2 closed-loop: ConcurrentCell { requests: [Request { worker_id, ttft, ... } × M] } where M = sum over workers of completed requests
              → p50/p95/p99(ttft), aggregate_tokens_per_sec = sum(tokens) / wall_duration
```

Aggregation happens after the worker pool's join. Wall duration = `duration - max(start_skew)` to be precise; under typical 30s benches the skew is sub-millisecond.

### 3.4 Industry reference (informs design)

- **wrk**: closed-loop, multi-thread + multi-connection per thread. Reports avg/stdev/max latency, total req/s. Doesn't expose percentiles natively.
- **wrk2**: extends wrk to open-loop (rate-controlled with HdrHistogram). v3 candidate.
- **vLLM `benchmark_serving.py`**: per-request metric collection into Python lists, numpy percentiles. Matches what v2 needs.
- **fortio**: production-grade load tester (both closed and open loop). Heavy; v2 doesn't need its sophistication.

iron-bench v2 stays minimal: tokio workers + Vec collection + sort-and-index for percentiles. No HdrHistogram dep, no special data structures.

## §4 Architecture

### 4.1 New CLI surface

```
iron-bench --target name=URL [...]
           --model-dir PATH
           --prompt-len 128,512,2048
           --max-tokens 128
           [--concurrent N]              ← NEW. opt-in v2 mode.
           [--duration 30]               ← NEW. v2 wall-clock seconds (default 30).
           [--runs 5]                    ← v1 only (mutually exclusive with --concurrent).
           [--warmup 1]                  ← v1 only.
           [--warmup-duration 5]         ← NEW. v2 warmup wall-clock seconds (default 5).
           [--format markdown|csv|json]
           [--timeout 300]
```

**Mode dispatch logic in `main.rs`:**

- `--concurrent N` absent → v1 sequential mode (existing `run_cell` path, no behavior change)
- `--concurrent N` present (and N ≥ 1) → v2 concurrent mode (new `run_cell_concurrent` path)
- `--concurrent N` + `--runs M` together → conflict; clap-level mutually exclusive validation (or warn + ignore `--runs` with eprintln). Choose: clap mutex.

`--concurrent 1` is legal and effectively sequential, but uses the v2 runner path (with full per-request metrics + duration-based stopping). Useful for "one client, sustained load" baseline against multi-client tests.

### 4.2 New `runner_concurrent` module (or extend `runner`)

`ironmlx/iron-bench/src/runner.rs` gains a new function:

```rust
pub async fn run_cell_concurrent(
    client: Arc<reqwest::Client>,
    target_name: &str,
    target_url: &str,
    model: &str,
    pp: usize,
    tg: usize,
    warmup_duration: Duration,
    duration: Duration,
    concurrent: usize,
    tokenizer: Arc<Tokenizer>,
) -> Result<ConcurrentCellResult>;
```

Implementation outline:

```rust
// 1. Warmup: same N workers, warmup_duration. Discard outcomes.
let warmup_deadline = Instant::now() + warmup_duration;
let mut warmup_handles = Vec::with_capacity(concurrent);
for worker_id in 0..concurrent {
    let client_w = client.clone();
    let tokenizer_w = tokenizer.clone();
    let url = target_url.to_string();
    let model_w = model.to_string();
    warmup_handles.push(tokio::spawn(async move {
        let mut nonce = nonce_seed() ^ (worker_id as u64);
        while Instant::now() < warmup_deadline {
            let (prompt, _) = synthesize_prompt(&tokenizer_w, pp, nonce)?;
            let _ = run_chat_completion(&client_w, &url, &model_w, &prompt, tg).await?;
            nonce = nonce.wrapping_add(1);
        }
        Ok::<(), anyhow::Error>(())
    }));
}
for h in warmup_handles { h.await??; }

// 2. Timed phase: same N workers, duration. Collect outcomes.
let cell_start = Instant::now();
let timed_deadline = cell_start + duration;
let mut timed_handles = Vec::with_capacity(concurrent);
for worker_id in 0..concurrent {
    let client_w = client.clone();
    let tokenizer_w = tokenizer.clone();
    let url = target_url.to_string();
    let model_w = model.to_string();
    timed_handles.push(tokio::spawn(async move {
        let mut outcomes = Vec::new();
        let mut nonce = nonce_seed() ^ ((worker_id as u64) << 16);
        while Instant::now() < timed_deadline {
            let (prompt, prompt_local) = synthesize_prompt(&tokenizer_w, pp, nonce)?;
            let result = run_chat_completion(&client_w, &url, &model_w, &prompt, tg).await?;
            outcomes.push(RequestOutcome {
                worker_id,
                prompt_tokens_local: prompt_local,
                result,
            });
            nonce = nonce.wrapping_add(1);
        }
        Ok::<Vec<RequestOutcome>, anyhow::Error>(outcomes)
    }));
}
let mut all_outcomes: Vec<RequestOutcome> = Vec::new();
for h in timed_handles {
    all_outcomes.extend(h.await??);
}
let cell_end = Instant::now();

// 3. Build ConcurrentCellResult.
Ok(ConcurrentCellResult {
    target_name: target_name.into(),
    target_url: target_url.into(),
    pp_target: pp,
    tg_target: tg,
    concurrent,
    cell_start,
    cell_end,
    outcomes: all_outcomes,
})
```

`RequestOutcome` is the per-request analog of v1's `RunOutcome` with a `worker_id` field added. `ConcurrentCellResult` is the v2 analog of `CellResult` carrying cell wall-clock bounds + per-request outcomes.

### 4.3 Module surface

```text
iron-bench/src/main.rs            — MODIFY (~80 lines)
  ~ Add --concurrent / --duration / --warmup-duration flags
  ~ Mode dispatch: sequential (v1) vs concurrent (v2)
  ~ For v2: build Arc<Client>, Arc<Tokenizer>, call run_cell_concurrent per (pp, target)

iron-bench/src/runner.rs          — MODIFY (~100 lines)
  + run_cell_concurrent function
  + ConcurrentCellResult struct
  + RequestOutcome struct
  (existing run_cell + RunOutcome + CellResult unchanged for v1 mode)

iron-bench/src/report.rs          — MODIFY (~200 lines)
  + render_markdown_concurrent / render_csv_concurrent / render_json_concurrent
  + percentile helpers (p50, p95, p99 over sorted Vec<f64>)
  + per-worker breakdown formatting
  ~ Dispatch in main.rs based on cell type (sequential vs concurrent)

iron-bench/src/client.rs          — no change
iron-bench/src/prompt.rs          — no change

iron-bench/README.md              — MODIFY (~30 lines new section)
  + Section "v2 concurrent mode" with example invocation
  + Update "Single-request only" claim (now: "Sequential default; --concurrent N opt-in")
```

Total new code: ~410 lines. Net delta ~+450 (existing v1 code largely unchanged; main.rs grows by mode dispatch).

### 4.4 Percentile computation

Closed-loop on a single 30s cell typically yields 50-500 requests per worker. Total per cell: 200-2000 outcomes at N=4-8. Sort-and-index percentile is fine (O(M log M) on M ≤ 2000):

```rust
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
```

No HdrHistogram dep. Caveat: at very high concurrency × long duration (e.g., 64 workers × 5 min = potentially 100K requests), `Vec<RequestOutcome>` per cell grows. RequestOutcome is ~100B; 100K = 10MB. Acceptable.

### 4.5 Worker startup skew

`tokio::spawn` semantics: spawned task runs at the next reactor tick. With N=64 workers, all spawns happen in one polling cycle but actual first HTTP request firing depends on scheduler fairness. Under tokio multi-thread runtime (default ≥4 worker threads), startup skew is sub-millisecond.

Workers all share the same `Instant::now()` deadline reference, so even with 5ms startup skew, on a 30s cell the impact is < 0.02% — negligible. We don't compensate.

### 4.6 HTTP client pool sizing

`reqwest::Client` by default uses a connection pool. Under N=64 concurrent workers + 1 target, the pool will grow to ~64 connections to the same host. `reqwest::ClientBuilder::pool_max_idle_per_host` default is `usize::MAX`. No tuning needed; the OS-level limit (typically 1024-65535 file descriptors) is the constraint, far above our N.

For multi-target benchmarks, each target gets its own host connections; pool is per (host, port). 4 targets × 64 workers = ~256 connections. Still fine.

### 4.7 Tokenizer concurrency

`tokenizers::Tokenizer` is `Send + Sync` for `encode` operations (the hot path for `synthesize_prompt`). v1 passes `&Tokenizer` by reference; v2 wraps in `Arc<Tokenizer>` and clones the Arc into each worker task. No mutex needed.

### 4.8 Invariants

1. **Per-worker nonce uniqueness**: `nonce = nonce_seed() ^ ((worker_id as u64) << 16)` then `wrapping_add(1)` per iteration. Each worker generates distinct prompts across its lifetime (prevents server-side prefix-cache cross-iteration hits).
2. **`Arc<Tokenizer>` shared, no per-worker clone**: tokenizer load is expensive (~50ms for Qwen3.5 tokenizer.json); load once in main, share via Arc.
3. **`Arc<reqwest::Client>` shared**: same rationale; reuse connection pool across workers.
4. **Worker outcome aggregation**: each `tokio::spawn` returns its own `Vec<RequestOutcome>` via the join handle. Main thread `extend`s into a flat `Vec` for the cell. Total memory ~10MB worst-case per cell.
5. **Cell isolation**: each cell `for pp { for target { run_cell_concurrent(...) }}` waits for ALL N workers to finish via join before moving to the next cell. No overlap between cells.
6. **Timing reference**: `cell_start = Instant::now()` immediately before spawning the N timed workers; `cell_end = Instant::now()` after `await??` on all join handles. Wall duration = `cell_end - cell_start`; outcomes' TTFT/ITL are recorded by individual worker's `Instant::now()` calls referencing the same monotonic clock (no skew).

## §5 Tests

### 5.1 Unit tests for new aggregation logic

In `report.rs::tests`:

1. **`percentile_basic`**: hand-computed p50 / p95 / p99 on small fixtures (5, 10, 100 elements).
2. **`percentile_edge_cases`**: empty input → 0.0; single element → that element; all-same elements → that value at all percentiles.
3. **`render_markdown_concurrent_smoke`**: synthetic `ConcurrentCellResult` (3 outcomes, 1 worker), render markdown, assert output contains "p50 TTFT", "p95 TTFT", "tokens/s aggregate".

In `runner.rs::tests`:

4. **`nonce_uniqueness_across_workers`**: simulate 4 workers × 100 iterations each; collect all nonces; assert no collisions.

### 5.2 Integration smoke test

`iron-bench/tests/concurrent_smoke.rs` (new):

```rust
//! Smoke test: spin up a mock SSE server, run iron-bench v2 against it,
//! verify the report includes concurrent metrics.

#[tokio::test]
async fn concurrent_smoke_against_mock_server() {
    // 1. Launch a small axum mock server that streams 5 SSE chunks
    //    after a 20ms delay, then `data: [DONE]`.
    // 2. Spawn iron-bench v2 via std::process::Command with:
    //      --target mock=http://127.0.0.1:PORT
    //      --concurrent 2 --duration 1 --warmup-duration 0
    //      --prompt-len 16 --max-tokens 5
    //      --format json
    // 3. Parse stdout JSON; assert it has:
    //    - cells[0].concurrent == 2
    //    - cells[0].outcomes.len() > 0 (workers actually ran)
    //    - percentiles fields exist
}
```

(May need to fall back to dependency-injection if `std::process::Command` is awkward; the spawn-server-and-Command pattern is standard.)

### 5.3 No regression to v1

- `iron-bench --target X=URL --prompt-len 128 --runs 3 --warmup 1` (the v1 invocation form) must produce identical output to pre-v2 builds.
- Existing iron-bench memory note: "Boss preference: prefer iron-bench over custom perf scripts; iron-bench 已 fix omlx thinking-mode + handles SSE/median/warmup" — v2 preserves all those properties (no changes to `client.rs`).

## §6 Acceptance gates

- All new lib unit tests + integration smoke test PASS
- v1 sequential mode produces bit-identical output for an existing test invocation
- New `--concurrent N --duration S` mode produces a valid report with all expected percentile fields
- `cargo +nightly fmt --check`, `clippy -D warnings`, `cargo build --release -p iron-bench`: clean
- LOC growth: ~+450 (within plan estimate)

## §7 Estimate

**3–4 working days:**

- Day 1: New CLI flags + main.rs mode dispatch + ConcurrentCellResult / RequestOutcome structs
- Day 2: `run_cell_concurrent` runner + per-worker nonce + Arc-share
- Day 3: Report module — percentile helpers + 3 formatters (md / csv / json) + lib tests
- Day 4: Integration smoke test + README update + manual smoke against a real ironmlx server

This is the smallest sub-project I've spec'd since P6 — narrow scope, no model dependency, no GPU code. Risk profile: low.

## §8 Compat sunset notes

No new compat. v1 mode is preserved as the default and is the documented "compare two engines on identical workload" path. v2 mode is additive; opt-in via `--concurrent N`.

iron-bench v2 introduces no new sunset markers; it consumes ironmlx's existing OpenAI SSE protocol unchanged.

## §9 Risk register

| Risk | Mitigation |
| --- | --- |
| **R1.** Tokio runtime starvation at high N (>64 workers, default 4 worker threads) | Document recommended `TOKIO_WORKER_THREADS=N` env var in README. For ironmlx perf benches typical N=4–16, no tuning needed. |
| **R2.** Server-side rate limit hits cause workers to spin failing | Default `--timeout 300` already in v1; v2 reuses. On HTTP 429 / 503, `run_chat_completion` returns Err, worker propagates via join (cell fails fast). Documented as expected behavior — bench should not run during traffic. |
| **R3.** Wall-clock drift across workers | All workers use the same `Instant::now()` reference monotonic clock; no skew. Verified by integration smoke. |
| **R4.** Percentile computation incorrect at small M | Lib unit tests cover M=0, M=1, M=5. |
| **R5.** Worker outcome `Vec` memory blowup at extreme N×duration | Calculation: 100K outcomes × 100B = 10MB. Within budget. Documented as expected. |
| **R6.** ironmlx 3c-3 `Err("scheduler full")` mid-bench creates noisy failure | `run_chat_completion` propagates HTTP-level error; worker join returns Err. Cell fails. Solution: client must run at N ≤ ironmlx's b_max (typically 4) for clean closed-loop. Document this. v3 + ironmlx 3d (admission queue) lifts the constraint. |
| **R7.** Connection pool fd exhaustion at large N | Documented; `ulimit -n 65536` recommended in README for N > 256. Default macOS limit is 1024 which covers N ≤ 256. |

## §10 Alternatives considered

| Decision | Selected | Rejected |
| --- | --- | --- |
| Concurrency model | Closed-loop (Boss A) | Open-loop (Poisson — defer to v3); both modes in v2 (scope creep) |
| v1 compat | Preserve as default (no breakage) | Replace v1 with v2 (would require updating all existing benchmark invocations in any docs / scripts) |
| Stopping criterion (v2) | Wall-clock `--duration S` | Request-count `--n-requests M` (less natural for closed-loop; M depends on server speed); both (CLI bloat) |
| Percentile data structure | Sort `Vec<f64>` + index | HdrHistogram (overkill at M ≤ 2000); streaming quantile estimator (premature optimization) |
| Per-worker reporting | Both aggregate + per-worker breakdown | Aggregate only (loses worker-balance insight); per-worker only (no headline number) |
| Tokenizer sharing | `Arc<Tokenizer>` shared across workers | `Tokenizer` per-worker (50ms load × N = startup cost wasted) |
| HTTP client | `Arc<reqwest::Client>` shared (connection pool reuse) | Per-worker `Client` (loses connection reuse, hurts measured throughput by ~5-10% on local benches) |
| Warmup mode (v2) | Same workers run for `--warmup-duration S` then discard | Skip warmup entirely (first-iteration MLX compile cost pollutes timed phase) |
| Smoke test approach | Mock SSE server via axum (in-process) | Real ironmlx server in CI (requires model fixture; flaky); skip smoke test (no automated regression) |
| Open-loop deferral | v3 + ironmlx 3d (admission queue available) | v2 (premature without backpressure support on server side) |

## §11 Linked artifacts

- iron-bench v1 spec: [`docs/superpowers/specs/2026-05-08-p7-iron-bench-design.md`](2026-05-08-p7-iron-bench-design.md)
- iron-bench README (current single-request claim): [`iron-bench/README.md:88-89`](../../iron-bench/README.md#L88)
- iron-bench v1 runner (target of refactor): [`iron-bench/src/runner.rs`](../../iron-bench/src/runner.rs)
- iron-bench v1 client (no change in v2): [`iron-bench/src/client.rs`](../../iron-bench/src/client.rs)
- iron-bench v1 report (target of new formatters): [`iron-bench/src/report.rs`](../../iron-bench/src/report.rs)
- ironmlx 3c-3 close-out (continuous batching shipped — unblocks this work): [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout/report.md)
