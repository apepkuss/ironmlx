# MTP Phase 5 Server API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose server-level MTP configuration and live scheduler MTP counters through the existing `/healthz` JSON API.

**Architecture:** Keep MTP as a server-startup feature in Phase 5 and do not add per-request OpenAI parameters. Thread a small immutable MTP server config plus existing SchedulerActor MTP atomics into `SchedulerHealthCollector`, then serialize an `mtp` object in every health snapshot.

**Tech Stack:** Rust, axum state, serde, existing scheduler actor atomics.

---

### Task 1: Add Health MTP Snapshot Shape

**Files:**
- Modify: `ironmlx/src/core/server/health.rs`

- [ ] **Step 1: Write failing health snapshot tests**

Add tests that construct `SchedulerHealthCollector` with no MTP config and with enabled MTP config. Assert that disabled snapshots serialize `enabled=false`, and enabled snapshots include `draft_tokens`, `prefill_count`, and `step_count`.

- [ ] **Step 2: Run RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx core::server::health::tests::snapshot_mtp -- --nocapture
```

Expected: fail because the `mtp` snapshot fields do not exist.

- [ ] **Step 3: Implement health MTP structs**

Add `MtpHealthConfig`, `MtpHealthInfo`, and an `mtp` field to `HealthSnapshot`. Store optional MTP config and MTP counters in `SchedulerHealthCollector::snapshot`.

- [ ] **Step 4: Run GREEN**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx core::server::health::tests::snapshot_mtp -- --nocapture
```

Expected: both snapshot tests pass.

### Task 2: Wire Server MTP Config Into Health Collector

**Files:**
- Modify: `ironmlx/src/core/server/mod.rs`
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`

- [ ] **Step 1: Write failing server-level tests**

Add tests in `server::tests` for the helper that builds a health collector from a plain scheduler handle and from an MTP-enabled handle. Assert plain collectors report disabled MTP and MTP collectors report the configured draft token count.

- [ ] **Step 2: Run RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx core::server::tests::health_collector_mtp -- --nocapture
```

Expected: fail because the helper/config wiring does not exist.

- [ ] **Step 3: Implement wiring**

Expose a small `SchedulerMtpHealthConfig` from the scheduler actor handle, have the plain spawner return `None`, and have the MTP spawner return `Some { draft_tokens }`. Use a `build_health_collector` helper so tests do not need to start an HTTP server.

- [ ] **Step 4: Run GREEN**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx core::server::tests::health_collector_mtp -- --nocapture
```

Expected: server-level MTP health wiring tests pass.

### Task 3: Verify And Document API Contract

**Files:**
- Modify: `docs/benchmarks/mtp-phase4-policy/2026-06-07-160427/summary.md` only if needed.
- Create: `docs/mtp-server-api.md`

- [ ] **Step 1: Add concise API documentation**

Document that Phase 5 exposes MTP as a startup-level server feature through `/healthz`, not as an OpenAI per-request parameter. Include the JSON shape and current constraints.

- [ ] **Step 2: Run focused tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx health_collector_mtp snapshot_mtp -- --nocapture
```

Expected: all focused tests pass.

- [ ] **Step 3: Run required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all commands exit 0.
