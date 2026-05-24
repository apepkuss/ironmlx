# P5h+2.c — Scheduler Finished-phase ERROR Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the per-request `[SchedulerActor] step error: step illegal in Finished phase: call prefill_admitted first` ERROR that triggered P5h+2.b acceptance failure, by adding actor-side `Finished`-batch finalization + extracting empty-scheduler handoff into a shared helper. Preserve `scheduler.rs` fail-fast semantics; add a regression test proving the ERROR branch is no longer hit.

**Architecture:** Two new private helpers in `ironmlx/src/core/server/scheduler_actor.rs`: `finalize_finished_batch_if_any` (no-op if `phase != Finished`; otherwise `evict_all` + clear `event_txs`) and `drive_empty_scheduler_handoff` (first finalizes `Finished`, then returns `RollingControl` enum encoding `ContinueRolling/BreakRolling/ContinueOuter/ReturnActor`, replacing the existing duplicated empty-batch transition block). Rolling-loop top calls the handoff helper before any event pick when `phase == Finished` (covers biased select Admit preference per Codex Q6). Outer-loop top runs defensive finalize (covers prior-iter leak). Driver-side: `tools/p5h_2b_protocol_experiment.py` aborts on `step illegal in <phase>` ERROR detection unless `--allow-server-errors` flag is set.

**Tech Stack:** Rust (`ironmlx/src/core/server/scheduler_actor.rs` + `ironmlx/src/core/scheduler.rs` + new `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs`), Python 3 (driver guard in `tools/p5h_2b_protocol_experiment.py`), `cargo` (fmt + nightly fmt + clippy + build + test), `uv` for Python tooling.

**Spec ref:** `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix-design.md` (current working-copy design).

---

## File structure (created / modified by this plan)

**Create:**
- `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs` — actor integration test proving `step illegal in Finished phase` ERROR not hit on 3 sequential `max_new_tokens=1` requests (T2; uses existing `QWEN35_MODEL` fixture pattern)
- `docs/p5h+2-c-close-out.md` — T3 close-out (PASS path; will be committed)
- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2c_findings.md` — memory entry (outside repo)
- `reports/p5h+2-c-bench-log.md` — gitignored bench/test log

**Modify:**
- `ironmlx/src/core/server/scheduler_actor.rs` — add `RollingControl` enum + `finalize_finished_batch_if_any` + `drive_empty_scheduler_handoff` helpers; rolling-loop-top hook; outer-loop-top defensive hook; replace existing empty-handoff block at line 435 with helper call; add narrow `p5h-profile` counter for integration-test proof
- `ironmlx/src/core/scheduler.rs` — add 2 unit tests in `#[cfg(test)] mod tests` using a local fake model (T1; locks `max_new_tokens=1 → phase=Finished` AND `step(Finished)` still Err)
- `tools/p5h_2b_protocol_experiment.py` — `check_no_scheduler_errors` post-cell guard + `--allow-server-errors` CLI flag
- `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md` — add `project-p5h-2c-findings` index entry

**Do NOT modify:**
- `ironmlx/src/core/scheduler.rs` core semantics (`step_inner` phase guard at line 1286, `prefill_admitted_inner` at line 1241-1250, `evict_all` at line 790-797 — all unchanged; only tests added)
- `ironmlx/tests/p5h_t5_attribution_capture.rs` (P5h T5 baseline)
- `ironmlx/tests/p5i_c_phase_0_capture.rs` (P5h+2.b capture harness; reusable as-is)
- iron-bench (bug is server-side; iron-bench just exposes)
- Other P5h+2.b infrastructure (multi_repeat / pp_tps_envelope / phase0_compose / etc.)

---

## Task 1: T0 — Actor finalization + empty-handoff helpers + driver guard (~2 hr)

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs:262-616` (lots of changes; details below)
- Modify: `tools/p5h_2b_protocol_experiment.py` (add `check_no_scheduler_errors` + `--allow-server-errors`)

- [ ] **Step 1.1: Read existing scheduler_actor.rs control flow to ground refactor**

```bash
cd /Users/xin/workspace/ironmlx-backend
grep -n "^fn \|^pub fn \|^async fn\|^pub async fn\|^enum " ironmlx/src/core/server/scheduler_actor.rs | head -20
sed -n '262,290p' ironmlx/src/core/server/scheduler_actor.rs    # outer loop entry
sed -n '337,360p' ironmlx/src/core/server/scheduler_actor.rs    # rolling loop top
sed -n '425,610p' ironmlx/src/core/server/scheduler_actor.rs    # active_count==0 + queue drain + try_recv re-prefill
```

Confirm: 4 empty-handoff sites at lines 279 (outer-loop first-admit failed), 435 (rolling-loop top-level post-step), 465 (nested queue-drain), 552 (nested try_recv). Verify imports already include `Phase` from `crate::core::scheduler`.

- [ ] **Step 1.2: Add `RollingControl` enum + `finalize_finished_batch_if_any` helper**

Append after existing `enum RollingEvent` definition (find via `grep -n "enum RollingEvent" ironmlx/src/core/server/scheduler_actor.rs`):

```rust
/// Result returned by [`drive_empty_scheduler_handoff`] encoding what the
/// caller's rolling loop should do next. Matches the existing `continue
/// 'rolling` / `break 'rolling` / `continue 'outer` / `return` patterns
/// without exposing label control to the helper.
///
/// Added by P5h+2.c to make the empty-batch handoff path reusable from
/// (a) the existing post-step empty-handoff site and (b) the new
/// pre-event Finished-batch finalization at the rolling-loop top.
enum RollingControl {
    /// Re-enter the rolling loop (a new batch was admitted + prefilled).
    ContinueRolling,
    /// Exit the rolling loop into the outer-loop tail cleanup (no
    /// queued or pending admits; outer will block on `cmd_rx.recv()`).
    BreakRolling,
    /// `continue 'outer` — outer loop body resumes from its top
    /// (e.g., poisoned-state recovery).
    ContinueOuter,
    /// `return` from the actor (cmd_rx disconnected; all senders dropped).
    ReturnActor,
}

/// Finalize a `Phase::Finished` batch: evict slots + release budget +
/// reset to `Phase::Idle`, then close per-request event channels.
///
/// Returns `Ok(true)` if finalization happened (caller MUST go to the
/// empty-scheduler handoff path, NOT continue the normal event pick;
/// per spec § 4.2.1 hard binding).
/// Returns `Ok(false)` if `phase != Finished` (no-op; safe to continue).
/// Returns `Err` if `evict_all` failed (caller should reject queued
/// admits + `continue 'outer` per existing pattern).
///
/// Added by P5h+2.c. The `Phase::Finished` state arises naturally when
/// `prefill_admitted` completes a batch where every request has
/// `max_new_tokens=1` (the prefill samples first+last token in one
/// pass), which is the standard `iron-bench --max-tokens 1` perf
/// measurement workload.
fn finalize_finished_batch_if_any<M: Model>(
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
) -> Result<bool> {
    if sched.phase() != Phase::Finished {
        return Ok(false);
    }
    match sched.evict_all() {
        Ok(()) => {
            event_txs.clear();
            Ok(true)
        }
        Err(e) => {
            tracing::warn!(
                "[SchedulerActor] finalize_finished_batch: evict_all failed: {e:?}"
            );
            Err(e)
        }
    }
}
```

- [ ] **Step 1.3: Add `drive_empty_scheduler_handoff` helper**

Append immediately after `finalize_finished_batch_if_any`. This helper encapsulates the existing empty-batch handoff at line 435 (queued admits → try_recv → break) so the new rolling-loop-top hook can share the same logic.

Read existing block at lines 435-604 to ground exact structure:

```bash
sed -n '435,604p' ironmlx/src/core/server/scheduler_actor.rs | wc -l
```

Then write the helper using the documentation and fixed signature below. This snippet is the signature contract only; the function body is the moved code from Step 1.4, not a separate stub.

```rust
/// Finalize a just-finished batch if needed, then drain queued admits
/// (or a single pending `cmd_rx.try_recv` admit) into a fresh batch, run
/// `prefill_admitted`, and return how the caller's rolling loop should
/// proceed. Lifts the existing empty-batch transition logic at the
/// rolling-loop tail so it can also be invoked from the new pre-event
/// Finished-batch finalization at the rolling-loop top.
///
/// This helper is the single empty-batch handoff path. It first calls
/// [`finalize_finished_batch_if_any`], so callers must not separately
/// finalize before invoking it. After that call the scheduler is either
/// `Phase::Idle` or an unexpected `Phase::Decoding`/`Phase::Finished`
/// empty state from the legacy post-step path; the helper preserves the
/// current reset semantics for those legacy states before starting the
/// next batch.
///
/// Behavior per branch:
/// - Queued admit present → pop head, fresh batch via `handle_admit` +
///   `drain_window` + `prefill_admitted`; returns `ContinueRolling`.
/// - Queue empty + `cmd_rx.try_recv()` returns `Ok(cmd)` → fresh batch
///   via the same path; returns `ContinueRolling`.
/// - Queue empty + `try_recv` returns `Empty` → returns `BreakRolling`.
/// - Queue empty + `try_recv` returns `Disconnected` → clear `event_txs`,
///   returns `ReturnActor`.
/// - Any `finalize`, legacy reset, or `prefill_admitted` failure →
///   reject queued admits, clear `event_txs`, returns `ContinueOuter`.
///
/// Added by P5h+2.c. Replaces the existing `if sched.active_count() == 0
/// { ... }` block at rolling-loop tail to avoid divergent copies.
#[allow(clippy::too_many_arguments)]
fn drive_empty_scheduler_handoff<M>(
    sched: &mut Scheduler<M>,
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    model: &Arc<Mutex<M>>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    batch_count: &Arc<AtomicU64>,
    b_max: usize,
    admission_queue_max: usize,
    admission_deadline: Duration,
    rt: &tokio::runtime::Handle,
) -> RollingControl
where
    M: Model + DenseVlMethods + Send + 'static
```

The committed code must contain no Rust stub macro or placeholder panic in this helper. Step 1.4 is mandatory, not optional.

- [ ] **Step 1.4: Lift existing rolling-loop-tail block (lines 435-604) into helper body**

Read the existing block. Approximately (verify against actual code):

```rust
// At line 435+ in current rolling loop:
if sched.active_count() == 0 {
    if !admission_queue.is_empty() {
        // ... evict_all + pop head + handle_admit + drain_window +
        //     prefill_admitted + match → ContinueRolling or ContinueOuter
    }
    // Queue empty + no active rows
    match cmd_rx.try_recv() {
        Ok(cmd) => {
            // evict_all + handle_admit + drain_window + prefill_admitted
            // → ContinueRolling
        }
        Err(TryRecvError::Empty) => break 'rolling,
        Err(TryRecvError::Disconnected) => {
            event_txs.clear();
            return;
        }
    }
}
```

Move this verbatim into `drive_empty_scheduler_handoff` body (Step 1.3), with these exact structural edits:

- Add at helper body top:

```rust
match finalize_finished_batch_if_any(sched, event_txs) {
    Ok(_) => {}
    Err(_e) => {
        while let Some(pending) = admission_queue.pop_front() {
            let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                "scheduler poisoned during Finished-batch finalize"
            )));
        }
        event_txs.clear();
        return RollingControl::ContinueOuter;
    }
}
```

- Move the inside of the existing `if sched.active_count() == 0 { ... }` block into the helper body, not the outer `if` itself.
- Replace `continue 'rolling` with `return RollingControl::ContinueRolling;`.
- Replace `break 'rolling` with `return RollingControl::BreakRolling;`.
- Replace `continue 'outer` with `return RollingControl::ContinueOuter;`.
- Replace actor-exit `return;` with `return RollingControl::ReturnActor;`.
- Preserve existing `tracing::warn!` / `tracing::error!` messages and comments unless a comment explicitly refers to the old inline location.
- Preserve the current behavior that resets `Phase::Decoding`/`Phase::Finished` empty states before starting a fresh batch, while skipping `evict_all()` when already `Phase::Idle`.

Then at the rolling-loop original site (line 435), REPLACE the entire `if sched.active_count() == 0 { ... }` block with:

```rust
// P5h+2.c: extract empty-batch handoff into reusable helper.
if sched.active_count() == 0 {
    match drive_empty_scheduler_handoff(
        &mut sched,
        &mut cmd_rx,
        &mut event_txs,
        &mut admission_queue,
        &model,
        &admit_count,
        &saturate_triggered,
        &queue_depth_peak,
        &queue_rejected,
        &batch_count,
        b_max,
        admission_queue_max,
        admission_deadline,
        &rt,
    ) {
        RollingControl::ContinueRolling => continue 'rolling,
        RollingControl::BreakRolling => break 'rolling,
        RollingControl::ContinueOuter => continue 'outer,
        RollingControl::ReturnActor => return,
    }
}
```

- [ ] **Step 1.5: Hook rolling-loop TOP — pre-event Finished-batch finalization**

Insert at line 337 (very top of `'rolling: loop`), BEFORE the `tokio::select!`:

```rust
'rolling: loop {
    // P5h+2.c: pre-event Finished-batch finalization + handoff. If
    // previous iteration's prefill_admitted/step left phase=Finished
    // (e.g. max_tokens=1 workload), handle the completed batch BEFORE
    // dispatching another event. Per Codex Q6: biased select may pick
    // Admit over Step, so this must run before the event pick — or the
    // actor could call admit_mid_begin() in Phase::Finished.
    if sched.phase() == Phase::Finished {
        // `drive_empty_scheduler_handoff` itself calls
        // `finalize_finished_batch_if_any`; do not duplicate finalization
        // here. This avoids two divergent finalize/error paths.
        match drive_empty_scheduler_handoff(
            &mut sched,
            &mut cmd_rx,
            &mut event_txs,
            &mut admission_queue,
            &model,
            &admit_count,
            &saturate_triggered,
            &queue_depth_peak,
            &queue_rejected,
            &batch_count,
            b_max,
            admission_queue_max,
            admission_deadline,
            &rt,
        ) {
            RollingControl::ContinueRolling => continue 'rolling,
            RollingControl::BreakRolling => break 'rolling,
            RollingControl::ContinueOuter => continue 'outer,
            RollingControl::ReturnActor => return,
        }
    }

    let evt: RollingEvent = rt.block_on(async {
        // ... existing tokio::select! ...
    });
    // ... existing match arms ...
}
```

- [ ] **Step 1.6: Hook outer-loop TOP — defensive finalize before next blocking recv**

Insert at line 269 (very top of `'outer: loop`), BEFORE the `cmd_rx.recv()` block:

```rust
'outer: loop {
    // P5h+2.c defensive: ensure scheduler is in Phase::Idle before
    // blocking on next admit. Most error paths already call evict_all,
    // but this guards any future code path that leaves phase=Finished.
    // If finalize fails, the actor cannot safely admit more requests
    // (the scheduler would be in an unrecoverable state); terminate
    // cleanly rather than emit ERROR per request.
    if sched.phase() == Phase::Finished {
        if let Err(e) = finalize_finished_batch_if_any(&mut sched, &mut event_txs) {
            tracing::error!(
                "[SchedulerActor] outer-loop finalize failed: {e:?}; \
                 actor cannot reset Finished batch safely — terminating"
            );
            event_txs.clear();
            return;
        }
    }

    // ===== Outer Idle: block waiting for first admit (or shutdown). =====
    let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
        return;
    };
    handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);
    // ... rest of existing outer-loop body unchanged ...
}
```

- [ ] **Step 1.7: Build verify (early — before Python work)**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo build --release -p ironmlx --features p5h-profile --tests 2>&1 | tail -10
```

Expected: clean build. If errors, fix typos / missing imports. `scheduler_actor.rs` already imports `Arc`, `Duration`, `AtomicU64`, and `AtomicUsize`; keep `queue_depth_peak` typed as `Arc<AtomicUsize>`.

- [ ] **Step 1.8: Add Python driver guard**

Edit `tools/p5h_2b_protocol_experiment.py`. Add module-level fn after imports:

```python
def check_no_scheduler_errors(server_log_path: Path, allow_server_errors: bool) -> None:
    """Acceptance precondition per Codex round-3 design question #3.

    Inspects server.log for `step illegal in <phase> phase` ERROR lines
    (production scheduler phase-guard violations). Default-deny: any
    such ERROR aborts the sweep + preserves the artifact directory.
    Diagnostic experiments wanting to allow these ERRORs explicitly set
    --allow-server-errors.
    """
    if allow_server_errors:
        return
    if not server_log_path.exists():
        return  # missing log handled by caller's downstream check
    count = 0
    with server_log_path.open() as f:
        for line in f:
            if "step illegal in" in line and "phase" in line:
                count += 1
    if count > 0:
        raise SystemExit(
            f"{server_log_path}: {count} `step illegal in <phase>` ERROR lines detected. "
            "Acceptance precondition VIOLATED (Codex round-3 design question #3). "
            "Re-run with --allow-server-errors to override for diagnostic experiments."
        )
```

Add CLI flag in the argparse setup (search for `p.add_argument("--mlx-dir"`):

```python
p.add_argument(
    "--allow-server-errors",
    action="store_true",
    default=False,
    help="Allow `step illegal in <phase>` server ERROR lines (default: abort sweep). "
    "Use for diagnostic experiments where scheduler errors are expected.",
)
```

Hook into per-cell flow. Locate `run_one_repeat` — after each cell relocation (`shutil.move(...)`), add the check:

```python
# After: shutil.move(str(src), str(dst))
check_no_scheduler_errors(dst / "server.log", args.allow_server_errors)
```

Loop through all PPs in the for loop so each cell is checked individually.

- [ ] **Step 1.9: Python hygiene check**

```bash
cd /Users/xin/workspace/ironmlx-backend
uv run --with ruff ruff check tools/p5h_2b_protocol_experiment.py
uv run --with ruff ruff format --check tools/p5h_2b_protocol_experiment.py
```

Expected: clean.

- [ ] **Step 1.10: Full Rust gate (early — catch clippy issues before tests)**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings 2>&1 | tail -10
cargo build --release
```

Expected: all PASS. Do not add broad clippy allowances during this task; if clippy flags a real issue, follow the diagnostic and keep the helper contracts intact.

**No commit at T0** (single-commit policy per spec § 5.4; T3 commits everything atomically).

---

## Task 2: T1 — Scheduler unit tests lock fail-fast semantic (~30 min)

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs::tests` (add 2-3 tests)

- [ ] **Step 2.1: Read existing test patterns to ground new tests**

```bash
cd /Users/xin/workspace/ironmlx-backend
grep -n "#\[test\]\|fn step_\|fn prefill\|force_phase\|type TestScheduler\|fn mk_req" ironmlx/src/core/scheduler.rs | head -40
sed -n '2200,2310p' ironmlx/src/core/scheduler.rs
```

Confirm: existing unit tests use `TestScheduler = Scheduler<Qwen35Model>` for tests that do not drive model forward. For the new prefill test, add a tiny local fake model inside `mod tests` so the test does not require `QWEN35_MODEL` and does not instantiate a real Qwen model.

- [ ] **Step 2.2: Add local fake model for scheduler unit tests**

Append this helper inside `#[cfg(test)] mod tests`, after `mk_req` and before the first `#[test]`:

```rust
    struct P5h2cFakeModel;

    impl Model for P5h2cFakeModel {
        fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> crate::Result<Vec<LayerCache>> {
            Ok(Vec::new())
        }

        fn forward_on(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<Array> {
            unreachable!("P5h+2.c unit tests never call decode forward")
        }

        fn batched_prefill(
            &self,
            input_ids: &Array,
            _position_ids: &Array,
            _attention_mask: &Array,
            _linear_attention_mask: &Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<Array> {
            let b = input_ids.shape().as_slice()[0] as usize;
            let vocab = 8_usize;
            let mut flat = vec![0.0_f32; b * vocab];
            for row in 0..b {
                flat[row * vocab + 3] = 100.0;
            }
            let logits_bv: Array = (&flat[..], &[b as i32, vocab as i32][..])
                .try_into()
                .expect("fake logits [B,V]");
            logits_bv
                .reshape(&[b as i32, 1, vocab as i32][..])
                .map_err(|e| anyhow::anyhow!("fake logits reshape failed: {e:?}"))
        }

        fn forward_text_hidden(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<Array> {
            unreachable!("P5h+2.c unit tests never call chunk hidden forward")
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl DenseVlMethods for P5h2cFakeModel {
        fn batched_prefill_vl(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _attention_mask: &Array,
            _linear_attention_mask: &Array,
            _per_row_lens: &[i32],
            _per_row_pixel_values: &[Option<&Array>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            _cache: Option<&mut [LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<Array> {
            unreachable!("P5h+2.c unit tests are text-only")
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &Array,
            _grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<Array> {
            unreachable!("P5h+2.c unit tests are text-only")
        }

        fn forward_vl_chunk(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _vision_embeds_slice: Option<&Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> crate::Result<Array> {
            unreachable!("P5h+2.c unit tests are text-only")
        }
    }
```

- [ ] **Step 2.3: Add `test_max_new_tokens_1_transitions_to_finished_after_prefill`**

Locate end of `mod tests` (find via `grep -n "^}" ironmlx/src/core/scheduler.rs | tail -5`). Append BEFORE the closing `}`:

```rust
    /// P5h+2.c regression: `max_new_tokens=1` requests must transition
    /// scheduler to `Phase::Finished` after `prefill_admitted` (the first
    /// sampled token is also the last token → request finished → no
    /// `any_unfinished` → phase = Finished per scheduler.rs:1247-1250).
    ///
    /// This locks the spec invariant that the actor's
    /// `finalize_finished_batch_if_any` helper relies on.
    #[test]
    fn test_max_new_tokens_1_transitions_to_finished_after_prefill() {
        let model = P5h2cFakeModel;
        let mut s = Scheduler::<P5h2cFakeModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");

        let mut req = mk_req(vec![1, 2, 3, 4]);
        req.max_new_tokens = 1;
        req.stop_token_ids = vec![];

        let id = s.admit(req).expect("admit OK");
        assert!(matches!(s.phase(), Phase::Admitting | Phase::Idle));
        let events = s.prefill_admitted(&model).expect("prefill OK");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, id);
        assert_eq!(events[0].finish_reason, Some("length"));
        assert_eq!(
            s.phase(),
            Phase::Finished,
            "max_new_tokens=1 should transition to Finished after prefill"
        );
    }
```

- [ ] **Step 2.4: Add `test_step_finished_phase_still_returns_err`**

```rust
    /// P5h+2.c regression: `step` MUST still raise an Err in
    /// `Phase::Finished` to preserve fail-fast discipline. The actor-side
    /// fix in P5h+2.c works AROUND this guard (via pre-event finalization)
    /// rather than relaxing it. If this test ever passes by returning Ok,
    /// the scheduler core semantics were silently changed.
    #[test]
    fn test_step_finished_phase_still_returns_err() {
        let model = P5h2cFakeModel;
        let mut s = Scheduler::<P5h2cFakeModel>::new(
            1,
            32768,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("scheduler startup");
        // Force scheduler into Finished phase (use existing test seam).
        s.force_phase(Phase::Finished);
        let result = s.step(&model);
        assert!(result.is_err(), "step in Phase::Finished must return Err");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("step illegal in Finished phase"),
            "expected `step illegal in Finished phase` in error, got: {err_msg}"
        );
    }
```

(`force_phase` is the test-only seam at scheduler.rs:~2270; reuse.)

- [ ] **Step 2.5: Run new unit tests**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo test --release -p ironmlx --lib test_max_new_tokens_1 test_step_finished 2>&1 | tail -15
```

Expected: 2 new tests PASS without `QWEN35_MODEL`.

- [ ] **Step 2.6: Run full scheduler test suite — no regression**

```bash
cargo test --release -p ironmlx --lib scheduler 2>&1 | tail -20
```

Expected: all existing scheduler tests STILL pass (no semantic changes to scheduler.rs core).

**No commit at T1.**

---

## Task 3: T2 — Actor integration test proves ERROR not hit (~1 hr)

**Files:**
- Create: `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs`

- [ ] **Step 3.1: Pattern-match existing actor integration test**

```bash
cd /Users/xin/workspace/ironmlx-backend
sed -n '1,50p' ironmlx/tests/b1_p2_3c_plus_chunked_admit_mid.rs
```

This file already uses `spawn_scheduler_actor` + `SchedulerActorHandle` + `SchedulerCommand`. Use the same imports + helper pattern.

- [ ] **Step 3.2: Add `p5h-profile` test-observable counter (fallback proof per spec § 5.3)**

Reliable ERROR-detection in `tracing` capture is brittle across test harness setups. Per spec § 5.3 acceptance, the preferred proof is captured tracing/stderr OR a narrow counter. Because `ironmlx/tests/*` compiles `ironmlx` as a dependency, library `cfg(test)` items are not visible to this integration test. Gate the counter under `p5h-profile` instead; this keeps default release builds untouched and adds cost only on the specific profiled error branch.

Edit `ironmlx/src/core/server/scheduler_actor.rs` step error branch (at line 405 area). Add module-level counter:

```rust
#[cfg(feature = "p5h-profile")]
#[doc(hidden)]
pub static STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
```

In the existing `Err(e)` branch of `RollingEvent::Step` handler (line 405):

```rust
Err(e) => {
    tracing::error!("[SchedulerActor] step error: {e:?}");
    #[cfg(feature = "p5h-profile")]
    if format!("{e:?}").contains("step illegal in Finished phase") {
        STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT
            .fetch_add(1, Ordering::Relaxed);
    }
    // ... rest of existing error handling (evict_all + clear admission_queue + continue 'outer) ...
}
```

This counter is compiled only for `p5h-profile` builds. Default production builds are untouched; profiled builds pay one string check only when the Step error branch is already executing.

- [ ] **Step 3.3: Create `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs`**

```rust
//! P5h+2.c regression smoke — verify SchedulerActor no longer triggers
//! `step illegal in Finished phase` ERROR for max_new_tokens=1 requests.
//!
//! The bug (P5h+2.b root cause): `prefill_admitted` transitions
//! `phase = Finished` for `max_new_tokens=1` requests; the rolling loop's
//! biased `tokio::select!` falls through to `RollingEvent::Step` when
//! cmd_rx is empty; `sched.step()` rejects the Finished phase; actor
//! logs ERROR + `evict_all` + restarts outer loop. Per-request cycle
//! polluted P5h+2.b acceptance with 1116 ERROR lines per cell.
//!
//! P5h+2.c fix: actor-side pre-event finalization at rolling-loop top
//! evicts the Finished batch before any event pick. This test sends 3
//! sequential max_new_tokens=1 requests and asserts the p5h-profile
//! `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` counter stays at 0.

#![cfg(feature = "p5h-profile")]

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{oneshot, Mutex};

use ironmlx::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
use ironmlx::core::model::Model;
use ironmlx::core::sampler::Sampler;
use ironmlx::core::server::scheduler_actor::{
    spawn_scheduler_actor, SchedulerActorHandle, SchedulerCommand,
    STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT,
};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

fn make_request(prompt_ids: Vec<u32>, stop_token_ids: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens: 1,
        sampler: Sampler::greedy(),
        stop_token_ids,
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
        p5h_trace: None,
        p5h_root_span: None,
    }
}

async fn admit_and_expect_single_finished_event(
    handle: &SchedulerActorHandle,
    request: GenerateRequest,
) {
    let (reply_tx, reply_rx) = oneshot::channel();
    handle
        .cmd_tx
        .send(SchedulerCommand::Admit { request, reply_tx })
        .await
        .expect("cmd send");
    let reply = reply_rx.await.expect("admit reply").expect("admit OK");
    let mut event_rx = reply.event_rx;
    let ev = tokio::time::timeout(Duration::from_secs(60), event_rx.recv())
        .await
        .expect("event timeout")
        .expect("first event");
    assert!(
        ev.finish_reason.is_some(),
        "max_new_tokens=1 should finish on the prefill event"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "p5h+2-c — actor smoke proving Finished-phase ERROR branch not hit"]
async fn test_scheduler_actor_max_tokens_1_no_finished_phase_error() {
    // Reset counter at test start (other tests in suite may have touched it).
    STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT.store(0, Ordering::Relaxed);

    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let loader = Loader::open(Path::new(&model_dir)).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let meta = model.model_meta();
    let model = Arc::new(Mutex::new(model));
    let handle =
        spawn_scheduler_actor(model, 4, Duration::from_millis(5), 32, 32768, meta)
            .expect("spawn_scheduler_actor");

    let prompt_ids = tokenize_prompt(&tokenizer, "Say one short word.");
    let stop_token_ids = tokenizer.eos_token_ids().to_vec();

    // Send 3 sequential max_new_tokens=1 admit cmds. Each should complete
    // (first token sampled) without triggering the Finished-phase step
    // error in the rolling loop.
    for i in 0..3 {
        let request = make_request(prompt_ids.clone(), stop_token_ids.clone());
        admit_and_expect_single_finished_event(&handle, request).await;
        eprintln!("[p5h+2-c smoke] request {i} completed");

        // Brief pause so the actor's rolling loop has a chance to attempt
        // a Step before next admit arrives (this is when the bug would
        // fire — `cmd_rx.recv()` empty → biased fall-through to Step).
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Final pause to ensure the actor has fully settled (any post-step
    // ERROR would have fired by now).
    tokio::time::sleep(Duration::from_millis(200)).await;

    let hit = STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        hit, 0,
        "expected 0 `step illegal in Finished phase` errors, got {hit}; \
         P5h+2.c fix regressed — actor is calling step() in Phase::Finished"
    );

    // Shutdown actor cleanly.
    drop(handle);
}
```

This test intentionally follows the existing ignored `QWEN35_MODEL` fixture pattern from `b1_p2_3b_2_scheduler_actor.rs` / `b1_p2_3c_plus_chunked_admit_mid.rs`; no fake actor model is used in the integration test.

- [ ] **Step 3.4: Run integration test**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo test --release -p ironmlx --features p5h-profile \
  --test p5h_2c_scheduler_finished_smoke -- --ignored --nocapture 2>&1 | tail -20
```

Expected: PASS — counter == 0.

- [ ] **Step 3.5: Run full ironmlx test suite — no regression**

```bash
cargo test --release -p ironmlx 2>&1 | tail -10
```

Expected: all existing tests STILL pass (b1_p2_* + p5h_* + others).

**No commit at T2.**

---

## Task 4: T3 — Rust gate + close-out + SINGLE final commit (~30 min)

**Files:**
- Create: `docs/p5h+2-c-close-out.md` (committed)
- Create: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2c_findings.md` (outside repo)
- Modify: `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md` (add index entry)
- Append: `reports/p5h+2-c-bench-log.md` (gitignored)
- COMMIT: scheduler_actor.rs + scheduler.rs + new integration test + protocol_experiment.py + close-out

- [ ] **Step 4.1: Full Rust gate**

```bash
cd /Users/xin/workspace/ironmlx-backend
export MLX_DIR=$HOME/.local/mlx
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings 2>&1 | tail -3
cargo build --release
cargo test --release -p ironmlx --features p5h-profile \
  --test p5h_2c_scheduler_finished_smoke -- --ignored --nocapture 2>&1 | tail -20
cargo test --release -p ironmlx 2>&1 | grep -E "test result|^running" | head -20
cargo test --release -p iron-bench 2>&1 | grep -E "test result|^running" | head -10
```

Expected: ALL pass, including the feature-gated ignored smoke.

- [ ] **Step 4.2: Python hygiene + pytest no-regression**

```bash
uv run --with ruff ruff check tools/p5h_2b_protocol_experiment.py
uv run --with ruff ruff format --check tools/p5h_2b_protocol_experiment.py
uv run --with pytest python -m pytest tools/p5h_aggregator/tests/ -v 2>&1 | tail -5
```

Expected: 139 PASS (no aggregator regression; no new Python tests in P5h+2.c).

- [ ] **Step 4.3: Write `docs/p5h+2-c-close-out.md`**

```markdown
# P5h+2.c — Scheduler Finished-phase ERROR Fix: Close-out

**Status:** PASS — `step illegal in Finished phase` ERROR eliminated; bug-surface integration test (`test_scheduler_actor_max_tokens_1_no_finished_phase_error`) PASSES with counter == 0; scheduler.rs fail-fast semantics preserved (unit test asserts `step(Finished)` still returns Err); full Rust + Python gates clean.

**Date:** 2026-05-25.
**Branch:** `ironmlx-p5h+2-c-scheduler-finished-fix`.
**Implementation commit:** the T3 commit containing scheduler_actor.rs + scheduler.rs tests + integration test + protocol driver guard + close-out doc.

**Sources:**
- Spec: `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix-design.md`
- Plan: `docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix.md`
- Codex review: `reports/p5h-scheduler-bug-fix-codex-review.md` § 9 (gitignored)
- Predecessor: `docs/p5h+2-b-close-out.md` (P5h+2.b T5F where root cause was identified)

## § 1 Acceptance per spec § 7 — ALL PASS

| # | Criterion | Verdict |
|---|---|---|
| 1 | Bug surface eliminated (counter == 0 on 3× max_new_tokens=1 smoke) | ✓ PASS |
| 2 | Scheduler fail-fast preserved (step(Finished) still Err) | ✓ PASS |
| 3 | No regression (full cargo test PASSES) | ✓ PASS |
| 4 | Rust gate (fmt + nightly fmt --check + clippy + build) | ✓ CLEAN |
| 5 | Python gate (ruff + pytest) | ✓ CLEAN (139 PASS) |
| 6 | Driver guard active (--allow-server-errors opt-in) | ✓ |
| 7 | Close-out doc + memory + commit per `[feedback-*]` | ✓ |

## § 2 What landed

- `ironmlx/src/core/server/scheduler_actor.rs`: new `RollingControl` enum + `finalize_finished_batch_if_any` helper + `drive_empty_scheduler_handoff` helper (lifted from existing duplicated empty-batch handoff block); rolling-loop top hook + outer-loop top defensive hook; `p5h-profile` `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` counter.
- `ironmlx/src/core/scheduler.rs::tests`: 2 new unit tests lock fail-fast semantic.
- `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs`: new actor integration test (3× max_new_tokens=1 → counter == 0).
- `tools/p5h_2b_protocol_experiment.py`: `check_no_scheduler_errors` guard + `--allow-server-errors` CLI flag.

## § 3 Mechanism summary

P5h+2.b root cause (per `[project-p5h-2b-findings]`): `prefill_admitted` leaves `phase=Finished` for `max_tokens=1` workload (line 1247-1250); rolling loop's biased `tokio::select!` falls through to `RollingEvent::Step` when cmd_rx empty; `sched.step()` rejects via phase guard at line 1286; actor logs ERROR + `evict_all` per request → 1116 ERROR/cell in P5h+2.b T4.3.

P5h+2.c fix: pre-event handoff at rolling-loop top — when `phase == Finished` after previous prefill, call `drive_empty_scheduler_handoff`, which first runs `finalize_finished_batch_if_any` (evicts batch + clears event_txs) and then handles queued admits / try_recv / break / return. The biased select never sees a `Finished` state.

Scheduler core semantics preserved: `step_inner` phase guard untouched at scheduler.rs:1286; `step(Phase::Finished)` still returns Err; unit test locks this.

## § 4 P5h+2.b re-attempt readiness

P5h+2.b T4 acceptance sweep can now be re-run with this fix. T0-T3 infrastructure (protocol_experiment.py with --allow-server-errors default off → strict precondition, lifecycle harness, multi_repeat aggregator, pp_tps_envelope tool) is reusable as-is. Expected outcome: 0 `step illegal in <phase>` ERROR per cell + envelope re-measurement.

## § 5 Memory update

Extends MEMORY.md with `project-p5h-2c-findings` entry pointing at `project_p5h_2c_findings.md`. Phase 0 + P5h+2.b memory entries remain unchanged (this fix doesn't backfill P5h+2.b/Phase 0; that's a separate re-run task).

## § 6 References

- Spec: `docs/superpowers/specs/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix-design.md`
- Plan: `docs/superpowers/plans/2026-05-25-ironmlx-p5h+2-c-scheduler-finished-fix.md`
- Codex review: `reports/p5h-scheduler-bug-fix-codex-review.md` § 9
- Predecessor: `docs/p5h+2-b-close-out.md`
- Memory: `[project-p5h-2c-findings]` (new), `[project-p5h-2b-findings]` (root cause source), `[project-p5i-c-phase-0-findings]` (Phase 0 still pending P5h+2.b re-run)
```

- [ ] **Step 4.4: Write memory file**

```bash
cat > /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2c_findings.md << 'MEM'
---
name: project-p5h-2c-findings
description: P5h+2.c scheduler Finished-phase ERROR fix — closed PASS 2026-05-25; actor-side pre-event finalization + empty-handoff helper; scheduler.rs fail-fast preserved; bug-surface integration test proves 0 `step illegal in Finished phase` on 3× max_new_tokens=1; unblocks P5h+2.b re-attempt
metadata:
  type: project
---

P5h+2.c closed 2026-05-25 as PASS.

**Fix shape**:
- `ironmlx/src/core/server/scheduler_actor.rs`: new `RollingControl` enum + `finalize_finished_batch_if_any` + `drive_empty_scheduler_handoff` (lifted from existing duplicated empty-batch handoff at line 435-604); pre-event finalize at rolling-loop top + defensive at outer-loop top; `p5h-profile` counter `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` for regression-test verification.
- `ironmlx/src/core/scheduler.rs`: core scheduler logic unchanged; 2 new unit tests lock `max_new_tokens=1 → Phase::Finished` AND `step(Phase::Finished)` still Err (fail-fast preserved per Codex Q1).
- `ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs`: new actor integration test sends 3× `max_new_tokens=1` admit cmds; asserts counter == 0.
- `tools/p5h_2b_protocol_experiment.py`: `check_no_scheduler_errors` post-cell guard + `--allow-server-errors` CLI flag (default off → strict acceptance precondition per Codex Q4).

**Why fix at actor not scheduler**: per Codex round-3 Q1 + spec § 3.3, keep scheduler.rs fail-fast discipline. Bug is rolling-loop control flow (biased `tokio::select!` doesn't check phase before issuing Step). Fix in actor.

**Key constraint Codex enforced (Q6)**: pre-event finalization MUST run BEFORE the biased select; if only Step branch is guarded, biased select might pick Admit and the actor would call `admit_mid_begin()` in `Phase::Finished` (separate phase-guard error). Spec § 3.1 hard binding.

**Why empty-handoff helper**: `drive_empty_scheduler_handoff` extracted from existing in-rolling-loop empty-batch transition (line 435-604). Same logic now reused from (a) pre-event Finished-batch finalization at rolling-loop top + (b) post-step active_count==0 site. Avoids divergent copies. Returns `RollingControl` enum (ContinueRolling/BreakRolling/ContinueOuter/ReturnActor) to encode caller's loop control.

**Acceptance counter approach**: `p5h-profile` atomic `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` added to the existing `Err` branch of the Step handler. Default release builds are untouched; profiled builds pay only on the Step error branch. Integration test resets counter at start, runs 3× max_new_tokens=1 admit cmds, asserts counter == 0 at end. More deterministic than tracing capture which is brittle across test harness setups.

**P5h+2.b re-attempt enabled**: with `step illegal in <phase>` ERROR eliminated AND driver guard active, P5h+2.b T4 acceptance sweep can re-run. T0-T3 infrastructure (protocol_experiment.py + capture harness + multi_repeat + pp_tps_envelope) is reusable as-is. Whether envelope ≤ ±2% target is reachable post-fix → separate empirical question (this fix removes the dominant ERROR-path overhead; remaining variance may or may not still exceed target).

**Phase 0 backfill**: NOT done here. P5h+2.b re-run is a separate follow-up; Phase 0 backfill triggers on that re-run, not this fix.

**Wall**: ~3 hr (T0 2 + T1 0.5 + T2 1 + T3 0.5).

Links: [[project-p5h-2b-findings]] (root cause source); [[project-p5i-c-phase-0-findings]] (Phase 0 still pending P5h+2.b re-run); [[project-p5h-findings]] (P5h+1 scheduler architecture context).
MEM
```

Then update `MEMORY.md`:

```bash
# Add new entry after project-p5h-2b-findings line
sed -i.bak '/project_p5h_2b_findings.md/a\
- [P5h+2.c scheduler Finished-phase ERROR fix PASS](project_p5h_2c_findings.md) — actor-side pre-event finalization + drive_empty_scheduler_handoff helper; scheduler.rs fail-fast preserved; bug-surface integration test 3x max_new_tokens=1 counter==0; unblocks P5h+2.b re-attempt
' /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md
rm /Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/MEMORY.md.bak
```

- [ ] **Step 4.5: Append T3 section to bench log**

```bash
mkdir -p reports
cat >> reports/p5h+2-c-bench-log.md << BENCHLOG_T3

# P5h+2.c T3 — close-out PASS

**Date**: 2026-05-25
**Wall**: ~3 hr total
**Commit record**:
- T3 single commit is recorded by `git log --oneline -1` after Step 4.6.

**Acceptance**:
- Bug-surface integration test PASSES (counter == 0)
- Scheduler unit tests PASS (fail-fast preserved)
- Full cargo test PASS (no regression)
- Rust gate clean (fmt + nightly fmt --check + clippy 0 warnings + build)
- Python ruff + pytest clean (139 PASS aggregator no-regression)

**P5h+2.b re-attempt next** (separate follow-up task).
BENCHLOG_T3
```

- [ ] **Step 4.6: SINGLE T3 commit**

```bash
cd /Users/xin/workspace/ironmlx-backend
git status --short  # verify all expected files modified/new
git add ironmlx/src/core/server/scheduler_actor.rs \
        ironmlx/src/core/scheduler.rs \
        ironmlx/tests/p5h_2c_scheduler_finished_smoke.rs \
        tools/p5h_2b_protocol_experiment.py \
        docs/p5h+2-c-close-out.md

git commit -m "$(cat <<'EOF'
fix(p5h+2-c): scheduler Finished-phase ERROR eliminated via actor-side finalization

Root cause (per P5h+2.b T5F close-out + Codex round-3 discovery):
prefill_admitted leaves phase=Finished for max_tokens=1 workload at
scheduler.rs:1247-1250 (first sampled token = last token; any_unfinished
false). Rolling loop's biased tokio::select! at scheduler_actor.rs:337
falls through to RollingEvent::Step when cmd_rx empty. sched.step()
rejects Finished via phase guard at scheduler.rs:1286. Actor logs ERROR
+ evict_all + continues outer loop. Per-request cycle → 1116 ERROR/cell
in P5h+2.b T4.3 acceptance → envelope > ±2% gate.

Fix (Codex-approved actor-side finalization, NOT scheduler core change):
- New `RollingControl` enum encoding caller loop control (Continue
  Rolling / BreakRolling / ContinueOuter / ReturnActor).
- New `finalize_finished_batch_if_any` private helper: evict_all +
  clear event_txs when phase == Finished; no-op otherwise.
- New `drive_empty_scheduler_handoff` private helper: lifts existing
  empty-batch handoff (scheduler_actor.rs:435-604) for reuse from
  (a) pre-event Finished finalization at rolling-loop top and (b)
  post-step active_count==0 site. Same logic, single source.
- Rolling-loop top hook: if phase == Finished, handoff helper finalizes
  first, then returns RollingControl (Codex Q6 hard constraint: must run BEFORE
  biased select to prevent admit_mid_begin in Finished phase).
- Outer-loop top defensive hook: same finalize before next admit.
- p5h-profile `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` counter for
  deterministic regression-test verification from the integration test.

Scheduler core preserved (Codex Q1): step_inner phase guard at line
1286 unchanged; `step(Phase::Finished)` still returns Err. Unit test
locks both invariants (max_new_tokens=1 → Phase::Finished post-prefill;
step(Finished) still Err).

Bug-surface integration test (ironmlx/tests/p5h_2c_scheduler_finished
_smoke.rs): spawn actor, send 3 sequential max_new_tokens=1 admit cmds,
assert counter == 0 at end. PASSES.

Driver-side acceptance precondition (Codex Q4): tools/p5h_2b_protocol
_experiment.py:check_no_scheduler_errors aborts sweep on `step illegal
in <phase>` ERROR detection; --allow-server-errors opt-in for
diagnostic experiments.

P5h+2.b T4 acceptance re-attempt unblocked. T0-T3 infrastructure
(protocol_experiment + capture harness + multi_repeat + pp_tps_
envelope) reusable as-is. Whether envelope ≤ ±2% reachable post-fix
is a separate empirical question (this fix eliminates dominant
ERROR-path overhead; residual variance may or may not still exceed
target).

Phase 0 backfill NOT done here; triggers on P5h+2.b re-run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

T3_SHA=$(git rev-parse --short HEAD)
echo "T3 SHA: $T3_SHA"
git log --oneline -3
```

- [ ] **Step 4.7: Post-commit sanity**

```bash
T3_SHA=$(git rev-parse --short HEAD)
echo "T3 SHA: $T3_SHA"
git status --short
rg '<[[:alnum:]_]+_sha>' docs/p5h+2-c-close-out.md reports/p5h+2-c-bench-log.md || true
rg 'todo[!]|unimplemented[!]' ironmlx/src/core/server/scheduler_actor.rs || true
```

Expected: `git status --short` has no unexpected tracked changes; the SHA-placeholder scan prints nothing; the Rust stub-macro scan prints nothing.

P5h+2.c closed PASS.

---

## Self-Review checklist (run before handoff per `[feedback-self-review-before-handoff]`)

1. **Spec coverage:**
   - Spec § 2 Goal #1 (eliminate ERROR) → Task 1 Steps 1.2-1.6 (helpers + hooks); Task 3 (integration test verifies)
   - Spec § 2 Goal #2 (preserve fail-fast) → Task 1 explicitly does NOT touch scheduler.rs core; Task 2 (unit test locks)
   - Spec § 2 Goal #3 (regression test) → Task 3
   - Spec § 2 Goal #4 (driver guard) → Task 1 Step 1.8
   - Spec § 2 Goal #5 (unblock P5h+2.b re-attempt) → handled by Goals 1+4 landing; doc'd in T3 close-out
   - Spec § 3.1 pre-event finalization → Task 1 Step 1.5 (rolling-loop-top hook BEFORE tokio::select)
   - Spec § 3.2 three prefill paths converge → Task 1 Step 1.3+1.4 (drive_empty_scheduler_handoff is single handler)
   - Spec § 3.3 fail-fast preserved → Task 2 Step 2.4 (test asserts)
   - Spec § 4.2.1-4.2.6 components → Task 1 Steps 1.2-1.8 1:1 mapping
   - Spec § 5.1-5.4 tasks → Tasks 1-4 1:1 mapping
   - Spec § 7 acceptance criteria → Task 4 Step 4.1-4.6 verifies + commits

2. **Placeholder scan:** No SHA placeholder is written into close-out, memory, or bench log. The helper extraction has a fixed signature plus a mandatory mechanical move from existing code, not a committed stub. Unit and integration tests now include concrete fake-model / real-fixture code and do not depend on nonexistent defaults.

3. **Type consistency:**
   - `RollingControl` enum (Task 1 Step 1.2) — 4 variants (ContinueRolling/BreakRolling/ContinueOuter/ReturnActor) used consistently in Steps 1.4, 1.5
   - `finalize_finished_batch_if_any<M: Model>` signature consistent across Steps 1.2, 1.5, 1.6
   - `drive_empty_scheduler_handoff<M>` signature consistent across Steps 1.3, 1.4, 1.5 with `M: Model + DenseVlMethods + Send + 'static` and `queue_depth_peak: Arc<AtomicUsize>`
   - `STEP_ILLEGAL_FINISHED_PHASE_HIT_COUNT` static name and `p5h-profile` gating consistent between Step 3.2 (definition) + Step 3.3 (test usage)
   - `check_no_scheduler_errors` Python fn signature consistent between Step 1.8 definition + Step 1.8 hook
   - `--allow-server-errors` CLI flag consistent across Step 1.8 + Step 4.3 close-out doc

No further inline fixes required. Plan ready for execution review per `[feedback-review-spec-before-commit]`.
