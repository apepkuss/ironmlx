# B1-p2.3c-3 — Continuous batching (mid-batch admit/evict + driver_loop rolling decode)

**Date:** 2026-05-15
**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3c-2 head `d27aced`)
**Predecessor sub-phase:** B1-p2.3c-2 — Per-row decode mask activation (closed at `d27aced`). 3c-2 wired `build_per_row_decode_mask` into `Scheduler::step` so SDPA correctly masks stale K/V cells under ragged offsets. The 4-stage admission window + lockstep evict_all driver_loop from 3b-3 is preserved.
**Sibling sub-phase:** **3c+** — chunked batched prefill (separate spec, depends on 3c-3's `admit_mid` path)
**Successor sub-phases:** 3d (admission queue + preemption + config exposure), 3e (per-row sampler tuning), B1-p2.4 (VL B>1)

---

## §1 Goals

1. Lift the "all rows finish together → `evict_all` → next batch" batch boundary. Allow `Scheduler::admit` and `Scheduler::evict` (single-row) during `Phase::Decoding`. Finished rows free their slot via a new `gc_finished_rows` method; new admits can reuse the slot mid-decode.

2. Add a new `Scheduler::admit_mid` method that takes a `GenerateRequest`, finds a free slot, runs an isolated B=1 prefill into a fresh temporary cache (GenerationStream-equivalent path), then *adopts* the prefilled row into the main cache via per-layer slice copies. Returns the first generated token. Mid-batch admit synchronously stalls the rolling decode loop for `~B=1_prefill_per_token_time × L_new` (typical 200ms-2s); chunked prefill in 3c+ reduces this further.

3. Add `KVCache::adopt_row_from(src, dst_row, src_row)` and `GatedDeltaCache::adopt_row_from(src, dst_row, src_row)` — slice-copy a single row's cached state from one cache instance to another. Encapsulates the temp-to-main cache transfer used by `admit_mid`.

4. Refactor `SchedulerActor::driver_loop` from the 4-stage Idle→Admitting→Decoding→Finished sequence to a rolling decode loop:
   - Outer Idle: block waiting for first admit (3b-3 behavior, unchanged)
   - 3b-3 admission window for initial batch (unchanged)
   - `prefill_admitted` for first batch (unchanged)
   - **Inner rolling decode:** `biased tokio::select! { cmd_rx.recv() | step_default }` per iteration. Mid-admits trigger `admit_mid` + synchronous prefill; default branch runs `Scheduler::step` + `gc_finished_rows`. Loop exits when `active_count == 0` AND `cmd_rx` has no pending command.
   - Return to outer Idle (with `evict_all` to reset cache + Phase).

5. Preserve all 12 existing regression suites (P6.3 / P6.6 / P6.7 / B1-p2.1 / B1-p2.2 / 3b-1 / 3b-2 / 3b-3 / 3b-4 / 3c-1 / 3c-2) bit-id-unchanged.

## §2 Non-goals

- **Chunked prefill.** Synchronous B=1 prefill accepted as 3c-3 scope limit. 3c+ ships chunked variant that interleaves chunks with decode steps.
- **Admission queue (3d).** When `b_max` slots are saturated, `Scheduler::admit_mid` returns `Err("scheduler full")` and `driver_loop` forwards the Err back to the caller. The HTTP layer surfaces 429/503 to clients. 3d adds an in-driver_loop queue + fair scheduling.
- **Preemption.** No row is ever kicked out mid-decode to make room for a new admit. 3e+ adds priority-based preemption.
- **Phase enum changes.** 4-state enum (Idle / Admitting / Decoding / Finished) is preserved; transitions are relaxed (Decoding↔Decoding with mid-admit / mid-evict allowed).
- **Cross-task prefill.** Synchronous in `driver_loop`. `Qwen35Model` is `!Send` (sampler holds `Cell<Array>`); cross-task async prefill needs Send-able model + multi-GPU, neither present in ironmlx. Defer to 3d+ if multi-GPU is ever attempted.
- **VL handling in mid-batch admit.** Same VL→GS fallback policy as 3b series (`use_scheduler` check in OpenAI/Anthropic handlers excludes VL requests). Mid-batch VL admit unsupported.

## §3 Background

### 3.1 What 3b-3 / 3c-1 / 3c-2 set up

- **3b-3** introduced the admission window: first admit triggers a 5ms `ADMISSION_DEADLINE` during which `driver_loop` packs additional admits into the batch via `tokio::select! { biased; deadline | cmd_rx.recv() }`. After the deadline (or `b_max` saturation), the batch runs through `prefill_admitted → step* → evict_all` synchronously.
- **3c-1** added per-row cache offsets (`KVCache.offsets: Vec<i32>`, `GatedDeltaCache.offsets: Vec<i32>`) and the right-padding migration. The cache write strategy (Strategy A B-loop in `KVCache::update_and_fetch_on`) is **per-row by construction**: `per_row_lens[i] == 0` skips row `i`'s K/V write entirely.
- **3c-2** wired `build_per_row_decode_mask` into `Scheduler::step` so decode-time SDPA masks stale K/V positions when row offsets diverge.

### 3.2 What's missing for continuous batching

Three structural barriers in 3c-2 prevent mid-batch admit:

1. **`Scheduler::admit` rejects `Phase::Decoding`** (`scheduler.rs:183`): `"scheduler in {:?} phase: cannot admit; call evict_all first"`. This is a deliberate batch-boundary guard from 3a.
2. **`Scheduler::evict` rejects `Phase::Decoding`** (`scheduler.rs:228`): single-row evict only works in Idle/Admitting/Finished. Currently `evict_all` is the only way to free slots during Decoding (drains all rows).
3. **No per-row cache transfer mechanism.** To admit a new request into a freed slot mid-decode without disturbing other active rows, we need to load that slot's cache state with the new request's prefilled K/V (and SSM state for GatedDelta layers). Existing `cache.reset()` is whole-cache; existing `update_and_fetch_on` writes into the position determined by the current `offsets[i]`. Both are insufficient. The clean primitive is "copy one row of cache state from a source cache to a destination slot" — `adopt_row_from`.

3c-3 lifts all three.

### 3.3 Why synchronous B=1 prefill into a temporary cache is the right design

The naive alternative — running `model.batched_prefill` with the full B=b_max main cache and `per_row_lens = [0,...,L_new,...,0]` — has three serious problems:

1. **Wasted compute.** The forward runs the full B-batch through embeddings, every decoder layer, every attention/MLP/RMSNorm. Other rows produce garbage outputs (from input_ids=0 → non-zero embeddings → garbage Q/K/V → masked SDPA → discarded), but GPU time scales as `~B × B=1_prefill_time`. At b_max=4, this is 4× the necessary work per mid-admit.

2. **Variable mask shape.** SDPA gets K shape `[B, n_kv_heads, max(post_write_offsets), head_dim]` where `max(post_write_offsets) = max(other_active_rows_offsets, L_new)`. Attention mask must be `[B, 1, L_new, max_off]`, not `[B, 1, L_new, L_new]`. A new variable-shape mask helper would need to handle stale K from masked rows correctly without polluting the active row's softmax.

3. **GatedDeltaNet recurrent_state corruption risk.** `GatedDeltaNet::forward_on` writes the full `[B, Hv, Dv, Dk]` recurrent_state via `update_recurrent`. For masked rows (`per_row_lens[i] = 0`), the kernel still consumes their state_in (from previous decode step) and produces state_out — but whether the kernel preserves `state_out[i] == state_in[i]` when its per-token mask is all-false depends on kernel internals. Risk of corrupting other active rows' SSM state during admit_mid prefill.

The **standalone B=1 temp cache + adopt** approach sidesteps all three:

1. **Minimal compute:** B=1 prefill runs only on the new request's tokens. ~B=1_prefill_time wall clock, NOT B × B=1_prefill_time.

2. **No mask shape complications:** B=1 prefill uses the existing `build_batch_attention_mask([prompt_len], prompt_len, dtype)` — same path GenerationStream uses today.

3. **Zero interference with other rows:** the temp cache is allocated fresh by `model.make_cache(batch=1, cap, dtype)`. Other active rows' main_cache entries are untouched during prefill. Adoption is a slice_update on the K/V/state buffers; main_cache offsets for other rows are not modified.

**Copy cost is sub-microsecond.** Per-layer slice_update of B=1 cache state into B-row destination: 32 layers × (K + V + conv_state + recurrent_state) ≈ ~1.6 MB total at typical Qwen3.5-4B dimensions. Apple Silicon unified memory bandwidth ~400 GB/s → ~4 μs. Compared to ~200ms-2s prefill compute, copy is negligible.

### 3.4 Why synchronous prefill is the only viable async-vs-sync option

ironmlx's serving architecture has three constraints that make async prefill not just unnecessary but unimplementable as currently structured:

1. **Single GPU (Apple Silicon Metal):** the M-series GPU is a unified compute unit with no kernel-level concurrency for prefill vs decode. Whether prefill runs in the driver_loop or in a spawned task, GPU time is the same.
2. **Model `Mutex`:** `Arc<Mutex<Qwen35Model>>` is held by `run_batch_once` for the duration of a forward pass. Async prefill would need to acquire the same lock from a separate task, gaining nothing — the lock serializes regardless of caller identity.
3. **`Qwen35Model: !Send`:** the model's sampler holds a `Cell<Array>` for per-step state, making the model not safely movable across task boundaries. Async would require redesigning the sampler.

The real mitigation for prefill stall is chunked prefill (3c+), which slices long prefills into multiple smaller chunks interleaved with decode steps. Chunked prefill needs `admit_mid` as its substrate.

### 3.5 Industry reference (informs design)

- **vLLM:** PagedAttention + per-request page tables. Mid-batch admit is straightforward because each request owns its pages; no cache row reuse needed (new request just allocates fresh pages). Chunked prefill via separate `--chunked-prefill` flag. Backpressure via `--max-num-seqs`.
- **SGLang:** RadixAttention + per-request offset. Mid-batch admit + chunked prefill both standard.
- **TGI (Hugging Face):** Dense KV cache with per-request "block" tracking. Mid-batch admit via slot reuse + zero-init the slot's KV.
- **llama.cpp server:** Static slot allocation. No mid-batch admit; clients queue per slot externally.

**ironmlx 3c-3 positioning:** dense KV cache + per-row `Vec<i32>` offsets + temp-cache adoption → mid-batch admit. Closer to TGI than vLLM/SGLang. Trades the page-pool complexity (deferred to B1-p2.5 production hardening or later) for a simpler same-shape cache that already works in 3a-3c-2. Chunked prefill (3c+) brings ironmlx toward vLLM-class prefill scheduling without requiring the page-pool rewrite.

## §4 Architecture

### 4.1 driver_loop rolling decode

```rust
fn driver_loop(model, b_max, mut cmd_rx, counters...) {
    let mut sched = Scheduler::new(b_max);
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let rt = tokio::runtime::Handle::current();

    'outer: loop {
        // ===== Idle: block waiting for first admit (or shutdown) =====
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
            return;
        };
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        if sched.active_count() == 0 {
            continue 'outer; // first admit failed; loop back
        }

        // ===== Admitting: drain initial admission window (3b-3 behavior) =====
        if sched.active_count() < b_max {
            rt.block_on(drain_window(
                &mut cmd_rx, &mut sched, &mut event_txs,
                &admit_count, &saturate_triggered, b_max,
                ADMISSION_DEADLINE,
            ));
        }

        // ===== First-batch prefill =====
        batch_count.fetch_add(1, Ordering::Relaxed);
        let prefill_result = {
            let model_lock = model.blocking_lock();
            sched.prefill_admitted(&model_lock)
        };
        match prefill_result {
            Ok(prefill_events) => {
                for ev in prefill_events {
                    route_event(ev, &event_txs);
                }
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] prefill error: {e:?}");
                let _ = sched.evict_all();
                event_txs.clear();
                continue 'outer;
            }
        }

        // ===== Rolling decode loop =====
        'rolling: loop {
            // Q2=A: biased select between cmd_rx and step.
            // `futures::future::ready(())` is always-ready so step wins
            // when cmd_rx is empty.
            let evt: RollingEvent = rt.block_on(async {
                tokio::select! {
                    biased;
                    maybe_cmd = cmd_rx.recv() => match maybe_cmd {
                        Some(cmd) => RollingEvent::Admit(cmd),
                        None => RollingEvent::Shutdown,
                    },
                    () = futures::future::ready(()) => RollingEvent::Step,
                }
            });

            match evt {
                RollingEvent::Shutdown => {
                    // cmd_rx closed. Drain finished rows so any in-flight
                    // handler gets EOF, then return from driver_loop.
                    event_txs.clear();
                    return;
                }
                RollingEvent::Admit(cmd) => {
                    handle_admit_mid(cmd, &mut sched, &mut event_txs, &admit_count, &model);
                }
                RollingEvent::Step => {
                    let step_result = {
                        let model_lock = model.blocking_lock();
                        sched.step(&model_lock)
                    };
                    match step_result {
                        Ok(events) => {
                            for ev in events {
                                route_event(ev, &event_txs);
                            }
                            sched.gc_finished_rows(&mut event_txs);
                        }
                        Err(e) => {
                            tracing::error!("[SchedulerActor] step error: {e:?}");
                            let _ = sched.evict_all();
                            event_txs.clear();
                            continue 'outer;
                        }
                    }
                }
            }

            // Exit rolling loop when no active rows and cmd_rx is empty.
            if sched.active_count() == 0 {
                match cmd_rx.try_recv() {
                    Ok(cmd) => {
                        handle_admit(cmd, &mut sched, &mut event_txs, &admit_count);
                        if sched.active_count() == 0 {
                            break 'rolling; // admit failed
                        }
                        // Run an initial-window drain for the new batch.
                        if sched.active_count() < b_max {
                            rt.block_on(drain_window(
                                &mut cmd_rx, &mut sched, &mut event_txs,
                                &admit_count, &saturate_triggered, b_max,
                                ADMISSION_DEADLINE,
                            ));
                        }
                        batch_count.fetch_add(1, Ordering::Relaxed);
                        let prefill_result = {
                            let model_lock = model.blocking_lock();
                            sched.prefill_admitted(&model_lock)
                        };
                        match prefill_result {
                            Ok(events) => {
                                for ev in events { route_event(ev, &event_txs); }
                            }
                            Err(e) => {
                                tracing::error!("[SchedulerActor] re-prefill error: {e:?}");
                                let _ = sched.evict_all();
                                event_txs.clear();
                                break 'rolling;
                            }
                        }
                        continue 'rolling;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        break 'rolling;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        event_txs.clear();
                        return;
                    }
                }
            }
        }

        // After rolling loop: reset cache + Phase. evict_all is now a
        // no-op for slot management (gc_finished_rows already cleared
        // them), but it resets the cache buffer for the next outer batch.
        let _ = sched.evict_all();
        event_txs.clear();
    }
}

enum RollingEvent {
    Admit(SchedulerCommand),
    Step,
    Shutdown,
}

fn handle_admit_mid(
    cmd: SchedulerCommand,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<Qwen35Model>>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let model_lock = model.blocking_lock();
    match sched.admit_mid(request, &model_lock) {
        Ok((id, prefill_event)) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            event_txs.insert(id, event_tx.clone());
            if reply_tx.send(Ok(AdmitReply {
                request_id: id,
                event_rx,
            })).is_err() {
                // Caller dropped reply_rx; evict the orphan slot.
                let _ = sched.evict(id);
                event_txs.remove(&id);
                return;
            }
            // Route the prefill event (first generated token).
            route_event(prefill_event, event_txs);
        }
        Err(e) => {
            let _ = reply_tx.send(Err(e));
        }
    }
}
```

**Key design points:**

- **Biased select (Q2=A):** `tokio::select!` with `biased;` and `cmd_rx.recv()` ahead of `futures::future::ready(())`. When `cmd_rx` has a pending admit, the Admit branch wins; otherwise the always-ready Step branch fires.
- **Model lock per iteration:** lock is acquired inside the match arms for Step / Admit, not held across the biased select. Releasing between iterations allows `cmd_rx.recv()` to make progress.
- **`gc_finished_rows`:** runs after every successful step. Drops `event_tx` for finished rows → handler sees EOF. Slot becomes `None`.
- **Exit condition:** `active_count == 0` triggers a `try_recv` poll. Empty queue → break to outer. Pending command → process and continue rolling.
- **Shutdown:** `cmd_rx.recv()` returning `None` (channel closed) → clear event_txs → return from driver_loop. Any in-flight handlers see EOF.

### 4.2 `Scheduler::admit` Phase guard relaxation

Before (3c-2):

```rust
pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId> {
    self.ensure_not_poisoned()?;
    match self.phase {
        Phase::Idle | Phase::Admitting => {}
        Phase::Decoding | Phase::Finished => {
            return Err(anyhow!(
                "scheduler in {:?} phase: cannot admit; call evict_all first",
                self.phase
            ));
        }
    }
    // ... rest: find empty slot, create RequestState, set Phase::Admitting
}
```

After (3c-3):

```rust
pub fn admit(&mut self, req: GenerateRequest) -> Result<RequestId> {
    self.ensure_not_poisoned()?;
    if self.phase == Phase::Finished {
        return Err(anyhow!(
            "scheduler in Finished phase: call evict_all first"
        ));
    }
    let row_idx = self.slots.iter().position(|s| s.is_none())
        .ok_or_else(|| anyhow!(
            "scheduler full: no row available (b_max={})", self.b_max
        ))?;
    // ... rest unchanged
    if self.phase == Phase::Idle {
        self.phase = Phase::Admitting;
    }
    // Decoding stays Decoding (mid-batch admit handled by admit_mid)
    Ok(id)
}
```

The slot insertion logic is unchanged. Phase transitions:

- Idle → Admitting (first admit, unchanged)
- Admitting → Admitting (subsequent admits during window)
- **Decoding → Decoding (NEW: but admit alone does not prefill — admit_mid handles prefill + adoption)**
- Finished → Err

### 4.3 `Scheduler::evict` Phase guard relaxation

```rust
pub fn evict(&mut self, id: RequestId) -> Result<()> {
    self.ensure_not_poisoned()?;
    // 3c-3: evict allowed in all phases. Slot is cleared; main cache
    // state for this row stays in place (no resource leak; next
    // admit_mid into this slot overwrites via adopt_row_from).
    let row_idx = self.slots.iter()
        .position(|s| matches!(s, Some(r) if r.id == id))
        .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
    self.slots[row_idx] = None;
    // Phase transitions:
    //   Admitting → Idle (active_count == 0; unchanged)
    //   Decoding → Finished (active_count == 0; NEW)
    if self.active_count() == 0 {
        if self.phase == Phase::Admitting {
            self.phase = Phase::Idle;
        } else if self.phase == Phase::Decoding {
            self.phase = Phase::Finished;
        }
    }
    Ok(())
}
```

### 4.4 `Scheduler::gc_finished_rows` (NEW)

```rust
/// Sweep finished rows: clear their slot, drop their event channel, and
/// return the evicted IDs. Cache buffer entries for evicted slots stay
/// in place — a subsequent `admit_mid` into the same slot overwrites
/// via `adopt_row_from`.
///
/// Phase transition: Decoding → Finished if `active_count == 0` after gc.
///
/// Called by `driver_loop` after every `step`.
pub fn gc_finished_rows<S>(
    &mut self,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<S>>,
) -> Vec<RequestId> {
    let mut evicted: Vec<RequestId> = Vec::new();
    for slot in self.slots.iter_mut() {
        if let Some(state) = slot.as_ref() {
            if state.finished {
                let id = state.id;
                event_txs.remove(&id);
                evicted.push(id);
                *slot = None;
            }
        }
    }
    if self.phase == Phase::Decoding && self.active_count() == 0 {
        self.phase = Phase::Finished;
    }
    evicted
}
```

**Note on generic `S`:** `gc_finished_rows` is called from `driver_loop` with `S = StepEvent`. The generic avoids coupling Scheduler to a specific event type if future code grows additional channels (e.g., metrics).

### 4.5 `Scheduler::step` Phase transition adjustment

Before (3c-2): step internally transitions to `Phase::Finished` when every active slot has `finished == true`.

After (3c-3): step never transitions phase. Phase transition is delegated to `gc_finished_rows` (when slots are emptied) and `evict_all` (when cache is reset). Step still:

- Marks `state.finished = true` and sets `state.finish_reason`
- Returns `StepEvent` with `finish_reason: Some(_)` for transitioning rows

This separation keeps `step` purely about token generation; phase management lives with the slot lifecycle methods.

### 4.6 `Scheduler::admit_mid` (NEW) — standalone B=1 prefill + adoption

```rust
/// Mid-batch admit + prefill. Caller is `driver_loop` after `cmd_rx`
/// delivers an Admit during the rolling decode loop.
///
/// Architecture: runs prefill in a temporary B=1 cache (the same path
/// GenerationStream uses today), then adopts the prefilled row into the
/// main cache via per-layer slice copies. This avoids:
///   - Wasted compute (no B=b_max forward; only B=1 work for L_new tokens)
///   - Variable-shape mask construction (existing helpers suffice for B=1)
///   - GatedDeltaNet state corruption for other active rows (their main
///     cache entries are not touched during prefill)
///
/// Performance: synchronous; stalls active rows for ~L_new × B=1_prefill
/// _per_token_time. Adoption cost is sub-microsecond (~1.6 MB slice copy
/// over unified memory). 3c+ chunked prefill reduces stall further.
///
/// Returns `(RequestId, StepEvent)` — the assigned request ID and the
/// first generated token's event. Caller registers the event channel
/// using the returned `id`.
pub fn admit_mid(
    &mut self,
    req: GenerateRequest,
    model: &Qwen35Model,
) -> Result<(RequestId, StepEvent)> {
    self.ensure_not_poisoned()?;
    if self.phase != Phase::Decoding {
        return Err(anyhow!(
            "admit_mid illegal in {:?} phase: only Decoding (use admit for Idle/Admitting)",
            self.phase
        ));
    }
    let row_idx = self.slots.iter().position(|s| s.is_none())
        .ok_or_else(|| anyhow!(
            "scheduler full: no row available (b_max={})", self.b_max
        ))?;

    // 1. Insert RequestState via the relaxed admit() path. Phase stays Decoding.
    let id = self.admit(req)?;
    let state_ref = self.slots[row_idx].as_ref().expect("admit inserted");
    let prompt_ids = state_ref.prompt_ids.clone();
    let prompt_len = prompt_ids.len() as i32;
    let cap_for_temp = (prompt_len + state_ref.max_new_tokens as i32).max(prompt_len);

    // 2. Capture KVCache dtype from the main cache (all Full layers share dtype).
    let main_cache = self.cache.as_mut()
        .ok_or_else(|| anyhow!("admit_mid called before prefill_admitted: cache absent"))?;
    let dtype = main_cache.iter().find_map(|c| match c {
        LayerCache::Full(kv) => Some(kv.dtype()),
        _ => None,
    }).unwrap_or(Dtype::Bfloat16);

    // 3. Allocate a fresh B=1 temp cache. Same layer topology as main.
    let mut temp_cache = model.make_cache(1, cap_for_temp, dtype)?;

    // 4. Build B=1 prefill inputs (mirror GenerationStream prefill).
    //    Right-pad single-row: input_ids [1, prompt_len] with prompt tokens.
    let input_ids_data: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
    let input_ids: Array = (&input_ids_data[..], &[1_i32, prompt_len][..]).try_into()?;
    let position_ids = build_position_ids_batched(&[prompt_len], prompt_len)?;
    let attention_mask = build_batch_attention_mask(&[prompt_len], prompt_len, dtype)?;
    let linear_attention_mask = build_batch_linear_mask(&[prompt_len], prompt_len)?;

    // 5. Run B=1 prefill into the temp cache. Returns logits [1, 1, vocab].
    let logits = model.batched_prefill(
        &input_ids,
        &position_ids,
        &attention_mask,
        &linear_attention_mask,
        &[prompt_len],
        Some(&mut temp_cache),
        (),
    )?;

    // 6. Adopt the temp cache's row 0 into main_cache at row_idx.
    //    Per-layer slice copy: K/V for Full layers, conv_state +
    //    recurrent_state for Linear (GatedDelta) layers.
    if main_cache.len() != temp_cache.len() {
        return Err(anyhow!(
            "admit_mid: cache layer count mismatch ({} vs {})",
            main_cache.len(), temp_cache.len()
        ));
    }
    for (main_layer, temp_layer) in main_cache.iter_mut().zip(temp_cache.iter()) {
        match (main_layer, temp_layer) {
            (LayerCache::Full(main_kv), LayerCache::Full(temp_kv)) => {
                main_kv.adopt_row_from(temp_kv, row_idx, 0)?;
            }
            (LayerCache::Linear(main_gd), LayerCache::Linear(temp_gd)) => {
                main_gd.adopt_row_from(temp_gd, row_idx, 0)?;
            }
            _ => return Err(anyhow!(
                "admit_mid: cache layer kind mismatch between main and temp at layer index"
            )),
        }
    }

    // 7. Sample first token from the prefill logits (last position).
    //    Logits shape [1, 1, vocab] -- slice [0, 0, :] reshape [vocab].
    let row_logits = slice_logits_row(&logits, 0)?;
    let state = self.slots[row_idx].as_ref().expect("admit_mid slot");
    let mut history: Vec<u32> = Vec::with_capacity(prompt_ids.len());
    history.extend_from_slice(&prompt_ids);
    let token = state.sampler.sample(&row_logits, &history)?;

    // 8. Update state + check termination.
    let state = self.slots[row_idx].as_mut().expect("admit_mid slot");
    state.generated_tokens.push(token);
    state.real_len += 1;

    if state.stop_token_ids.contains(&token) {
        state.finished = true;
        state.finish_reason = Some("stop");
    } else if state.generated_tokens.len() >= state.max_new_tokens {
        state.finished = true;
        state.finish_reason = Some("length");
    }

    Ok((id, StepEvent {
        id,
        token,
        finish_reason: state.finish_reason,
    }))
}
```

**Helper added to `core/generate.rs`:**

`slice_logits_row(logits: &Array, row_idx: usize) -> Result<Array>` — slices `logits[row_idx, 0, :]` and reshapes to `[vocab]`. Same pattern as `Scheduler::step_inner`'s per-row logit slice (could be extracted from step_inner if not already a helper, or just inlined in admit_mid).

### 4.7 `KVCache::adopt_row_from` (NEW)

```rust
/// Copy a single row's cache state from `src` (a different KVCache
/// instance) into `self` at `dst_row`. The destination slot's K/V at
/// positions [0..src.offsets[src_row]] is overwritten; positions beyond
/// (stale or unallocated) are not touched. `self.offsets[dst_row]` is
/// set to `src.offsets[src_row]`.
///
/// Requires matching n_kv_heads / head_dim / v_head_dim / dtype between
/// src and self. src and self may have different batch sizes (typical
/// usage: src.batch = 1, self.batch = b_max).
///
/// Errors:
///   - shape mismatch (n_kv_heads / head_dim / v_head_dim / dtype)
///   - dst_row >= self.batch
///   - src_row >= src.batch
///   - src.offsets[src_row] > self.cap
pub fn adopt_row_from(
    &mut self,
    src: &KVCache,
    dst_row: usize,
    src_row: usize,
) -> Result<()> {
    if self.n_kv_heads != src.n_kv_heads
        || self.head_dim != src.head_dim
        || self.v_head_dim != src.v_head_dim
        || self.dtype != src.dtype
    {
        anyhow::bail!(
            "KVCache::adopt_row_from: shape/dtype mismatch (self={}/{}/{}/{:?}, src={}/{}/{}/{:?})",
            self.n_kv_heads, self.head_dim, self.v_head_dim, self.dtype,
            src.n_kv_heads, src.head_dim, src.v_head_dim, src.dtype,
        );
    }
    if dst_row >= self.batch as usize {
        anyhow::bail!("dst_row {} >= self.batch {}", dst_row, self.batch);
    }
    if src_row >= src.batch as usize {
        anyhow::bail!("src_row {} >= src.batch {}", src_row, src.batch);
    }
    let src_off = src.offsets[src_row];
    if src_off > self.cap {
        anyhow::bail!("src.offsets[{}] = {} > self.cap {}", src_row, src_off, self.cap);
    }

    if src_off > 0 {
        // Ensure self.keys / values are allocated up to src_off.
        let current_capacity = self.keys.as_ref()
            .map(|a| a.shape().as_slice()[2])
            .unwrap_or(0);
        if src_off > current_capacity {
            let target_capacity =
                ((src_off + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, ())?;
        }

        let src_keys = src.keys.as_ref().ok_or_else(|| anyhow!("src keys unallocated"))?;
        let src_values = src.values.as_ref().ok_or_else(|| anyhow!("src values unallocated"))?;

        // Slice src[src_row, :, 0..src_off, :].
        let k_slice = mlx::ops::indexing::slice_strided(
            src_keys,
            &[src_row as i32, 0, 0, 0][..],
            &[src_row as i32 + 1, self.n_kv_heads, src_off, self.head_dim][..],
            &[1_i32, 1, 1, 1][..],
        )?;
        let v_slice = mlx::ops::indexing::slice_strided(
            src_values,
            &[src_row as i32, 0, 0, 0][..],
            &[src_row as i32 + 1, self.n_kv_heads, src_off, self.v_head_dim][..],
            &[1_i32, 1, 1, 1][..],
        )?;

        // Write into self[dst_row, :, 0..src_off, :].
        let keys_full = self.keys.as_ref().expect("alloc above");
        let values_full = self.values.as_ref().expect("alloc above");
        let new_keys = mlx::ops::indexing::slice_update(
            keys_full,
            &k_slice,
            &[dst_row as i32, 0, 0, 0][..],
            &[dst_row as i32 + 1, self.n_kv_heads, src_off, self.head_dim][..],
            &[1_i32, 1, 1, 1][..],
        )?;
        let new_values = mlx::ops::indexing::slice_update(
            values_full,
            &v_slice,
            &[dst_row as i32, 0, 0, 0][..],
            &[dst_row as i32 + 1, self.n_kv_heads, src_off, self.v_head_dim][..],
            &[1_i32, 1, 1, 1][..],
        )?;
        self.keys = Some(new_keys);
        self.values = Some(new_values);
    }
    self.offsets[dst_row] = src_off;
    Ok(())
}
```

**Why `dtype` field accessor:** the existing `KVCache` already stores `dtype` privately. Adding a `pub fn dtype(&self) -> Dtype` accessor is cheap and lets `adopt_row_from` validate without separate state tracking.

### 4.8 `GatedDeltaCache::adopt_row_from` (NEW)

```rust
/// Copy a single row's full SSM state from `src` into `self` at
/// `dst_row`. The destination's `conv_state[dst_row, :, :]` and
/// `recurrent_state[dst_row, :, :, :]` slabs are overwritten;
/// `self.offsets[dst_row]` is set to `src.offsets[src_row]`.
///
/// Errors:
///   - shape mismatch (kernel_size-1, conv_dim, hv, dv, dk)
///   - dst_row >= self.b
///   - src_row >= src.b
///   - src.offsets[src_row] > self.cap
pub fn adopt_row_from(
    &mut self,
    src: &GatedDeltaCache,
    dst_row: usize,
    src_row: usize,
) -> Result<()> {
    let self_conv_dims = self.conv_state.shape();
    let self_conv_dims = self_conv_dims.as_slice();
    let src_conv_dims = src.conv_state.shape();
    let src_conv_dims = src_conv_dims.as_slice();
    if self_conv_dims[1] != src_conv_dims[1] || self_conv_dims[2] != src_conv_dims[2] {
        anyhow::bail!(
            "GatedDeltaCache::adopt_row_from: conv_state shape mismatch (self [_,{},{}] src [_,{},{}])",
            self_conv_dims[1], self_conv_dims[2],
            src_conv_dims[1], src_conv_dims[2],
        );
    }
    let self_rec_dims = self.recurrent_state.shape();
    let self_rec_dims = self_rec_dims.as_slice();
    let src_rec_dims = src.recurrent_state.shape();
    let src_rec_dims = src_rec_dims.as_slice();
    if self_rec_dims[1] != src_rec_dims[1]
        || self_rec_dims[2] != src_rec_dims[2]
        || self_rec_dims[3] != src_rec_dims[3]
    {
        anyhow::bail!(
            "GatedDeltaCache::adopt_row_from: recurrent_state shape mismatch"
        );
    }
    if dst_row >= self.offsets.len() {
        anyhow::bail!("dst_row {} >= self.B {}", dst_row, self.offsets.len());
    }
    if src_row >= src.offsets.len() {
        anyhow::bail!("src_row {} >= src.B {}", src_row, src.offsets.len());
    }
    let src_off = src.offsets[src_row];
    if src_off > self.cap {
        anyhow::bail!("src.offsets[{}] = {} > self.cap {}", src_row, src_off, self.cap);
    }

    let kernel_minus_one = self_conv_dims[1];
    let conv_dim = self_conv_dims[2];
    let hv = self_rec_dims[1];
    let dv = self_rec_dims[2];
    let dk = self_rec_dims[3];

    // Copy conv_state[src_row, :, :] -> self.conv_state[dst_row, :, :].
    let src_conv_slice = mlx::ops::indexing::slice_strided(
        &src.conv_state,
        &[src_row as i32, 0, 0][..],
        &[src_row as i32 + 1, kernel_minus_one, conv_dim][..],
        &[1_i32, 1, 1][..],
    )?;
    self.conv_state = mlx::ops::indexing::slice_update(
        &self.conv_state,
        &src_conv_slice,
        &[dst_row as i32, 0, 0][..],
        &[dst_row as i32 + 1, kernel_minus_one, conv_dim][..],
        &[1_i32, 1, 1][..],
    )?;

    // Copy recurrent_state[src_row, :, :, :] -> self.recurrent_state[dst_row, :, :, :].
    let src_rec_slice = mlx::ops::indexing::slice_strided(
        &src.recurrent_state,
        &[src_row as i32, 0, 0, 0][..],
        &[src_row as i32 + 1, hv, dv, dk][..],
        &[1_i32, 1, 1, 1][..],
    )?;
    self.recurrent_state = mlx::ops::indexing::slice_update(
        &self.recurrent_state,
        &src_rec_slice,
        &[dst_row as i32, 0, 0, 0][..],
        &[dst_row as i32 + 1, hv, dv, dk][..],
        &[1_i32, 1, 1, 1][..],
    )?;

    self.offsets[dst_row] = src_off;
    Ok(())
}
```

### 4.9 Module surface summary

```text
ironmlx/src/core/cache/kv_cache.rs            — MODIFY (+~70 lines)
  + adopt_row_from(src, dst_row, src_row)
  + dtype() accessor

ironmlx/src/core/cache/gated_delta.rs         — MODIFY (+~70 lines)
  + adopt_row_from(src, dst_row, src_row)

ironmlx/src/core/generate.rs                  — MODIFY (+~15 lines)
  + slice_logits_row helper (factored from existing step_inner pattern;
    or just inlined in admit_mid if extraction is awkward)

ironmlx/src/core/scheduler.rs                 — MODIFY (~150 lines)
  ~ admit Phase guard relaxation (Finished → Err; else allow)
  ~ evict Phase guard relaxation (all phases)
  ~ step Phase transition removed (delegated to gc_finished_rows)
  + admit_mid(req, model) -> Result<(RequestId, StepEvent)>
  + gc_finished_rows(event_txs) -> Vec<RequestId>

ironmlx/src/core/server/scheduler_actor.rs    — MODIFY (~140 lines)
  ~ driver_loop refactor: rolling decode loop with biased select
  + handle_admit_mid helper
  + RollingEvent enum
  ~ run_batch_once removed (logic inlined into driver_loop)

ironmlx/tests/b1_p2_3c_3_continuous_batching.rs — NEW (~320 lines)
  + continuous_batching_mid_decode_admit
  + continuous_batching_full_reject
  + continuous_batching_drains_to_empty
```

No changes to: `nn/*`, `models/*`, `core/server/{openai,anthropic}.rs`, `core/generate.rs::GenerationStream`.

## §5 Tests

### 5.1 KVCache lib unit tests (3 new)

In `kv_cache.rs::tests`:

1. **`kvcache_adopt_row_from_basic`** — src B=1 cache, write `[4]` (single row offset=4, K markers e.g. 7.0); dst B=2 cache (initially empty). `dst.adopt_row_from(&src, dst_row=1, src_row=0)`. Verify:
   - `dst.offsets() == [0, 4]`
   - `dst.keys[1, :, 0..4, :]` matches src.keys[0, :, 0..4, :] (exhaustive elementwise)
   - `dst.keys[0, :, :, :]` unchanged (whatever it was — for fresh dst, zero buffer)

2. **`kvcache_adopt_row_from_shape_mismatch_err`** — src has n_kv_heads=4, dst has n_kv_heads=2 → Err containing `"shape"` or `"mismatch"`.

3. **`kvcache_adopt_row_from_out_of_bounds_err`** — dst B=2; `adopt_row_from(src, dst_row=2, src_row=0)` → Err containing `"dst_row"` and `"self.batch"`.

### 5.2 GatedDeltaCache lib unit tests (2 new)

In `gated_delta.rs::tests`:

1. **`gdcache_adopt_row_from_state_and_offset`** — src B=1 cache, mutate conv_state[0] to all 1.0 and recurrent_state[0] to all 2.0, set offsets[0]=4. dst B=2 cache (fresh zeros). `dst.adopt_row_from(&src, dst_row=1, src_row=0)`. Verify:
   - `dst.offsets() == [0, 4]`
   - `dst.conv_state[1, :, :]` is all 1.0
   - `dst.conv_state[0, :, :]` is all 0.0 (untouched)
   - `dst.recurrent_state[1, :, :, :]` is all 2.0
   - `dst.recurrent_state[0, :, :, :]` is all 0.0 (untouched)

2. **`gdcache_adopt_row_from_out_of_bounds_err`** — dst_row >= self.B → Err.

### 5.3 Integration scenarios (3 new in `tests/b1_p2_3c_3_continuous_batching.rs`)

All `#[ignore]`-gated, drive `SchedulerActor` directly via `cmd_tx` (3c-3's value lives in the actor's rolling decode loop).

1. **`continuous_batching_mid_decode_admit`** (central correctness gate)
   - `spawn_scheduler_actor(model, b_max=2)`.
   - Admit request A (prompt "Hello", max_new=3).
   - Admit request B (prompt "World", max_new=8).
   - Drain events; record per-row token sequences.
   - After A finishes (~3 tokens), admit request C (prompt "Goodbye", max_new=5) — should land in row 0 (A's vacated slot) via admit_mid.
   - Drain C and B's remaining events.
   - Assertions:
     - A produces exactly 3 tokens, finish_reason='length'
     - B produces exactly 8 tokens, finish_reason='length'
     - C produces exactly 5 tokens, finish_reason='length'
     - A's tokens bit-id ≥ 0.95 vs B=1 baseline(prompt="Hello", max_new=3)
     - B's tokens bit-id ≥ 0.95 vs B=1 baseline(prompt="World", max_new=8)
     - C's tokens bit-id ≥ 0.95 vs B=1 baseline(prompt="Goodbye", max_new=5)

2. **`continuous_batching_full_reject`**
   - `spawn_scheduler_actor(model, b_max=2)`.
   - Admit A + B both with max_new=20 (long enough to keep slots occupied).
   - Immediately admit C while A, B are decoding.
   - C's `reply_rx.await` should return `Err("scheduler full")`.
   - Drain A, B normally.

3. **`continuous_batching_drains_to_empty`**
   - `spawn_scheduler_actor(model, b_max=2)`.
   - Admit A, drain to completion. Verify the actor returns to outer Idle (e.g. via `batch_count` observation or admit_count == 1 / batch_count == 1).
   - Admit B 100ms after A finishes. Verify B prefills + completes through the second batch.

### 5.4 Regression sweep updates

- **Existing `b1_p2_3b_2` / `3b-3` / `3b-4`:** must still PASS. The biased-select rolling decode collapses to the pre-3c-3 behavior when no mid-batch admit happens (each step's biased select sees `cmd_rx` empty → step branch fires → finishes batch → outer Idle).
- **Existing `b1_p2_3c_1` / `3c-2`:** must still PASS. Scheduler API surface changes are additive (admit_mid, gc_finished_rows, adopt_row_from); existing methods preserve behavior for non-mid-batch paths.

## §6 Acceptance gates

- All 5 new lib unit tests + 3 new integration scenarios PASS
- All 12 existing regression suites PASS unchanged (token output bit-id identical to pre-3c-3 for non-mid-batch paths)
- `cargo +nightly fmt --check`, `clippy -D warnings`, `cargo build --release -p ironmlx`: clean
- Lib test count: 205 (3c-2) + 5 = **210 lib tests**
- `continuous_batching_mid_decode_admit` per-row bit-id ≥ 0.95 on all 3 rows (A pre-evict, B uninterrupted, C post-admit_mid)

## §7 Estimate

**5-7 working days:**

- Day 1: `KVCache::adopt_row_from` + `KVCache::dtype()` accessor + 3 lib tests
- Day 2: `GatedDeltaCache::adopt_row_from` + 2 lib tests
- Day 3: `Scheduler::admit` + `evict` Phase guard relaxation + `gc_finished_rows` + `step` transition delegation
- Day 4: `Scheduler::admit_mid` integration (the most complex piece — B=1 temp cache management, adoption loop, sampler reuse)
- Day 5: `SchedulerActor::driver_loop` refactor + `handle_admit_mid` helper + `RollingEvent` enum
- Day 6: 3 new integration scenarios + 12-suite regression sweep
- Day 7: Buffer for unexpected issues (e.g., temp cache lifetime issues, sampler clone semantics in admit_mid)

## §8 Compat sunset notes

3c-3 inherits all 5 sunset markers from 3b series + 3c-1:

| Compat | Sunset trigger |
| --- | --- |
| OpenAI VL → GS | B1-p2.4 batched VL |
| OpenAI long-prompt → GS | 3c+ chunked-prefill |
| Anthropic long-prompt → GS | 3c+ chunked-prefill |
| Anthropic image-content → 400 | Future Anthropic VL phase |
| `ADMISSION_DEADLINE` hardcoded 5ms | 3d/3e config |

3c-3 introduces no new sunset markers. The Phase guard relaxation is a permanent design change.

3c-3 closes one limitation:

- **Pre-3c-3:** "batch boundary at evict_all" (3a/3b convention). Removed.

3c-3 documents new limitations:

- **Prefill stall:** synchronous B=1 prefill in `admit_mid` stalls active rows for `~L_new × B=1_prefill_per_token_time`. Sunset trigger: **3c+ chunked prefill** (interleaves chunks with decode steps to bound stall to `chunk_size × prefill_per_token_time`).
- **`Phase::Finished` admit reject:** during the narrow window between `step` returning a `finish_reason` event and the next `gc_finished_rows`, Phase might be Finished (if it transitions inside step on the last row's finish). In practice `gc_finished_rows` runs immediately after step in driver_loop, so this window is sub-millisecond. No special handling needed.

## §9 Risk register

| Risk | Mitigation |
| --- | --- |
| **R1.** `KVCache::adopt_row_from` slice_update of a partially-allocated dst (dst.keys is None when src_off > 0) | adopt_row_from calls `grow_to(target_capacity)` before slice_update if `src_off > current_capacity`. Lib test `kvcache_adopt_row_from_basic` exercises the fresh-dst path. |
| **R2.** Temp cache lifetime in admit_mid: `temp_cache: Vec<LayerCache>` is owned by the function; it must outlive the adoption loop. | Local Vec ownership — dropped at end of admit_mid. Cache buffers are reclaimed normally. |
| **R3.** Sampler in admit_mid: `state.sampler.sample(...)` borrows the sampler from the inserted RequestState. Sampler holds `Cell<Array>` for per-step state. | The sampler was cloned into RequestState at admit() time. `sample` takes `&self` (interior mutability via Cell). No move; no borrow conflict. Same pattern as `Scheduler::step_inner` line 633. |
| **R4.** `driver_loop` biased-select with always-ready future causes 100% CPU spin when no admits + no active rows (Idle path) | The outer Idle uses `rt.block_on(cmd_rx.recv())` which IS a blocking wait. Inner rolling loop only runs while `active_count > 0`, so always-ready Step is correct (it's productive work, not spin). |
| **R5.** Mid-batch admit races with shutdown (cmd_rx.recv() returning None during rolling select) | RollingEvent::Shutdown handles this: clear event_txs, return. Test `continuous_batching_shutdown_during_decode`: admit A (max_new=20), wait for 1 token, drop SchedulerActorHandle. Verify driver_loop returns within 100ms. (Optional in 3c-3; can defer to 3d.) |
| **R6.** `gc_finished_rows` runs after step but before the next biased-select iteration — if cmd_rx delivers an Admit between step and gc, the Admit could try to use the about-to-be-freed slot | gc runs synchronously after step, BEFORE the next select iteration starts. `tokio::select!` is single-threaded; the driver_loop is one task. No race. |
| **R7.** `handle_admit_mid` invokes prefill which holds model lock for L_new × token time; other workers can't make progress | This IS the documented prefill stall (Q4=A trade-off). 3c+ chunked prefill resolves. No mitigation in 3c-3. Documented in §8. |
| **R8.** Performance regression: rolling loop's biased select + try_recv adds per-step overhead vs 3c-2's tight step-loop | At b_max=4, decode step is ~20-50ms GPU; the tokio::select + try_recv overhead is sub-microsecond. Negligible. Regression sweep verifies no suite slowdowns > 10%. |
| **R9.** `model.make_cache(1, cap, dtype)` creates a fresh B=1 cache. If `Qwen35Model::make_cache` is parametric over batch size today, it works. If hardcoded somewhere to `b_max` from a static config, it doesn't. | Verify in Task 1: `make_cache(1, ...)` returns a working `Vec<LayerCache>`. If not, fix `make_cache` to actually honor the `batch` parameter (it should — it's used by `Scheduler::new` with `b_max` which varies). |
| **R10.** Adoption may need explicit `eval()` if MLX's lazy graph defers the slice_update — adopting before next forward could see stale buffer in the main cache | In MLX, `slice_update` returns a new Array node; reassigning `self.keys = Some(new_keys)` updates the field. The next forward reads `self.keys.as_ref()` which is the post-adoption value. No lazy-eval hazard. Lib test `kvcache_adopt_row_from_basic` reads back via `.to_vec()` which forces eval; if test passes, this risk is closed. |

## §10 Alternatives considered

| Decision | Selected | Rejected |
| --- | --- | --- |
| Scope split | Unified 3c-3 (Q1=A) | Split 3c-3a/b/c (Q1=B/C — too much sub-phase overhead; tight coupling between cache adoption, scheduler API, and driver_loop refactor) |
| Admit check timing | Per-step biased select (Q2=A) | Throttled N-step (Q2=B — added latency); deadline-based (Q2=C — degenerates to per-step at decode time scales) |
| Cache row reuse | Temp B=1 cache + adopt_row_from (β) | reset_row + B=b_max sub-batch + variable mask (α — 3-8× slower, GatedDelta state corruption risk); cache.reset() (too coarse) |
| Prefill timing | Synchronous in driver_loop (Q4=A) | Async via spawn_blocking (Q4=B — non-viable, model `!Send`, single GPU); buffered queue (Q4=C — equivalent to A) |
| Phase enum | Keep 4-state unchanged (Q5=A) | Add PartialAdmitting (Q5=B — extra surface, no gain); compress to Idle/Decoding (Q5=C — too disruptive) |
| Backpressure | Err on full (Q6=A) | In-loop backlog queue (Q6=B — conflicts with 3d admission queue design); preemption (Q6=C — 3e+ work) |
| `gc_finished_rows` integration | Call from driver_loop after step | Inline into step itself (mixes concerns: token generation vs slot lifecycle) |
| `step` Phase transition | Delegated to gc_finished_rows | Keep in step (would require step to know about event_txs, coupling) |
| `admit_mid` prefill batching | Standalone B=1 temp cache (β) | Reuse main cache with B=b_max + per_row_lens skip (α — covered above) |
| Idle exit on cmd_rx Disconnected | Return from driver_loop | Continue with active rows until natural drain (cmd_rx closed means no new admits; finish current work is courteous but might confuse shutdown logic in HTTP server) |

## §11 Linked artifacts

- Predecessor spec: [`docs/superpowers/specs/2026-05-14-b1-p2-3c-2-decode-mask-design.md`](2026-05-14-b1-p2-3c-2-decode-mask-design.md)
- Predecessor close-out: [`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_2_closeout/report.md`](../../ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_2_closeout/report.md)
- `SchedulerActor::driver_loop` current implementation (target of refactor): [`ironmlx/src/core/server/scheduler_actor.rs:111`](../../ironmlx/src/core/server/scheduler_actor.rs#L111)
- `Scheduler::admit` / `evict` (Phase guard relaxation): [`ironmlx/src/core/scheduler.rs:179`](../../ironmlx/src/core/scheduler.rs#L179)
- `Scheduler::step` (transition delegation): [`ironmlx/src/core/scheduler.rs:534`](../../ironmlx/src/core/scheduler.rs#L534)
- KVCache offsets + grow_to (target of adopt_row_from): [`ironmlx/src/core/cache/kv_cache.rs`](../../ironmlx/src/core/cache/kv_cache.rs)
- GatedDeltaCache offsets + state (target of adopt_row_from): [`ironmlx/src/core/cache/gated_delta.rs`](../../ironmlx/src/core/cache/gated_delta.rs)
- Existing batched_prefill (admit_mid uses with B=1): [`ironmlx/src/models/qwen3_5/model.rs::batched_prefill`](../../ironmlx/src/models/qwen3_5/model.rs)
- Existing mask helpers (admit_mid uses): [`ironmlx/src/core/generate.rs::{build_batch_attention_mask, build_batch_linear_mask, build_position_ids_batched}`](../../ironmlx/src/core/generate.rs)
