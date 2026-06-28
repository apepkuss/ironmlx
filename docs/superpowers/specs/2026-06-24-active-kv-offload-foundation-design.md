# Active KV Offload Foundation Design

## Goal

The first production step is not a raw swap path. It establishes an extensible
Active KV Storage / Residency layer and uses it for conservative request-level
pause and restore. Normal active decode keeps the existing full-resident
attention path. A later phase can replace that attention path with chunked or
streaming attention without discarding this foundation.

## Scope

First version:

- Default off.
- Add an Active KV offload setting for app/server mode.
- Track KV residency states: `resident`, `offloaded`, `loading`, `dirty`.
- Persist offloaded KV payloads to a temporary SSD directory.
- Expose status and counters through `/healthz`.
- Show the setting and runtime status in Dashboard.
- Use the storage layer for request-level pause/restore only.
- Before a paused request resumes, restore all required KV back to resident
  memory, preserving current attention correctness.

Out of scope for the first version:

- Token-level transparent active KV hot/cold migration during a decode step.
- Streaming or chunked attention over offloaded pages.
- Reusing active KV offload entries across processes or server restarts.

## Architecture

`core::cache::active_kv` owns the storage and residency primitives. The module
contains:

- `ActiveKvResidencyState`: page lifecycle enum.
- `ActiveKvPageResidency`: per-page logical state and dirty bit.
- `ActiveKvOffloadStore`: temporary SSD storage for request cache payloads.
- `ActiveKvOffloadStats`: atomics shared by scheduler, actor, and `/healthz`.
- `ActiveKvOffloadConfig`: enabled flag and storage directory.

The first implementation stores a request payload as the existing
`PagedPrefixEntry` tensor container plus request metadata. The tensor payload
may contain paged full-attention KV, TurboQuant packed KV, Linear cache state,
MLA latent cache, and optional MTP cache payload. This keeps the persisted
format aligned with current cache export and restore helpers.

## Scheduler Behavior

When enabled, the scheduler actor may park an eligible request to make room for
queued work or memory pressure handling. Parking means:

1. Export the request's compact cache row and related MTP state.
2. Save the payload through `ActiveKvOffloadStore`.
3. Remove the request from resident scheduler slots and release its KV budget.
4. Keep its event channel open.

Restoring means:

1. Load the payload.
2. Admit the parked request back into a free slot.
3. Rebuild cache layout and restore the payload into the new compact cache row.
4. Continue decode from the same request id and token history.

The first version does not offload rows that cannot be restored correctly. Such
cache types are marked as `not_applicable` in status, not treated as errors.

## Observability

`/healthz` gains an `active_kv_offload` object:

- `enabled`
- `mode`: `request_preemption`
- `storage_dir`
- `resident_pages`, `offloaded_pages`, `loading_pages`, `dirty_pages`
- `parked_requests`
- `offloaded_bytes`
- `swap_out_count`, `swap_in_count`, `swap_error_count`
- `last_swap_out_us`, `last_swap_in_us`
- `supported_cache_kinds`
- `not_applicable_cache_kinds`

Dashboard reads this from the existing health polling path.

## Failure Handling

Offload is opportunistic. If save or restore fails, the scheduler records the
error counter and falls back to the existing queueing behavior when possible.
It must not corrupt a live request. A parked request whose payload cannot be
restored fails that request explicitly rather than silently producing incorrect
tokens.

## Validation

Required validation before completion:

- Unit tests for residency state transitions and temporary store lifecycle.
- Scheduler tests proving parked requests keep request id, token history, and
  event channel semantics after restore.
- `/healthz` serialization tests for the new status object.
- App tests proving the setting is persisted and translated into backend CLI
  arguments.
- Functional smoke test with offload disabled and enabled.
- Performance baseline showing disabled mode has no measurable decode-path
  regression and enabled mode only adds cost when offload/restore triggers.
