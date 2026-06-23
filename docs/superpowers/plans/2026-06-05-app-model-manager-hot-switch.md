# App Model Manager Hot Switch Plan

## Goal

Support the Swift menu bar app as a long-lived backend client by adding an app daemon mode:

- `ironmlx serve --model <path>` keeps the existing single-model CLI behavior.
- `ironmlx serve` starts a resident app backend without a bound model.
- Admin APIs load, unload, list, and set the default model.
- In-flight requests keep the model runtime they acquired; new requests use the new default model.
- Loading failures caused by GPU memory pressure return a user-facing professional message instead of auto-unloading another model.
- App-mode scheduler profile lookup falls back to default config with a warning when no profile is available.

## Steps

1. Add tests for optional `--model`, daemon model manager semantics, and Swift launch arguments.
2. Extract reusable model runtime construction from the existing single-model server path.
3. Refactor OpenAI and Anthropic handlers so app daemon can dispatch a selected runtime into the existing generation logic.
4. Add `core::server::model_manager` with loaded-runtime map, default model pointer, admin API handlers, and aggregate health.
5. Update `cli::serve` so no `--model` starts daemon mode.
6. Update Swift backend launch to start daemon mode, then call backend admin load/unload/default APIs from the dashboard bridge.
7. Run Rust and Swift verification required for this worktree.
