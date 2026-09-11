# Runtime log level

[简体中文](zh-CN/runtime-log-level.md)

The Settings log-level control applies and saves one setting immediately, without restarting the backend or saving other pending Settings edits. It controls App diagnostics and IronMLX backend diagnostics. The Logs page's level selector only filters existing records for display.

The choices are All, DEBUG, INFO (default), WARNING, and ERROR. All includes TRACE; stored `TRACE` and `WARN` values are accepted as aliases for `ALL` and `WARNING`. Backend dependency logs remain capped at WARNING (ERROR when ERROR is selected), so verbose project diagnostics do not enable dependency wire dumps.

The App installs the saved filter at startup and passes `IRONMLX_LOG_LEVEL` to each helper launch. Standalone CLI usage still accepts `RUST_LOG` when no App level is supplied. Swift DEBUG diagnostics cover backend state transitions, download queue scheduling, and level application; enabling a level does not create diagnostics for code paths that do not emit them or recover previously suppressed records.

## Applying a change

The App reads the backend level, applies the new filter with a process ID and revision precondition, persists only `log_level`, then updates its own filter. A stopped backend needs no HTTP request; the saved level applies on its next launch. Changes during startup, recovery, shutdown, or another setting application are rejected for retry.

The control is disabled while applying. A failed application restores the saved selection and reports the failure. When a response is lost or configuration persistence fails, the App queries the backend and conditionally restores the previous level. If the process/revision no longer matches or rollback cannot be verified, the UI explicitly reports that recovery is unconfirmed rather than claiming success.

## Local management API

`GET /admin/api/log-level` returns `level`, `revision`, and `process_id`. A custom CLI `RUST_LOG` filter is represented by a null `level` until replaced by a canonical level; the App requires a canonical initial level.

`POST /admin/api/log-level` accepts:

```json
{"level":"DEBUG","expected_revision":0,"expected_process_id":12345}
```

It returns the updated snapshot. Stale revision/process preconditions return 409; invalid JSON or unsupported levels are rejected; unavailable runtime control returns 503. The route is merged only into the loopback listener and is absent from the LAN listener, even with a valid LAN API key.

## Verification

Focused tests check actual log-file output, reload output, configuration isolation, failed persistence, lost responses, rollback failure, stopped-backend behavior, and the Dashboard message flow. The opt-in `runtimeLogLevelLiveHelperAndStreamingInference` test uses a built helper and an existing model snapshot to check five levels during SSE inference, unchanged PID, stream completion, and saved-level restoration after a planned restart. It requires the current App to be stopped because the production helper enforces one instance per macOS user.
