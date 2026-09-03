# Data locations and uninstall

## Default locations

| Data | Path |
| --- | --- |
| App configuration | `~/.ironmlx/config/app_config.json` |
| Hugging Face / ModelScope snapshots | `~/.ironmlx/models/` |
| Paged SSD prefix cache | `~/.ironmlx/cache/paged_prefix_cache/` |
| App and backend logs | `~/.ironmlx/logs/` |
| Model parameters | `~/.ironmlx/model_params.json` |
| Backend incidents | `~/.ironmlx/incidents/backend-incidents.json` |
| Scheduler profile store | `~/.ironmlx/scheduler-profiles/` |
| Scheduler calibration reports | `~/.ironmlx/reports/scheduler-autotune/` |

After a custom cache directory is configured in Dashboard, cache data is written
there instead of the default path. LAN API keys, CA, and TLS private keys are
managed by macOS Keychain under service `com.ironmlx.lan-security.v1`.

## Uninstall

1. Quit IronMLX and confirm the backend has stopped.
2. Delete `IronMLX.app`.
3. If you do not need models or configuration, delete `~/.ironmlx`.
4. Delete a custom cache directory separately, if one was configured.
5. If LAN mode was enabled, remove the IronMLX LAN security entries in Keychain Access.

Deleting `~/.ironmlx` permanently removes downloaded models, partial downloads,
configuration, logs, caches, and reports. To keep models, back up
`~/.ironmlx/models` first instead of deleting the entire directory.
