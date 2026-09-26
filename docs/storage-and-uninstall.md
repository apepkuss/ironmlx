# Data locations and uninstall

## Default locations

| Data | Path |
| --- | --- |
| App configuration | `~/.ironmlx/config/app_config.json` |
| Hugging Face / ModelScope snapshots and imported standalone models | `~/.ironmlx/models/` |
| Paged SSD prefix cache | `~/.ironmlx/cache/paged_prefix_cache/` |
| App and backend logs | `~/.ironmlx/logs/` |
| Model parameters | `~/.ironmlx/model_params.json` |
| Backend incidents | `~/.ironmlx/incidents/backend-incidents.json` |
| Scheduler profile store | `~/.ironmlx/scheduler-profiles/` |
| Scheduler calibration reports | `~/.ironmlx/reports/scheduler-autotune/` |

After a custom cache directory is configured in Dashboard, cache data is written
there instead of the default path. LAN API keys, CA, and TLS private keys are
managed by macOS Keychain under service `com.ironmlx.lan-security.v1`.

## Choose what to remove

- **Remove only the App**: quit IronMLX, confirm the backend has stopped, then delete `IronMLX.app`. Models and settings remain.
- **Reinstall while keeping data**: remove the App as above and install its replacement, retaining `~/.ironmlx` and Keychain entries. This is not a clean installation.
- **Remove App data**: after quitting, remove the App, `~/.ironmlx` and any separately configured cache directory. If LAN mode was used, remove IronMLX LAN security entries in Keychain Access.

Deleting `~/.ironmlx` permanently removes its models, unfinished downloads, settings, logs, caches and reports. Back up anything needed first and check custom directories separately.
This lists App-managed data locations; it does not claim to clear macOS trust history or every system preference.

## Reference: incident retention

`~/.ironmlx/incidents/backend-incidents.json` retains up to 20 incidents and 1 MiB total, with at most 32 KiB of log tail per record. Incident JSON export is capped at 512 KiB. Corrupt or oversized history is treated as empty without blocking startup. Dashboard unread markers reside in WebKit local storage.
