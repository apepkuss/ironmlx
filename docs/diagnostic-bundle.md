# Privacy-safe diagnostic export

The Dashboard Logs page provides **Export diagnostic information**. The App
collects data locally through native messaging and saves a ZIP only to the
location explicitly chosen in the macOS save panel. It does not use the backend
HTTP API, accept LAN-client triggers, upload data, or enable telemetry.

## Fixed format

Schema version is `1` and ZIP entries are ordered as follows:

1. `manifest.json`
2. `system.json`
3. `runtime-health.json`
4. `models.json`
5. `incidents.json`
6. `logs/app.log`
7. `logs/backend.log`

The manifest records App/backend/MLX build identity, channel, backend status,
signature/notarization status, and each entry's state, byte count, and truncation
flag. If the backend is offline or health fails within two seconds, the archive
still completes with a stable error code; collection never starts, restarts, or
stops the backend.

## Allowlist and redaction

The archive includes system facts, runtime health, model provider/repo and
immutable revision, normalized incidents, and bounded tails of the two current
log files. It excludes original configuration, environment variables, Keychain,
request bodies, weights, tokenizers, complete model configuration, sidecars, and
stable identity values such as username, hostname, serial number, Apple ID, and
MAC address.

Structured allowlisting is followed by uniform redaction of prompts/messages,
tool arguments, Authorization/Bearer, Cookie, HF token, LAN API key, passwords,
usernames, and home-directory paths.

## Size and safe writing

The archive bounds models and incidents to 512 KiB each, App logs to 512 KiB,
backend logs to 1.5 MiB, all uncompressed entries to 3.5 MiB, and the final ZIP
to 4 MiB. Logs use `O_NOFOLLOW` and bounded tail reads. ZIP data is generated
from redacted memory in fixed order; a `0600` hidden temporary file is fsynced and
atomically published, while cancellation or errors remove the temporary file.

**Export incident records** in Incident history is a separate filtered JSON
export; it is not the complete diagnostic ZIP.
