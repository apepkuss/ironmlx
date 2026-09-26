# Privacy and network boundary

IronMLX performs inference locally on Apple Silicon. In `local` mode it binds
only to loopback, so external hosts cannot directly access the API.

## Operations that access the network

- User-initiated model search and downloads access Hugging Face or ModelScope.
- Update-enabled builds check the GitHub-hosted update feed and download release assets from the URLs it provides. Automatic checks depend on updater settings; manual checks also access the network.
- Opening an official Agent guide launches the browser and contacts that documentation site.
- User-enabled LAN mode serves HTTPS on the selected concrete LAN address.
- The inference API does not fetch remote images for the client; images must be
  uploaded as controlled base64 content.

Version 0.2.0 has no product telemetry or cloud inference upload path. Download
progress and errors are written to local logs. Model and update hosting services may record
client IPs, requests, and token use under their own policies.

## Local data

Configuration, models, caches, logs, incident records, and scheduler reports are
stored under `~/.ironmlx` by default. LAN API keys and TLS private keys are kept
in the user's default macOS Keychain; ordinary configuration stores only a
credential identifier and certificate fingerprint. See [Data locations and uninstall](storage-and-uninstall.md).

## Requests and logs

Backend logs may contain model IDs, runtime parameters, errors, and performance
diagnostics. Do not send raw logs to untrusted parties. Dashboard **Export
diagnostic information** uses a structured allowlist and redaction to create a
local ZIP that excludes request bodies, credentials, identity paths, original
configuration files, and the full environment. Review an export before sharing.
See [Diagnostic information export](diagnostic-bundle.md) for limits.

LAN mode assumes one trusted network/model-engine domain. Do not reuse
cross-request Prompt Lookup history across untrusted tenants.

## Incident-record privacy

Incident records exclude prompts, full request bodies and credentials; log tails are redacted before persistence and export. Incident operations belong to the embedded App bridge, not backend or LAN HTTP routes. See [Diagnostic export](diagnostic-bundle.md) for the full diagnostic ZIP scope.
