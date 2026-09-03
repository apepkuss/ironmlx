# Troubleshooting

| Symptom | Check and action |
| --- | --- |
| App will not start | Confirm Apple Silicon and macOS 26.2+; for a local build run `scripts/verify-app-bundle.sh dist/IronMLX.app`. |
| Model download interrupted | Retry the same model in Dashboard; the downloader resumes by immutable commit and Range/ETag. Do not move `.partial` files manually. |
| Model rejected | Check Dashboard readiness and `~/.ironmlx/logs/backend.log`; verify architecture, quantization metadata, disk, memory, and snapshot integrity. |
| API connection fails | Confirm the App endpoint and port; local mode is local-only. Request `/health` before `/healthz`. |
| LAN returns 401 | Send `Authorization: Bearer ...`; copy or rotate the API key again. Every LAN route requires authentication. |
| LAN TLS fails | Install or specify the CA exported by the App and connect to the concrete IP in the certificate. Do not disable verification. |
| Image request rejected | Use JPEG/PNG/WebP base64 only; HTTP/HTTPS URLs are disabled. Check body, byte, and pixel limits. |
| Slow first token or low throughput | Use a Release build and NAX-enabled MLX; inspect memory pressure, concurrency, prefill chunks, KV/prefix cache, and scheduler profile. |
| Cache is too large | Adjust the Dashboard capacity or clear `~/.ironmlx/cache/paged_prefix_cache` after stopping the backend. |
| Backend exits repeatedly | In Dashboard **Logs → Incident history**, inspect the structured reason and recovery action. **Export incident records** saves the current filtered JSON; **Export diagnostic information** saves the full redacted ZIP. |

When reporting a problem, include App/CLI version, macOS and chip, immutable
model commit, quantization, reproduction request, and redacted logs. For
performance issues also fix concurrency, prompt/output tokens, cache warmth,
and provide multiple runs.
