# Network and image security boundary

[简体中文](zh-CN/security-boundary.md)

IronMLX starts in `local` mode. It accepts only a loopback `--host` and serves plain HTTP on that loopback address. External hosts cannot reach this listener.

`lan` mode adds a second listener on one explicitly selected, active LAN IP. The LAN listener always uses HTTPS and requires `Authorization: Bearer <API Key>` on every route, including health, inference, model management, and administrative routes. Wildcard, multicast, unspecified, and loopback LAN addresses are rejected. The loopback listener remains available to the local App and Dashboard.

## LAN client enrollment

1. In Dashboard settings, select **LAN (HTTPS + API Key)** and one concrete LAN interface.
2. Save the setting. The App generates an API key, a local CA, and a server certificate whose subject alternative name is the selected IP.
3. Use the native **Copy API Key** action and immediately place the value in the LAN client's secret store. The Dashboard never receives the key; the native clipboard is cleared after 60 seconds if its value has not changed.
4. Use **Copy CA Certificate** and install or explicitly configure that CA on the LAN client. Do not disable TLS certificate verification.
5. Configure the LAN client, such as an Agent, with the displayed `https://<selected-ip>:<port>/v1` endpoint and send the API key only in the `Authorization` header. Never put it in a URL or query parameter.

**Rotate** creates and copies a new key before activation, restarts the backend, and then retires the previous Keychain item. A failed restart restores the previous configuration and credential.

The App persists only a credential identifier and certificate fingerprint in its ordinary configuration. API keys and TLS private keys are persisted as generic-password data in the user's default macOS Keychain, protected by the Keychain access-control model. IronMLX uses this as its only credential storage path; it does not fall back to files or ordinary configuration when Keychain access fails. The backend receives only an API-key SHA-256 digest and in-memory TLS material through stdin; neither secrets nor the bootstrap payload are placed in arguments, environment variables, Dashboard bootstrap data, logs, or incidents.

## Image input

The OpenAI-compatible API accepts image content only as strict `data:image/jpeg;base64,...`, `data:image/png;base64,...`, or `data:image/webp;base64,...` values. The Anthropic-compatible API accepts the equivalent controlled base64 source. IronMLX never fetches an HTTP or HTTPS image URL. A cross-machine client, including an Agent, must read the image itself and upload its contents.

Limits are enforced before expensive preprocessing:

| Resource | Limit |
| --- | ---: |
| HTTP request body | 32 MiB |
| Text content per request | 2 MiB |
| Images per request | 8 |
| Decoded bytes per image | 10 MiB |
| Total decoded image bytes | 24 MiB |
| Width or height | 8192 px |
| Pixels per image | 16,777,216 |
| Total pixels per request | 33,554,432 |
| Decoder allocation per image | 96 MiB |
| Concurrent image preprocessing jobs | 2 |

## Stable security error codes

| HTTP status | Code | Meaning |
| ---: | --- | --- |
| 401 | `auth_invalid` | Missing, malformed, or incorrect Bearer API key |
| 413 | `request_body_too_large` | HTTP request body exceeds 32 MiB |
| 400 | `image_remote_url_forbidden` | HTTP or HTTPS image URL was supplied |
| 400 | `image_data_url_invalid` | Data URL or base64 is malformed |
| 400 | `image_media_type_unsupported` | Type is not JPEG, PNG, or WebP, or declared and detected types differ |
| 413 | `image_encoded_too_large` | Base64 value exceeds its pre-decode bound |
| 413 | `image_decoded_too_large` | One decoded image exceeds 10 MiB |
| 413 | `image_total_decoded_too_large` | Decoded images exceed 24 MiB in total |
| 413 | `image_count_exceeded` | More than 8 images were supplied |
| 413 | `image_dimensions_exceeded` | Width or height exceeds 8192 px |
| 413 | `image_pixel_budget_exceeded` | One image exceeds the pixel budget |
| 413 | `image_total_pixel_budget_exceeded` | All images exceed the request pixel budget |
| 413 | `text_content_too_large` | Request text exceeds 2 MiB |
| 400 | `image_decode_failed` | The image cannot be decoded within safety limits |

LAN authentication is deliberately applied around the entire router, so new inference or administrative endpoints are protected by default rather than relying on a route-by-route allowlist.
