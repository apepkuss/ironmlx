# Developer guide

[简体中文](zh-CN/developer-guide.md)

Start here to develop IronMLX from source, integrate APIs, or configure CLI serving. For App installation and model management, see the [User guide](user-guide.md).

[Source development](#source-development) · [API integration](#api-integration) · [CLI and advanced configuration](#cli-and-advanced-configuration) · [Contribute and maintain](#contribute-and-maintain)

## Source development

Start with [Build from source](building-from-source.md) for platform requirements,
MLX dependencies, App and CLI builds, and the local runtime environment.
Check [Supported models](supported-models.md) when adding or modifying a model.

### Project structure and crate references

| Crate | Responsibility / reference |
| --- | --- |
| [`ironmlx-core`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-core) | Shared tensor and weight primitives |
| [`ironmlx-lm`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-lm) | Language and vision models, including [multimodal embedding encoders](text-embeddings.md) |
| [`ironmlx-image`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-image) | Image-generation models; see the [image API](image-generation-api.md) |
| [`ironmlx-audio`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-audio) | Audio models; see the [developer reference](../ironmlx-audio/README.md) |
| [`ironmlx-decision`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-decision) | Decision models; see the [System One API](laya-systemone-api.md) |
| [`ironmlx-runtime`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-runtime) | Execution, scheduling, lifecycle, and resources |
| [`ironmlx`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx) | HTTP API and CLI |
| [`ironmlx-app`](https://github.com/apepkuss/ironmlx/tree/dev/ironmlx-app) | macOS App and Dashboard |
| [`iron-bench`](https://github.com/apepkuss/ironmlx/tree/dev/iron-bench) | Benchmark tooling |

Crate-name links open source directories. When changing a model or protocol,
trace the path from the App or HTTP request through runtime execution to the response.

## API integration

Use the [API reference](api-reference.md) to look up all documented endpoints and choose a service and management, text and vision, embedding, speech synthesis, image generation or System One topic.

### API quick start

This example uses Responses with a text or vision model. For other capabilities, use the request examples in the corresponding topic linked from the API reference.

Start the App and load a compatible model. The App endpoint defaults to
`http://127.0.0.1:9068`; direct CLI serving defaults to port 8080. Use the actual
configured endpoint. Only one backend may run per macOS user; exit the existing
App backend before starting a separate CLI server.

Check the service and discover model IDs:

```bash
curl http://127.0.0.1:9068/health
curl http://127.0.0.1:9068/healthz
curl http://127.0.0.1:9068/v1/models
```

`/health` checks HTTP responsiveness; `/healthz` reports runtime state. The App
model list also contains registered models that are not loaded. Use the ID of
an available model and replace `your-model-id` below:

```bash
curl http://127.0.0.1:9068/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "input": "Hello", "store": false, "max_output_tokens": 128, "stream": false}'
```

Responses is stateless: send complete history with each request. IronMLX does
not store conversations or execute tools; the client executes tool calls.

See [Text and vision API](text-vision-api.md) for protocol fields and limits,
examples, SSE events, tools, structured outputs and errors. Addresses, LAN authentication,
health checks, model discovery and management endpoints are defined in
[Service and management API](service-api.md). Network configuration and resource
limits are documented in [Security boundary](security-boundary.md).

## CLI and advanced configuration

| Task | Guide |
| --- | --- |
| Configure DFlash2 | [CLI generation, HTTP serving, and App settings](dflash2-server-api.md) |
| Configure Qwen MTP | [Matching weights, startup parameters, and request limits](mtp-server-api.md) |
| Serve multiple models | [CLI manifest, routing, loading, and unloading](engine-pool.md) |
| Calibrate scheduling | [Optional offline performance calibration](scheduler-profile-v5.md) |

## Contribute and maintain

| Task | Entry point |
| --- | --- |
| Verify changes | [Rust, Swift, and App Bundle checks](building-from-source.md#verify-changes) |
| Submit a contribution | [Contribution terms and requirements](../CONTRIBUTING.md) |
| Investigate a problem | [Troubleshooting](troubleshooting.md) and [diagnostic export](diagnostic-bundle.md) |
| Request support | [Reporting guidance](../SUPPORT.md) |
| Report a vulnerability | [Private security reporting](../SECURITY.md) |
| Understand releases | [Versions and channels](versioning-and-releases.md) |
| Review release changes | [0.2.0 notes](release-notes/0.2.0.md) and [0.1.0 notes](release-notes/0.1.0.md) |
