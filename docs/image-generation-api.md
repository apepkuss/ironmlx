# Image generation API

[简体中文](zh-CN/image-generation-api.md) · [API reference](api-reference.md) · [Service and management API](service-api.md) · [Supported models](supported-models.md)

IronMLX exposes OpenAI-compatible text-to-image generation and single-image
conditional editing for `mlx-community/Qwen-Image-2.1-MLX-4bit` through:

```text
POST /v1/images/generations
POST /v1/images/edits
```

The runtime is implemented in native Rust/MLX. It does not require Python or
`mflux` at inference time. The App downloads, scans, loads and manages the
model. IronMLX performs model inference; image presentation, mask authoring,
canvas operations and post-generation compositing belong to the calling
application.

## Request

```bash
curl http://127.0.0.1:9068/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen-Image-2.1-MLX-4bit",
    "prompt": "A small red circle centered on a clean white background",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json",
    "seed": 7,
    "inference_steps": 40
  }'
```

| Field | Support |
| --- | --- |
| `model` | Optional when the service has an unambiguous default; otherwise identifies the loaded image-generation model |
| `prompt` | Required non-empty text prompt |
| `n` | Optional; only `1` is supported |
| `size` | Optional `WIDTHxHEIGHT`; defaults to `1024x1024` |
| `response_format` | Optional; only `b64_json` is supported |
| `quality`, `style`, `user` | Accepted for OpenAI request-shape compatibility; they do not change local inference |
| `seed` | IronMLX extension; optional unsigned integer for reproducible generation |
| `inference_steps` | IronMLX extension; optional integer, defaults to `40` |

Width and height must each be a multiple of 32 in the range 256–2048. Total
pixels must not exceed 1,572,864, and `inference_steps` must be in the range
2–200.

## Conditional editing request

`POST /v1/images/edits` accepts `multipart/form-data`. It feeds one condition
image plus the prompt to Qwen Image 2.1 and returns the newly generated image:

```bash
curl http://127.0.0.1:9068/v1/images/edits \
  -F 'model=mlx-community/Qwen-Image-2.1-MLX-4bit' \
  -F 'prompt=Change the centered circle from red to blue' \
  -F 'image=@condition.png;type=image/png' \
  -F 'size=1024x1024' \
  -F 'response_format=b64_json' \
  -F 'seed=11' \
  -F 'inference_steps=40'
```

The edit endpoint shares the generation contract for `model`, `prompt`, `n`,
`size`, `response_format`, `seed` and `inference_steps`. `quality` and `user`
are accepted but do not alter inference. Exactly one JPEG, PNG or WebP
condition image is required; it may be at most 10 MiB, 8192 pixels on either
side and 16,777,216 pixels in total. IronMLX verifies the declared media type
against the actual file bytes. The model-side preprocessing preserves aspect
ratio, targets approximately 1024² pixels and aligns both dimensions to 32.

`mask` is deliberately rejected with `unsupported_image_mask`: this endpoint
provides whole-image conditional editing, not masked inpainting.

## Response

The response uses the OpenAI Images shape. Each `b64_json` value is a complete
Base64-encoded PNG:

```json
{
  "created": 1790480000,
  "data": [
    { "b64_json": "iVBORw0KGgo..." }
  ]
}
```

Requests are executed on a bounded serial image lane. A full queue returns an
OpenAI-style 503 error with code `image_queue_full` and may be retried later.
Invalid prompt, count, format, size or step values return an OpenAI-style 400
error.

## Boundaries

- Masked inpainting and image-variation endpoints are not implemented.
- Mask creation/management, canvas operations, compositing and editor UI are
  outside the IronMLX inference-service boundary.
- There is no Anthropic image-generation endpoint.
- Qwen Image 2.1 is distributed under the Qwen Research License. Review its
  noncommercial/research-use conditions before download or deployment.

## Developer verification

`ironmlx/tests/qwen_image_http.rs` is the ignored real-model acceptance test.
It starts the App daemon with a 32 GiB total and 16 GiB per-model software
memory limit, dynamically registers and loads the exact model, generates a PNG,
uses that PNG in a conditional-edit request, verifies the returned PNG and
protocol boundaries, and unloads the model.

After building `dist/IronMLX.app`, run the fast 256×256, two-step lifecycle
check against that exact Bundle with:

```sh
IRONMLX_QWEN_IMAGE_MODEL_DIR=/path/to/Qwen-Image-2.1-MLX-4bit \
IRONMLX_QWEN_IMAGE_HTTP_BUNDLE="$PWD/dist/IronMLX.app" \
  cargo test --locked -p ironmlx --release --test qwen_image_http -- \
  --ignored --nocapture --test-threads=1
```

The 32 GiB target-machine qualification uses the same test but exercises the
API defaults (1024×1024 and 40 steps) and verifies the machine's reported
physical memory before loading the model:

```sh
IRONMLX_QWEN_IMAGE_MODEL_DIR=/path/to/Qwen-Image-2.1-MLX-4bit \
IRONMLX_QWEN_IMAGE_HTTP_BUNDLE="$PWD/dist/IronMLX.app" \
IRONMLX_QWEN_IMAGE_FULL_ACCEPTANCE=1 \
IRONMLX_QWEN_IMAGE_EXPECTED_MEMORY_GB=32 \
  cargo test --locked -p ironmlx --release --test qwen_image_http -- \
  --ignored --nocapture --test-threads=1
```

The test prints physical memory, load time, generation/edit time, server RSS
and PNG size. Bundle mode uses only the Bundle helper and metallib, clears
`MLX_*` and `DYLD_*` overrides in the child, and runs the service outside the
checkout. On failure it preserves the temporary server log and prints its
directory. A run on a larger machine with software limits is useful evidence
but does not replace the physical 32 GiB qualification.
