# Embedding API

[简体中文](zh-CN/text-embeddings.md) · [API reference](api-reference.md) · [Service and management API](service-api.md)

IronMLX supports the text, vision and audio encoders in `mlx-community/embeddinggemma-2-bf16`
and `mlx-community/embeddinggemma-2-4bit` (MLX affine, 4 bits, group size 64).
Video inputs are not supported. The models can be downloaded,
loaded and unloaded in the App; existing verified file-only downloads are
re-evaluated by the scanner and do not need another weight download.

The encoder lives in `ironmlx-lm::models::embedding_gemma2`. It shares the
existing full-precision and quantized layer primitives, but uses bidirectional
attention and a dedicated PLE graph rather than the generation `Model` trait.
`ironmlx-runtime` owns its bounded worker queue, memory admission and model leases.
`ironmlx` provides HTTP transport on the App daemon and engine-pool listeners.

## API

`POST /v1/embeddings` accepts a required registered `model` and `input` as a
nonempty string, an ordered content object, or an array of 1–32 strings/content objects. Each sample may contain at
most 8192 tokens, including special tokens and task prefixes; inputs are rejected
rather than silently truncated. `dimensions` is optional (default 768) and accepts
128, 256, 512 or 768. `encoding_format` accepts `float` (default) or `base64` (little-endian float32).
Token-ID and video inputs are not supported. The content-object form is an IronMLX extension; the original string forms and response format remain OpenAI-compatible.

```sh
curl http://127.0.0.1:9068/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/embeddinggemma-2-4bit","input":["task: search result | query: Which planet is red?","title: none | text: Mars is known as the Red Planet."],"dimensions":256}'
```

## Images and combined inputs

An ordered `content` array produces **one vector for the entire sample**. Text uses
`{"type":"text","text":"..."}`. Images use
`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`.
Image-only samples omit text. An input array produces one vector per sample and
can mix strings and content objects. Parts are encoded in their supplied order,
including interleaved text and multiple images.

```python
import base64
import json
import urllib.request
from pathlib import Path

image = base64.b64encode(Path("photo.png").read_bytes()).decode("ascii")
payload = {
    "model": "mlx-community/embeddinggemma-2-4bit",
    "input": {"content": [
        {"type": "text", "text": "A description of the photograph. "},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
    ]},
    "dimensions": 768,
}
request = urllib.request.Request(
    "http://127.0.0.1:9068/v1/embeddings",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(request)))
```

Images must be JPEG, PNG or WebP data URLs; remote URLs and server file paths are
rejected. Each request allows at most 8 images, 10 MiB per encoded image file,
24 MiB of image file bytes after base64 decoding in total, and a 32 MiB JSON body.
Image dimensions are limited to 8192 pixels per side, 16,777,216 pixels per image
and 33,554,432 pixels per request. Each sample allows up to 64 parts and 128 KiB
of text. Payload/image budget violations return HTTP 413; malformed images return 400.

The model adapter converts images to RGB, performs aspect-preserving,
Pillow-compatible bicubic resize within the default 280-soft-token budget, rescales
pixels by 1/255, and pads patches with an attention mask. It inserts image boundary
tokens and replaces only image placeholder embeddings with projected vision
features. Supply original images and text, without manually inserting media tokens
or resizing images. The 8192-token limit includes text, special tokens and actual
image soft tokens; image soft tokens are also counted in `usage`.
All four dimensions support images and combined inputs. Prefer 256–768 dimensions
for image retrieval; 128 dimensions are primarily suited to text, as described in
the [official model guidance](https://huggingface.co/google/embeddinggemma-2#3-matryoshka-dimension-truncation).

## Audio and text/audio combinations

Use `{"type":"input_audio","input_audio":{"data":"BASE64","format":"wav"}}`
inside the same ordered `content` array. Audio-only samples can omit text; text,
audio and images can be interleaved, and each sample still produces one vector.
`data` is canonical standard base64 of the complete file, without a data-URL prefix.
The decoder never fetches remote URLs or reads server-side file paths.

```python
from pathlib import Path
import base64
import requests

audio = base64.b64encode(Path("recording.wav").read_bytes()).decode("ascii")
response = requests.post("http://127.0.0.1:9068/v1/embeddings", json={
    "model": "mlx-community/embeddinggemma-2-4bit",
    "input": {"content": [
        {"type": "text", "text": "Recorded speech: "},
        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
    ]},
    "dimensions": 256,
})
response.raise_for_status()
vector = response.json()["data"][0]["embedding"]
```

Supported files are PCM16/24/32 or float32 WAV, FLAC and MP3, mono/stereo at
8–96 kHz. Floating PCM must be finite and normalized to [-1, 1]. Audio must be
longer than 10 ms and no longer than 30 seconds per clip. Each request allows
up to 8 clips, 16 MiB per encoded file, 24 MiB total decoded file bytes and
60 seconds total after resampling. Decoded source PCM is bounded to 48 MiB
per clip. The existing 32 MiB JSON body, 32 samples and 64 parts per sample
limits also apply. Malformed/mismatched formats return 400; capacity violations
return 413. Four bounded CPU decoders run before GPU model acquisition; busy
decoders return 429. Audio is rejected on overflow, never silently cropped.

IronMLX downmixes stereo and resamples to mono 16 kHz with the shared sinc
resampler. The model adapter applies the checkpoint's semicausal 320-sample
periodic Hann window, 160-sample hop, 512-point FFT and 128 HTK magnitude mel
filters with `log(mel + 0.001)`. Frame masks pass through two stride-2
subsampling blocks; valid output frames determine the actual audio soft-token
count. The adapter inserts audio boundary tokens and replaces placeholders
with audio features. No manual mel extraction or media tokens are required.
Audio, text/audio and mixed media support all four output dimensions and both
response encodings. `usage` includes text, boundaries and actual audio/image
soft tokens. Task prefixes remain explicit and apply to text parts only.

The OpenAI-compatible response contains `object: "list"`, ordered `data` entries
with `object: "embedding"`, `index` and a float vector, plus `model` and
`usage.prompt_tokens` / `usage.total_tokens`. Vectors use mean pooling over all
valid tokens, including prompt tokens, followed by L2 normalization. Shortened
vectors are normalized again. Requests are serialized per model, and sequences
within a batch are encoded independently to bound memory and preserve single-input
results. A missing resident model is loaded through the normal lifecycle.

Supply upstream task prefixes explicitly; the API does not insert a retrieval
prefix or chat template. Use the same checkpoint, dimensions and task-prefix
convention for indexing and querying. See the [official model instructions](https://huggingface.co/google/embeddinggemma-2).
Non-quantized weights and activations remain BF16 (FP32 for pooling and rotary
angles); they must not be converted to FP16. The affine4 checkpoint quantizes the text encoder; its vision tower, image projection and audio tower retain BF16 weights; the audio projection is affine4. The audio tower uses FP32 intermediate computation before projecting to BF16 text activations.

Invalid inputs return an OpenAI error object with HTTP 400. A full embedding queue
returns 429, unavailable workers/models return 503, and a request timeout returns
504. In-flight GPU work retains its model lease even if the client disconnects.
Embedding models do not accept generation settings or serve chat endpoints.

## Runtime status

The App Status page displays embedding queue activity and dedicated vector metrics,
without decode, prefill or TTFT fields. Completed/failed counts are cumulative since
loading; recent latency and throughput are P50 values over the last 60 seconds
(up to 4096 samples). Runtime latency includes queue wait and processing, excluding
HTTP media decoding, model loading and response serialization. Input-token and
vector throughput use processing time, excluding queue wait; image and audio soft tokens
count as input tokens and each batch item produces one vector. Empty/expired
performance samples display a dash. Counts reset when the model is unloaded and
loaded again. Failures cover runtime rejections/execution failures, queue overflow,
timeouts and queued cancellation; HTTP validation errors before runtime admission
are excluded. A timeout racing worker completion records only one outcome.
Metrics are exposed as `embedding_metrics` in `/healthz` model entries and
`/admin/api/models/loaded`.

## Qualification

The pinned numerical fixture uses MLX 0.32.3 and the text, vision and audio implementations in
[MLX-VLM revision 3d87e884](https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0).
BF16 and affine4 are compared to their respective upstream versions, not to each
other. Audio fixtures pin Transformers `cb33194ad6152bd9fad6305378d92db385dd7b32` and NumPy 1.26.4, including feature masks, speech retrieval, silence, short clips and ordered text/audio. Audio vectors require cosine similarity above 0.9995 and maximum component error below 0.005; FP32 encoder kernel rounding can cross BF16 activation boundaries. Text and image numerical tolerances remain unchanged. The image fixtures cover image-only, text/image interleaving, multiple images and basic cross-modal retrieval. These are numerical, retrieval sanity and lifecycle checks, not a full retrieval benchmark.

Set `MLX_DIR` to the compiled MLX installation, `IRONMLX_EMBEDDING_BF16_DIR` and
`IRONMLX_EMBEDDING_AFFINE4_DIR` to the corresponding local snapshots, then run:

```sh
cargo test --release -p ironmlx-lm --test embedding_gemma2_checkpoint
cargo test --release -p ironmlx-lm --test embedding_gemma2_checkpoint -- --ignored --test-threads=1
cargo test --release -p ironmlx --lib embedding_real_app_lifecycle_and_api -- --ignored
```
