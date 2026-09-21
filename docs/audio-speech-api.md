# Speech synthesis API

The server exposes `POST /v1/audio/speech` through the model pool and the App
model-management daemon. `ironmlx-audio` supplies native synthesis and audio IO;
`ironmlx-runtime` owns loading, scheduling, cancellation and memory admission.

## Use from IronMLX.app

Download `mlx-community/IndexTTS-2.5-fp16` in the Dashboard. The App selects the
supported immutable revision and automatically downloads and verifies the
reference-encoder dependencies, WeText data and UniDic-lite dictionary. It builds
the auxiliary safetensors natively; Python, PyTorch and package installation are
not required. The first download requires network access. Verified resources are
reused offline and are stored under `~/.ironmlx/audio/indextts25/<revision>/`.

When preparation completes, load the model from the model list. The App passes
its saved resource configuration to the backend and restores loaded/pinned models
on restart. Call `/v1/audio/speech` using the model identifier and a Base64 reference
audio clip as shown below. Playback is provided by your API client or the standalone
example, rather than an embedded Dashboard player.

For a previously downloaded model with missing or changed resources, use **Prepare
resources** in the model list. Interrupted downloads can be retried; verified model
weights are reused. Other TTS repositories and revisions require their own verified
runtime profile and are not enabled merely by their model type.

## Register local resources

An audio model needs a verified source snapshot, derived reference-encoder
resources, the fixed resource lock, WeText FST files and the UniDic-lite dictionary.
See [the audio library](../ironmlx-audio/README.md) for the resource profile and
conversion tools. Resource paths are explicit local paths. Loading validates the
resources; the server does not download or convert them.

For the model pool, use `ironmlx serve --model-manifest models.json`:

```json
{
  "models": [
    {
      "id": "mlx-community/IndexTTS-2.5-fp16",
      "path": "/models/indextts25/snapshots/<revision>",
      "load_policy": "lazy",
      "audio": {
        "derived_resources": "/models/indextts25-derived",
        "resource_lock": "/resources/indextts25/sources.json",
        "wetext_fsts": "/resources/wetext/fsts",
        "unidic_dir": "/resources/unidic_lite/dicdir"
      }
    }
  ]
}
```

For the model-management daemon (`ironmlx serve`), the existing local
`/admin/api/models/register` and `/admin/api/models/load` endpoints accept the same
`audio` object, together with `model` (the public identifier) and `model_dir`
(the source snapshot). Existing unload, pinning, TTL and model-count policies apply.
Audio models report `runtime_kind: "tts"`, architecture `indextts25` and scheduler
`serial_audio`. Causal scheduler, sampling, MTP and PromptLookup overrides are rejected.

The executable defaults `MLX_ENABLE_TF32` to `0` before initializing MLX or
starting workers. Explicit environment settings are preserved; the audio loader
rejects values other than `0`. Applications embedding the Rust runtime must set
this process-wide precision policy before their first MLX call. Audio workers use
separate registered streams and request-local random state. Memory and allocator
cache limits remain process-wide and shared with language models.

## Request

Complete WAV response:

```json
{
  "model": "mlx-community/IndexTTS-2.5-fp16",
  "input": "这是需要合成的文本。",
  "ref_audio": "<Base64>",
  "response_format": "wav"
}
```

PCM stream:

```json
{
  "model": "mlx-community/IndexTTS-2.5-fp16",
  "input": "这是需要合成的文本。",
  "ref_audio": "<Base64>",
  "response_format": "pcm",
  "stream": true
}
```

`model`, `input` and `ref_audio` are required nonempty strings; model and input
must contain non-whitespace characters. `response_format` defaults to `wav`;
`stream` defaults to `false`. The two combinations above are the supported output
modes. Unknown fields, duplicate fields and explicit nulls are rejected.

`ref_audio` contains canonical standard Base64 with padding and no whitespace.
Encode the complete audio file into that string; file paths, URLs and data URLs
are not accepted. Audio format is identified from decoded bytes. Supported inputs
are WAV (PCM 16/24/32 or IEEE float32), FLAC and MP3. The entire input file is
validated before the reference is cropped to its first 15 seconds.

Input is a complete JSON request. Automatic language selection and the fixed
synthesis profile are documented in the audio library. Streaming granularity is
a completed text segment: each segment must finish acoustic synthesis before its
PCM can be emitted.

## Responses and client handling

Both modes produce 22050 Hz, mono, signed 16-bit little-endian samples. Responses
include `X-Request-Id`, `X-Audio-Sample-Rate: 22050`, `X-Audio-Channels: 1` and
`X-Audio-Sample-Format: s16le`.

| Mode | Content-Type | Length | Cache-Control |
| --- | --- | --- | --- |
| Complete WAV | `audio/wav` | Full file `Content-Length` | `no-store` |
| PCM stream | `audio/pcm` | No `Content-Length` | `no-store, no-transform` |

PCM responses also include `X-IronMLX-Streaming-Granularity: segment`. The stream
contains raw samples without a WAV header or JSON events. Network reads can end
on any byte boundary, including an odd byte count: retain the last unmatched byte
and combine it with the next read before interpreting little-endian samples.
Transport chunks do not identify text segments. WAV and PCM use the same rounding
and saturation rules.

The server waits for the first nonempty playable chunk before sending HTTP 200.
Errors before that point use the existing JSON error envelope. Failures after
HTTP 200 abort the response body; they do not append JSON or a normal completion.
Clients must distinguish clean completion from a truncated/failed transfer and
must not automatically replay a request after playing partial audio.

Disconnecting cancels work. A running GPU kernel cannot be preempted: worker
leases, execution permits and computation reservations remain owned until a
cancellation checkpoint and final GPU synchronization. Unloading an active model
enters draining; it cannot free the model beneath its worker.

## Limits and scheduling

| Policy | Default |
| --- | --- |
| HTTP body / decoded file / decoded f32 PCM | 32 MiB / 16 MiB / 48 MiB |
| Reference | 1–60 seconds, 8–96 kHz, mono or stereo |
| Text | 65536 UTF-8 bytes, 16384 canonical tokens, 256 segments |
| Segment token budget | 120 |
| Total output, including inter-segment silence | 13230000 frames (600 seconds) |
| Per-model execution / waiting queue | 1 / 4 |
| Input preparation execution / waiting capacity | 1 / 4 |
| Output channel | 8 chunks, at most 8192 mono frames each |
| Input / model load deadline | 60 / 300 seconds |
| Queue / first audio / total execution / blocked consumer | 60 / 120 / 900 / 30 seconds |

The `audio.execution` configuration optionally accepts `queue_timeout_ms`,
`first_audio_timeout_ms`, `execution_timeout_ms`, `slow_consumer_timeout_ms`,
`max_output_frames` and `segment_tokens`. Deadlines must be positive and at most
24 hours. Output capacity may be lowered from 13230000 frames; segment budget may
be set within 6–120 tokens. First-audio and total-execution deadlines start when
the request obtains its execution permit, after queueing and model loading.
These service policies are not model hard limits or latency guarantees.

Input storage, resident model resources, temporary synthesis tensors, complete
segments and transport buffers participate in the shared process governor. The
initial conservative admission policy reserves 256 MiB per admitted input and
32 GiB per executing synthesis, with output storage and allowed allocator-cache
growth reserved separately. This can reject an otherwise feasible short request
on a memory-constrained machine. The bounded output channel alone is not the
synthesis memory budget. Reservations for outgoing bytes remain alive until the
network stack consumes or drops those bytes.

## Error classes

| Status | Examples |
| --- | --- |
| 400 | Invalid fields/Base64/audio, unsupported response combination, model task mismatch |
| 401 | Authentication failure under the existing LAN security policy |
| 404 | Unregistered or disabled model |
| 413 | Request, decoded audio or text capacity exceeded |
| 503 | Queue full, insufficient memory, unavailable resources; includes `Retry-After` |
| 504 | Queue, first-audio or execution deadline before response headers |
| 500 | Invalid weights, generation limit without completion, unexpected worker or inference failure |

After response headers, generation limits, deadlines, backpressure failure and
worker failure terminate the body with a transport error regardless of the
corresponding pre-header status.

## Developer verification

`ironmlx/tests/audio_http.rs` contains an ignored real-model HTTP acceptance test.
Its documented environment variables supply verified local resources, a reference
WAV and a local LLM snapshot. It exercises native synthesis through the actual
server process, including failure and cancellation paths. Ordinary tests do not
replace this acceptance run. Release App packaging and client playback require
their own acceptance checks.

### Standalone macOS playback client

The [Swift example](../examples/speech-client.swift) plays through the default
audio output device using AVAudioEngine. Build and run it with the Xcode command
line tools:

```sh
swiftc -parse-as-library -swift-version 6 examples/speech-client.swift -o /tmp/ironmlx-speech-client
/tmp/ironmlx-speech-client --reference reference.wav --text '这是需要合成的文本。' --format wav --output speech.wav
/tmp/ironmlx-speech-client --reference reference.wav --text '这是需要合成的文本。' --format pcm --output speech.pcm
```

The default endpoint is `http://127.0.0.1:9068/v1/audio/speech`; `--url` and
`--model` select another endpoint or registered model. Set `IRONMLX_API_KEY` when
authentication is required. Register the model resources as described above
before running the client. Dashboard resource configuration and playback are not
part of this example.

PCM playback begins during receipt, with at most two seconds queued in the
audio player. Reads may contain an odd number of bytes; the decoder preserves
the unmatched byte. WAV playback starts after the entire response and its header
have been validated. Both modes validate the format headers rather than guessing
sample rate or channel count. JSON output records frames reported played by the
audio device, time to first playback, download completion and playback completion.
These observations do not assess subjective voice quality.

The client uses macOS `/usr/bin/curl` with HTTP/1.1 to detect truncated chunked
responses; CFNetwork/URLSession may treat a missing final chunk as normal EOF.
An OS pipe applies backpressure while playback is full. The example does not
retry synthesis. Ctrl-C or `--cancel-after-ms N` stops playback and closes the
request. Transport, decoding or device errors exit nonzero; optional output is
published only after successful transfer and playback. Existing output files are
not overwritten. Temporary request files are stored in a private directory and
removed on exit.

The client tests separate wire fixtures from real model acceptance:

```sh
IRONMLX_SPEECH_CLIENT=/tmp/ironmlx-speech-client \
  IRONMLX_SPEECH_PLAYBACK_TESTS=1 \
  python3 -m unittest discover -s scripts/tests -p test_speech_client.py -v
```

The playback flag requires a working audio device. CI runs the transport checks
that do not require that device; it does not establish audible playback or real
model acceptance.

### Release Bundle acceptance

Build with `scripts/build-app-bundle.sh`, then run the static Bundle and model
distribution boundary checks. To repeat the real HTTP tests against that exact
Bundle, add the following variable to the real-resource test environment:

```sh
IRONMLX_AUDIO_HTTP_BUNDLE="$PWD/dist/IronMLX.app" \
  cargo test --locked -p ironmlx --release --test audio_http -- --ignored --nocapture --test-threads=1
```

This mode uses only the Bundle's helper and metallib, clears `MLX_*` and `DYLD_*`
overrides in the child and runs it outside the checkout. Models, derived weights,
FSTs and dictionary data remain explicit external resources; they are not bundled
with the app. Repeat client playback against the Bundle helper as a separate
check. A local ad-hoc-signed Bundle is not Developer ID signing, notarization or
authorization for public distribution.
