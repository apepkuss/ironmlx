# Speech recognition API

[简体中文](zh-CN/audio-transcriptions-api.md)

IronMLX exposes OpenAI-compatible `POST /v1/audio/transcriptions` on both the
managed App backend and the model-pool server. Recognition runs locally through
the native Rust/MLX audio engine. Supported checkpoints are
`mlx-community/Qwen3-ASR-0.6B-8bit` and `mlx-community/Qwen3-ASR-1.7B-8bit`.

## App service

Download or import either checkpoint in Models and load it to serve requests.
The App manages models and inference service status. Upstream applications send
audio, language and optional vocabulary context to the transcription API and
consume the returned text. The Dashboard does not provide a transcription client.
Registered models can also load on demand through API requests.

Previously downloaded file-only snapshots are revalidated by native metadata
preflight before promotion; model weights are retained. ASR models cannot become
the default chat-generation model.

## Local CLI

```sh
ironmlx transcribe --model /absolute/path/to/snapshot --audio speech.wav --language zh

ironmlx serve --model /absolute/path/to/snapshot \
  --model-id mlx-community/Qwen3-ASR-0.6B-8bit
```

For the App daemon, the existing `/admin/api/models/register` and
`/admin/api/models/load` endpoints take `model` and `model_dir`. ASR requires no
IndexTTS `audio` resources or auxiliary downloads. Pool manifest entries use the
same `id`, `path`, and `load_policy` fields as other managed models.

## Request and response

Use `multipart/form-data` with required `model` and binary `file` fields.
Optional fields are `language` (language code or supported language name),
`prompt` (context), and `response_format` (`json` or `text`, default `json`).
Omitting `language` enables language detection. Greedy decoding is used;
`temperature=0` is accepted. Set `stream=true` with JSON output for SSE events
`transcript.text.delta` and `transcript.text.done`; deltas concatenate to the final text.

```sh
curl http://127.0.0.1:9068/v1/audio/transcriptions \
  -F model=mlx-community/Qwen3-ASR-0.6B-8bit \
  -F file=@speech.wav \
  -F language=zh
```

```json
{"text":"甚至出现交易几乎停滞的情况。"}
```

The OpenAI Python SDK can use `client.audio.transcriptions.create` with IronMLX's
`base_url` and the local model ID. Existing network-mode authentication applies.
Anthropic Messages is not an ASR transport.

## Long recordings and lifecycle

The server automatically splits recordings near low-energy boundaries into contiguous
25–30 second segments, recognizes each segment, and combines text without cropping,
overlap or missing samples. English segments receive word spacing; CJK segments join
directly. Each segment defaults to a 1,024-token generation budget. A generation-limit
failure is reported rather than silently returning truncated text.
Realtime prefix tokens count toward the same per-segment budget.

The App model parameter page saves independent ASR settings for each model and
restores them to the backend after restart. Changes apply to new requests and
realtime sessions; in-flight work keeps its settings snapshot.

| Model | Default maximum audio duration | Default maximum output tokens per segment |
| --- | --- | --- |
| Qwen3-ASR-0.6B-8bit | 1,200 seconds | 1,024 |
| Qwen3-ASR-1.7B-8bit | 1,200 seconds | 1,024 |

The two sizes have no different official limits, so they start with the same
policy and can be configured independently. Duration uses the
[official chunk policy](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py)
as a reference; the output budget uses the [official evaluation configuration](https://github.com/QwenLM/Qwen3-ASR#evaluation).
These are IronMLX policy defaults, not model hard limits. Allowed values are
1–7,200 seconds and 1–8,192 tokens. Duration bounds one file request or one
explicitly committed realtime turn; tokens bound each recognition segment,
not the complete merged recording. Clear a field to use the model default.
Increase the model duration setting before submitting files longer than 20 minutes;
automatic segmentation remains enabled.

The App applies settings through `POST /admin/api/models/asr` with
`{"model":"MODEL_ID","settings":{"max_audio_duration_seconds":1200,"max_output_tokens":1024}}`.
`GET /admin/api/models/asr?model=MODEL_ID` reports effective settings and defaults.
This App-managed control endpoint does not load weights, reload models, or change
in-flight requests. The App persists the values in its model parameter store.

- WAV PCM/float, FLAC and MP3; the per-model duration setting applies, up to the two-hour service ceiling, 128 MiB encoded and 512 MiB decoded PCM.
  The decoded byte limit can bind before the duration limit for high-rate/stereo files.
- Mono/stereo, 8–96 kHz; native conversion to 16 kHz mono.
- Context: at most 16,384 UTF-8 bytes.
- Four waiting GPU jobs per model; long files yield the worker between segments so
  live transcription can run while a file is being processed.
- Upload timeout 60 seconds; processing deadline 180 seconds plus twice the recording duration.
- Disconnects cancel work at computation checkpoints. Model leases and memory
  reservations remain held until submitted GPU operations finish.

Unsupported parameters, duplicate fields, subtitles, timestamps and nonzero temperature
return explicit OpenAI-shaped errors. Forced alignment and speaker diarization are
not implemented. `stream=true` requires `response_format=json`.

## Realtime transcription

Upstream applications connect to
`ws://127.0.0.1:9068/v1/realtime?intent=transcription&model=mlx-community/Qwen3-ASR-0.6B-8bit`.
Use the existing authenticated `wss` LAN endpoint and Bearer header when in LAN mode.
Both App-managed and model-pool servers expose this route. The App has no recording UI.

The server sends `session.created`. Configure a transcription-only session:

```json
{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"mlx-community/Qwen3-ASR-0.6B-8bit","language":"zh"},"turn_detection":null}}}}
```

`prompt` is optional. Omit `language` for detection; a single-element `languages`
array is also accepted. Unknown processing options are rejected. Configuration can
change between turns. Beta `transcription_session.update` and PCM16 field names are
accepted as an alternative to the nested session schema.

Send base64-encoded **24 kHz mono signed PCM16 little-endian samples**:

```json
{"type":"input_audio_buffer.append","audio":"BASE64_PCM16"}
```

Send chunks of up to four seconds, preferably shorter audio frames. Recognition
runs every two seconds of new audio, before an explicit commit. It re-encodes the
current audio window and uses Qwen's five-token rollback prefix after the initial
updates. Session memory stays bounded; continuous input automatically completes
an item at a quiet boundary within 30 seconds and begins the next item. Boundaries
never precede audio already represented by the recognition prefix.

Consume these events by `item_id`:

| Event | Client behavior |
| --- | --- |
| `conversation.item.input_audio_transcription.partial` | IronMLX extension: replace the current provisional text with `transcript`; `revision` increases within the item |
| `conversation.item.input_audio_transcription.delta` | Append `delta`, which contains stable text |
| `conversation.item.input_audio_transcription.completed` | Use `transcript` as the authoritative final text |
| `conversation.item.input_audio_transcription.failed` | Handle the item error; no successful final transcript is implied |

Do not append provisional snapshots and stable deltas to the same text buffer.
Use `input_audio_buffer.commit` at the end of a speech turn (at least 100 ms of audio).
It emits `input_audio_buffer.committed` and a final transcript, then keeps the socket
open for another turn. The IronMLX `is_final` extension on `completed` is `false`
for automatic windows and `true` for the last item of an explicitly committed turn.
If several windows remain buffered at commit, wait for this marker before stopping
reception. `input_audio_buffer.clear` cancels/discards the pending turn
and emits `input_audio_buffer.cleared`. Closing the connection cancels in-flight work.

There are at most eight realtime connections per process, 60 seconds of buffered
audio per connection, 512 KiB per WebSocket message, a 90-second idle timeout and a
two-hour connection/audio limit. Backpressure and queue exhaustion produce explicit
errors. Active sessions retain their model lease. Timing depends on hardware and
concurrent inference; a socket connection alone is not a latency guarantee.

The protocol implements the transcription subset of OpenAI Realtime, with the
provisional-text extension above. It does not provide WebRTC, server VAD, noise
reduction, audio generation, timestamps or diarization. Upstream applications own
microphone capture, client VAD and rendering.

An executable upstream example is [`examples/transcription-client.swift`](../examples/transcription-client.swift).
It sends raw PCM at audio cadence while receiving provisional results, and writes
completed events as JSON lines to stdout. Input must be headerless PCM16 little-endian,
mono, 24 kHz. Convert the source in the upstream application, for example:

```sh
ffmpeg -i speech.wav -ar 24000 -ac 1 -f s16le speech.pcm
swiftc -parse-as-library examples/transcription-client.swift -o /tmp/ironmlx-transcription-client
/tmp/ironmlx-transcription-client --audio speech.pcm --language zh
```

Use `--model` for the 1.7B model and `--url` for another realtime endpoint.
LAN Bearer authentication reads `IRONMLX_API_KEY` from the environment rather than
command-line arguments. Committing an empty buffer remains a request error.

Model listings report `runtime_kind: "asr"`, audio input and streaming support.
Existing unload, TTL, model-count and process-memory policies apply; causal
scheduler/KV policy, MTP and PromptLookup controls do not apply to ASR.
