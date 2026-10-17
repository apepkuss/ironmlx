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
`prompt` (context), and `response_format` (`json`, `text` or `verbose_json`, default `json`).
Omitting `language` enables language detection. For standalone ASR use `json` or
`text` and omit `timestamp_granularities[]`; `verbose_json` enables the two-model
workflow described below. Greedy decoding is used;
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

Unsupported parameters, duplicate scalar fields, subtitles and nonzero temperature
return explicit OpenAI-shaped errors. Speaker diarization is not implemented. `stream=true` requires `response_format=json`.

## Timestamped transcription: fixed two-model workflow

Send `response_format=verbose_json` to the existing transcription endpoint.
`model` selects either supported Qwen3-ASR size. IronMLX transcribes first, then
aligns each recognized audio/text segment with the fixed auxiliary checkpoint
`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`. Download and register this checkpoint
in advance; requests do not download weights. There is no `alignment_model` option
or arbitrary model-composition interface in this version.

```sh
curl http://127.0.0.1:9068/v1/audio/transcriptions \
  -F model=mlx-community/Qwen3-ASR-0.6B-8bit -F file=@speech.wav \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word' -F 'timestamp_granularities[]=segment'
```

The response contains `task`, `text`, `language`, `duration` (seconds), and the
requested `words: [{word,start,end}]` and/or `segments: [{id,text,start,end}]`.
`timestamp_granularities[]` is an array field: `word` requests word/unit times and
`segment` requests segment times. Repeat the same multipart field to request both,
as in the example above. It requires `verbose_json`; omitting it defaults to
`segment`, while supplying it with `json` or `text` is an error. Both kinds of timestamps come from actual
alignment; segment times span the first through last aligned unit of each ASR
window. They are not sentence segmentation or VAD boundaries. Long-file times
include each window's offset in the original recording. Plain `json`/`text` and
SSE transcription continue to use only ASR; timestamp output requires nonstreaming
`verbose_json`. Realtime WebSocket timestamps are not added by this workflow.
Field names follow the [OpenAI transcription contract](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create);
Whisper-specific token scores and confidence fields are not synthesized.

Missing dependencies, unsupported alignment languages, capacity exhaustion or an
alignment failure return an error for the whole request, rather than a successful
response with fabricated or missing timestamps. The ASR lease is released before
alignment. Existing model-count, pinning and memory policies still apply: the App's
reject-on-capacity policy normally needs two available model slots, while a lazy
model-pool server can evict an idle unpinned model even with one loaded-model slot.

## Independent forced alignment (IronMLX extension)

`POST /v1/audio/alignments` accepts multipart fields `model`, `file`, `text`, and
`language`; all four are required. `model` identifies the registered ForcedAligner
checkpoint. This performs alignment only and does not run an ASR model.
`response_format=json` is optional. Sampling, prompts, streaming and timestamp
selection fields are not accepted. This endpoint is an IronMLX extension, not an
OpenAI or Anthropic standard API.

```sh
curl http://127.0.0.1:9068/v1/audio/alignments \
  -F model=mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
  -F file=@speech.wav -F 'text=甚至出现交易几乎停滞的情况。' -F language=zh
```

It returns `{text,language,duration,words:[{word,start,end}]}` with times in seconds.
Standalone audio is bounded to 300 seconds, text to 16,384 UTF-8 bytes and 2,048
spoken units, with an 8,192-token combined audio/text context. Split longer
standalone inputs into matching audio/text pairs upstream; audio cannot be split
without knowing which supplied words belong to each part. The combined workflow
handles this using the ASR window transcripts and retains the ASR duration setting.
Other file, upload, queue, cancellation and memory limits are shared with transcription.

The checkpoint supports Chinese, Cantonese, English, German, Spanish, French,
Italian, Portuguese, Russian, Japanese and Korean. Languages outside that set
remain available for plain ASR but cannot use this fixed alignment workflow.
Chinese ideographs are individual units; space-delimited text uses cleaned words.
Japanese uses character units and Korean uses whitespace units in this native
implementation, rather than the upstream nagisa/soynlp word segmentation. The
`words` field therefore does not promise linguistic word boundaries for every language.
Punctuation is preserved in the top-level text but receives no separate timestamp.
The model predicts 80 ms timestamp classes; decimal seconds do not imply finer
precision. It aligns supplied text and does not certify that the text is correct.

The App reports this model as `runtime_kind: "alignment"`, with audio input and
without streaming, chat defaults, generation sampling, or output-token settings.
Existing verified file-only snapshots are promoted by successful native preflight
when loaded, retaining downloaded weights. No recording or transcription UI is added.

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
provisional-text extension above. It does not provide WebRTC, noise
reduction, audio generation, timestamps or diarization. Upstream applications own
microphone capture and rendering; client VAD remains optional.

### Server VAD

Opt in with `session.audio.input.turn_detection` (or the beta top-level
`session.turn_detection`):

```json
{"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500}}}}}
```

Omitted configuration preserves the current mode; the initial mode is manual
(`null`). Set `turn_detection: null` between turns to disable VAD. Threshold is
0–1, prefix padding is 0–1,000 ms, and silence duration is 100–3,000 ms.
Earshot 1.2.2 performs local CPU detection on 16 ms frames using its embedded
classifier; no external model download is needed. Detector resampling does not
alter the original 24 kHz PCM sent to ASR. This is acoustic VAD, not semantic VAD.

The server emits `input_audio_buffer.speech_started` with `item_id` and
`audio_start_ms`, then `input_audio_buffer.speech_stopped` with the same ID and
`audio_end_ms`. Times refer to the accepted audio timeline, including prefix and
trailing silence; these are speech boundaries, not word timestamps. Audio must
continue carrying silence for silence detection: a pause in network delivery is
not silence. A stopped turn automatically emits `input_audio_buffer.committed`
and completes transcription with `is_final: true`, without a client commit.
Subsequent speech can queue while previous inference finishes; total queued PCM
remains bounded by 60 seconds. Continuous speech is automatically split at about
28 seconds to keep each ASR window bounded, even without a silent boundary.

`input_audio_buffer.commit` can still end active speech manually; partial detector
frames are preserved. Committing without active speech is an error. Clear cancels
pending and queued turns and resets detector state; disconnect cancels all work.
Per-model audio duration and output budgets still apply. Exceeding a VAD turn's
audio duration reports an error and closes the session, cancelling pending work.
Noise reduction, `semantic_vad`, and conversation response/interrupt options
are not supported. These settings belong to API sessions; the App adds no task UI.

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
