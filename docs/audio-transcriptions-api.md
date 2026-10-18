# Speech recognition, alignment and diarization API

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
`prompt` (context), and `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt` or `diarized_json`, default `json`).
Omitting `language` enables language detection. For standalone ASR use `json` or
`text` and omit `timestamp_granularities[]`; `verbose_json`, `srt` and `vtt` enable the two-model
workflow described below. Subtitle requests also omit `timestamp_granularities[]`. Greedy decoding is used;
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

Unsupported parameters, duplicate scalar fields, unsupported output formats and nonzero temperature
return explicit OpenAI-shaped errors. Speaker-labeled output uses the workflow documented below. `stream=true` requires `response_format=json`.

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
`segment`, while supplying it with any other response format is an error. Both kinds of timestamps come from actual
alignment; segment times span the first through last aligned unit of each ASR
window. They are not sentence segmentation or VAD boundaries. Long-file times
include each window's offset in the original recording. Plain `json`/`text` and
SSE transcription continue to use only ASR. Timestamped JSON uses nonstreaming
`verbose_json`; subtitle output uses nonstreaming `srt` or `vtt` as described below.
Realtime WebSocket timestamps are not added by this workflow.
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
`response_format` accepts `json` (default), `srt` or `vtt`. Sampling, prompts, streaming and timestamp
selection fields are not accepted. This endpoint is an IronMLX extension, not an
OpenAI or Anthropic standard API.

```sh
curl http://127.0.0.1:9068/v1/audio/alignments \
  -F model=mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
  -F file=@speech.wav -F 'text=甚至出现交易几乎停滞的情况。' -F language=zh
```

JSON returns `{text,language,duration,words:[{word,start,end}]}` with times in seconds.
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

## SRT and WebVTT subtitles

Use the standard OpenAI transcription `response_format=srt` or `response_format=vtt`.
Transcription runs the selected Qwen3-ASR and the fixed ForcedAligner; register both
checkpoints before calling. On the independent IronMLX alignment endpoint, the same
format options align supplied `text` and `language` without running ASR.
Omit `timestamp_granularities[]`, which is only valid with `verbose_json`.
`stream=true` is rejected; the server returns a complete subtitle document.

```sh
# Audio -> transcription -> alignment -> SubRip text
curl http://127.0.0.1:9068/v1/audio/transcriptions \
  -F model=mlx-community/Qwen3-ASR-0.6B-8bit \
  -F file=@speech.wav -F language=zh -F response_format=srt \
  -o speech.srt

# Audio + existing transcript -> alignment -> WebVTT text
curl http://127.0.0.1:9068/v1/audio/alignments \
  -F model=mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
  -F file=@speech.wav -F 'text=甚至出现交易几乎停滞的情况。' \
  -F language=zh -F response_format=vtt -o speech.vtt
```

| Format | Content-Type | Time notation | Header |
| --- | --- | --- | --- |
| `srt` | `application/x-subrip; charset=utf-8` | `HH:MM:SS,mmm` | No file header; numbered cues |
| `vtt` | `text/vtt; charset=utf-8` | `HH:MM:SS.mmm` | `WEBVTT` and a blank line |

The UTF-8 response body is subtitle text, without a JSON wrapper or server-side
file URL. Upstream applications can save it as `.srt` or `.vtt`. For example:

```srt
1
00:00:00,400 --> 00:00:03,680
甚至出现交易几乎停滞的情况。

```

Cues use aligned word/unit times in the original audio coordinate system. They
are rebuilt across ASR windows, rather than exposing a 25–30 second recognition
window as one caption. Defaults prefer sentence punctuation after one second,
clause punctuation after three seconds or one line, and pauses of at least 600 ms.
Targets are six seconds per cue and two lines of 42 display units each; CJK and
other full-width characters count as two units. These are formatting heuristics,
not linguistic segmentation or broadcast-standard certification. A long single
alignment unit, or merging untimed units, can exceed these targets; text is never
discarded to satisfy a limit. No extra subtitle-style request fields are introduced.

Source punctuation and letter case are retained, whitespace is normalized, and
literal `&`, `<` and `>` are escaped in cue text. Zero-duration units are merged
with timed neighbours. If nonempty speech has no positive interval at millisecond
precision, the API returns HTTP 422 with `error.code=subtitle_timing_unavailable`,
not guessed times or a partial subtitle file. Invalid/mismatched alignment returns
an OpenAI-shaped `subtitle_output_failed` server error. Empty recognized speech
returns empty SRT or a WebVTT header without placeholder cues.

Subtitle export inherits ASR errors and alignment precision, including the model's
80 ms timestamp classes and possible long-silence outliers; millisecond notation
does not improve accuracy. Existing duration, context, dependency, capacity,
deadline and cancellation rules still apply. No subtitles are added to SSE or
Realtime WebSocket responses, and no Dashboard transcription controls are added.

## Speaker diarization: Nemotron 3

`mlx-community/Nemotron-3-Diarization` (BF16, revision
`59ed2dbfc1346dcea9d423c71306a3a2499c568f`) runs natively with Rust/MLX.
It identifies up to eight generic speakers, including overlapping activity. Labels
such as `speaker_0` are recording-local, ordered by arrival; they are not personal
identities and cannot be compared across recordings. The 10 ms output frame stride
is a representation resolution, not a guarantee of boundary accuracy.
The App manages download, loading, unloading, lazy eviction and health reporting;
older verified file-only snapshots are promoted after native metadata preflight,
without downloading weights again. No recording or transcription client UI is added.
The checkpoint uses [OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization);
weights are user-downloaded and are not bundled with the App.

### Independent diarization API (IronMLX extension)

`POST /v1/audio/diarizations` accepts multipart `model` and binary `file`.
It runs only the diarization model and returns no transcript or separated audio.
WAV, FLAC and MP3, mono/stereo at 8–96 kHz are decoded and resampled to mono 16 kHz.
Limits are 128 MiB encoded audio, 512 MiB decoded PCM, and two hours per recording;
existing input-capacity, model queue, memory, deadline and cancellation gates apply.

| Optional field | Default | Accepted values |
| --- | --- | --- |
| `response_format` | `json` | `json`, `rttm` |
| `threshold` | `0.5` | Activity probability, strictly between 0 and 1 |
| `min_duration` | `0` | Minimum turn duration, 0–30 seconds; applied before gap merging |
| `merge_gap` | `0` | Merge same-speaker turns across gaps up to 0–5 seconds |
| `latency` | `offline` | `offline`, `low`, `very_low`, `ultra_low` |
| `stream` | `false` | `true` only with JSON output |

Latency presets have input buffers of 30.4 s, 1.04 s, 0.64 s and 0.32 s respectively,
excluding computation and STFT lookahead. Low latency trades context for response
speed. Speaker-cache/FIFO state remains bounded and is retained across windows.
Language, transcript text, prompts, generation sampling, speaker-name references
and a caller-specified speaker count are unsupported and return explicit errors.

```sh
curl http://127.0.0.1:9068/v1/audio/diarizations \
  -F model=mlx-community/Nemotron-3-Diarization -F file=@meeting.wav

curl http://127.0.0.1:9068/v1/audio/diarizations \
  -F model=mlx-community/Nemotron-3-Diarization -F file=@meeting.wav \
  -F response_format=rttm -o meeting.rttm
```

JSON has `{duration,num_speakers,segments:[{speaker,start,end}]}`. Times are seconds
on the original recording timeline. Segments for different speakers may overlap;
`num_speakers` counts active output labels, not an independently verified headcount.
RTTM is UTF-8 `text/plain`, with one `SPEAKER audio 1 start duration ...` line per turn.

With `stream=true`, the uploaded file produces SSE `diarization.chunk` events with
`{type,start,end,segments}` and a `diarization.done` event with `{type,result}`.
Chunk boundaries may split turns, and short-turn/gap processing is local to each
chunk. The final result applies filtering and merging over the whole recording and
is authoritative. A failed stream ends with an OpenAI-shaped `error` event.
This is uploaded-file streaming, not a microphone WebSocket protocol. The Rust
model API also exposes stateful `init_stream`/`feed`, with a final flush.

### Speaker-labeled transcription: fixed three-model workflow

Use the standard OpenAI `response_format=diarized_json` on
`POST /v1/audio/transcriptions`, selecting either supported Qwen3-ASR checkpoint.
Register that ASR, the fixed Qwen3-ForcedAligner and Nemotron checkpoint first.
Dependencies are checked before ASR inference. The server runs ASR, aligns its
text, runs diarization on the same recording and attributes words by temporal
overlap. The lazy engine-pool server may evict between steps even with one loaded-model slot.
The App preserves its manual capacity policy: allow room for all three models or
unload other models before requesting the workflow.

```sh
curl http://127.0.0.1:9068/v1/audio/transcriptions \
  -F model=mlx-community/Qwen3-ASR-0.6B-8bit -F file=@meeting.wav \
  -F language=zh -F response_format=diarized_json
```

The response uses OpenAI's `task`, `duration`, `text` and `segments`, whose items
have `type=transcript.text.segment`, `id`, `speaker`, `start`, `end`, and `text`.
IronMLX adds `speakers` and `ambiguous` per segment and the independent
`diarization` result. A label is assigned when exactly one speaker overlaps at least
20% of the aligned word. Multiple candidates produce `speaker=unknown`, their
labels in `speakers`, and `ambiguous=true`; no candidate gives `unknown` with an
empty list. This does not separate mixed voices or recover words missed by ASR.
Adjacent equal attributions are grouped while retaining transcript punctuation.
This workflow is nonstreaming; omit `timestamp_granularities[]`, which remains
exclusive to `verbose_json`. `chunking_strategy=auto` is accepted for compatibility;
IronMLX already applies its documented ASR chunking policy. Other chunking options,
known-speaker identification are unsupported. Realtime speaker labels are available through the IronMLX extension below.
Existing SRT/VTT output retains its ASR-plus-alignment behavior.

For local CLI use:

```sh
ironmlx diarize --model /path/to/Nemotron-3-Diarization --audio meeting.wav
ironmlx diarize --model /path/to/Nemotron-3-Diarization --audio meeting.wav \
  --latency low --response-format rttm
```

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
provisional-text extension above. WebRTC access is described below. It does not provide noise
reduction or audio generation. Optional aligned speaker labels are described below. Upstream applications own
microphone capture and rendering; client VAD remains optional.

### WebRTC access (OpenAI Realtime style)

WebRTC and WebSocket share transcription, VAD, optional speaker labels and managed
inference. The App remains a backend service without recording/rendering controls.
Use a browser `RTCPeerConnection` with one outgoing Opus audio track and a reliable,
ordered DataChannel named `oai-events`. Media carries audio; the data channel
carries JSON events, not Base64 audio appends.

Create a call with `POST /v1/realtime/calls` using multipart fields:

| Field | Meaning |
| --- | --- |
| `sdp` | Browser-generated SDP offer, up to 64 KiB; wait for local ICE gathering before posting |
| `session` | Optional JSON string containing the transcription session configuration |

```json
{"type":"transcription","audio":{"input":{"transcription":{"model":"mlx-community/Qwen3-ASR-0.6B-8bit","language":"zh"},"turn_detection":{"type":"server_vad"},"diarization":{}}}}
```

Omitting `diarization` disables it initially; `{}` enables defaults and `null`
disables it. The optional `model`, `threshold` and `latency` fields match the
speaker-label extension below. Auxiliary models remain the fixed ForcedAligner
and Nemotron pair, requiring capacity for three models when enabled.
`type: "realtime"` is also accepted and normalized to a transcription-only session;
no conversational generation or speech output is provided. Put the ASR model in
`audio.input.transcription.model`. Server VAD is enabled initially; explicitly
set `turn_detection: null` for manual commits.

The response is **201 Created** with `Content-Type: application/sdp` and the SDP
answer as its body. Apply it through `setRemoteDescription`. The `Location` header
identifies `/v1/realtime/calls/{call_id}`. Once the DataChannel connects it receives
`session.created`, followed by the existing transcription, VAD and speaker events.
Raw `application/sdp` offers are also accepted with a `model` URL parameter;
configure other options with `session.update` before sending speech.
SDP negotiates Opus with a 48 kHz RTP clock and its two-channel format identifier;
the backend decodes to mono 24 kHz PCM. PCM in the session view describes the
internal inference input, not the negotiated media codec. Video, other codecs,
multiple audio tracks and multiple data channels are rejected.

```sh
curl http://127.0.0.1:9068/v1/realtime/calls \
  -F 'sdp=<offer.sdp' \
  -F 'session={"audio":{"input":{"transcription":{"model":"mlx-community/Qwen3-ASR-0.6B-8bit","language":"zh"}}}}' \
  -D call-headers.txt -o answer.sdp
curl -X POST http://127.0.0.1:9068/v1/realtime/calls/CALL_ID/hangup
```

`POST /v1/realtime/calls/{call_id}/hangup` returns 200 and closes/cancels the call;
an absent or cleaned-up call returns 404. Hangup does not wait for final results.
Keep sending silence for VAD to finish, or send `input_audio_buffer.commit` and
wait for the final `completed` item (or `diarized` with labels enabled), then
hang up. Media and DataChannel events have no shared ordering: leave time for
media to arrive before a manual commit. The server drains its received jitter
buffer before committing. Clear also discards pending reordered audio and resets
the transcription/speaker context. Pause RTP and wait for pending audio/labels to
finish before updating configuration, then resume after the update acknowledgement.

Browsers require HTTPS or localhost. LAN access uses the existing HTTPS/Bearer
configuration. CORS for the call endpoints allows localhost/loopback and secure
origins on the serving interface by default. Configure additional exact HTTPS
origins through comma-separated `IRONMLX_WEBRTC_ALLOWED_ORIGINS`, for example
`https://speech.example.com`. Send a Bearer header directly or proxy SDP through
an upstream trusted server to keep a long-lived key out of browser code.
No ephemeral-key issuer or SIP endpoints are added.

Media UDP sockets bind the API interface receiving the request. No external ICE
service is contacted by default. `IRONMLX_WEBRTC_ICE_SERVERS` accepts an ICE server
JSON array for backend STUN/TURN gathering:

```json
[{"urls":["stun:stun.example.com:3478"]},{"urls":["turn:turn.example.com:3478"],"username":"USER","credential":"PASSWORD"}]
```

`IRONMLX_WEBRTC_ICE_POLICY` defaults to `all`; set `relay` to require TURN relay candidates on the backend.
If gathering produces no usable candidates, creation fails with `webrtc_ice_unavailable` (503).
Configure browser STUN/TURN separately through `RTCPeerConnection({iceServers})`.
Cross-NAT deployments need reachable STUN/TURN or appropriate forwarding; IronMLX
does not bundle those servers. HTTPS signaling and media UDP use separate ports.
The transport exchanges complete SDP once; PATCH trickle ICE and in-call
renegotiation are unsupported. Create a new call after a network-interface change.

WebRTC and WebSocket share an eight-session limit, two-hour connection/input
limits and the 90-second input idle timeout. Call bodies are limited to 128 KiB;
ICE gathering to 20 seconds, total creation to 30 seconds and waiting for the
DataChannel after returning SDP to 30 seconds. Each DataChannel message is at
most 256 KiB; send buffering is bounded and slow receivers time out. The transport
queues at most 128 pending input messages. RTP reordering retains at most 64
packets with a normal 60 ms wait; loss/DTX gaps retain their duration as silence
instead of shifting later text earlier. RTP jumps over 60 seconds, decode failures
and overflow produce explicit errors and end the call. Buffer/model windows are
not end-to-end latency guarantees.

The upstream browser module [`examples/webrtc-transcription-client.js`](../examples/webrtc-transcription-client.js)
handles microphone capture, SDP exchange and hangup:

```javascript
import { connectTranscription } from './webrtc-transcription-client.js';
const client = await connectTranscription({
  baseUrl: 'http://127.0.0.1:9068', language: 'zh', diarization: true,
  onEvent: event => console.log(event)
});
// Wait for the final results you need first; hangup cancels remaining work.
await client.hangup();
```

Rendering remains upstream. The backend uses a native WebRTC stack and statically
linked libopus; it requires no FFmpeg, Python or browser process. Browsers are only
needed by upstream clients and development acceptance tests.

### Realtime speaker labels (IronMLX extension)

Set an object in `session.audio.input.diarization` on a WebSocket or WebRTC
transcription session to opt in; the default is disabled and `null` disables it. This field
and the following three new events are IronMLX extensions, not OpenAI fields.
Register and prepare Qwen3-ASR, the fixed Qwen3-ForcedAligner-0.6B-8bit and
`mlx-community/Nemotron-3-Diarization`, with capacity for at least three loaded
models. The App remains a backend service without microphone or transcription controls.

```json
{"type":"session.update","session":{"audio":{"input":{"transcription":{"language":"zh"},"diarization":{"model":"mlx-community/Nemotron-3-Diarization","threshold":0.5,"latency":"low"}}}}}
```

`diarization: {}` uses these defaults. `threshold` must be strictly between 0 and 1.
`latency` accepts `low`, `very_low` or `ultra_low`: model input windows are about
1.04, 0.64 or 0.32 seconds, not end-to-end latency guarantees. Realtime mode rejects
`offline`, `min_duration`, `merge_gap`, requested speaker counts and known-speaker
references. Only this fixed auxiliary-model combination is supported.
Both manual commit and server VAD work. Send silence as part of the continuous
original audio. Conversion from 24 kHz to 16 kHz retains sinc filter context and
phase across PCM packet boundaries.

| New event | Meaning and client action |
| --- | --- |
| `input_audio_buffer.diarization.started` | Supplies `diarization_id` and `audio_start_ms` after enable/reset; scope speaker labels to that context ID |
| `input_audio_buffer.diarization.delta` | Supplies continuous `start`, `end` and activity `segments` in global accepted-audio seconds; chunks can be silent or overlapping and adjacent chunks can share a continuing speaker |
| `conversation.item.input_audio_transcription.diarized` | Supplies confirmed text with aligned speaker labels; associate it with the original `completed` event by `item_id` and render its `segments` |

Existing `partial`, `delta` and `completed` semantics are unchanged. Interim text
does not receive speculative labels. With diarization enabled, `completed` means
text is ready; also wait for `diarized` for the same item. It includes `content_index`,
`diarization_id`, `transcript`, `language`, `is_final`, global `start/end`, text
`segments` and independent activity `diarization`. Segment `speaker`, `speakers`
and `ambiguous` follow the file workflow: ambiguous overlap retains candidate
labels and returns `unknown`, without claiming to separate overlapping voices.
Concatenating segment text reproduces the item's transcript; segment times are
global seconds as well.

```json
{"type":"conversation.item.input_audio_transcription.diarized","item_id":"item_...","content_index":0,"diarization_id":"diar_...","transcript":"Hello.","language":"English","is_final":true,"start":2.0,"end":4.0,"segments":[{"type":"transcript.text.segment","id":"seg_0","speaker":"speaker_0","speakers":["speaker_0"],"ambiguous":false,"start":2.4,"end":3.2,"text":"Hello."}],"diarization":{"duration":2.0,"num_speakers":1,"segments":[{"speaker":"speaker_0","start":2.3,"end":3.4}]}}
```

Speaker caches survive explicit commits, VAD turns and 30-second rolling windows
within one context. Turn completion flushes a copy with available right context,
leaving the live stream unchanged. Confirmed labels use the audio snapshot at
that time and are not revised by later audio; online results can differ from a
whole-recording offline pass. Render confirmed text from `diarized.segments` and
use activity events for a separate activity timeline. Labels are model-assigned
session identifiers, not verified personal identities or guaranteed recognition.

Clearing cancels current/queued work, discards outstanding annotations and creates
a new `diarization_id`; the global accepted-audio timeline keeps advancing.
Changing diarization settings also starts a new context. Keeping those settings
unchanged while updating language, prompt or VAD preserves the context.
Session updates require all audio and annotations to finish; invalid settings leave
the old configuration intact. Each connection bounds pending audio to 60 seconds,
annotations to 16 items and 60 seconds of PCM, and activity history to 90 seconds.
Overload rejects appends before accepting their audio. Inference/alignment failure
emits `realtime_diarization_failed` and ends the connection; disconnect cancels work
and releases model leases.

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


Add `--diarization true` to enable Realtime speaker labels. The client waits for
the final `diarized` item and writes speaker-labelled text events to stdout, with
activity events on stderr. Prepare all three models first.

Use `--model` for the 1.7B model and `--url` for another realtime endpoint.
LAN Bearer authentication reads `IRONMLX_API_KEY` from the environment rather than
command-line arguments. Committing an empty buffer remains a request error.

Model listings report `runtime_kind: "asr"`, audio input and streaming support.
Existing unload, TTL, model-count and process-memory policies apply; causal
scheduler/KV policy, MTP and PromptLookup controls do not apply to ASR.
