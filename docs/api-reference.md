# API reference

[简体中文](zh-CN/api-reference.md) · [Developer guide](developer-guide.md#api-integration)

Choose an API topic by capability or look up a specific route in the endpoint index. Each topic defines request fields, responses, errors and limits. See the [Developer guide](developer-guide.md#api-quick-start) for a first request.

## API topics

| Topic | Scope |
| --- | --- |
| [Service and management API](service-api.md) | Shared access conventions, health, model discovery and management |
| [Text and vision API](text-vision-api.md) | Responses, Chat Completions and Messages; text and image understanding, reasoning, tools and structured outputs |
| [Embedding API](text-embeddings.md) | Text, image, audio and combined inputs; output dimensions and encoding |
| [Speech synthesis API](audio-speech-api.md) | Speech output, voice profiles and reference recordings |
| [Speech recognition and alignment API](audio-transcriptions-api.md) | Standalone transcription, forced alignment, timestamps and SRT/WebVTT subtitles |
| [Image generation API](image-generation-api.md) | Text-to-image generation and single-image conditional editing |
| [System One API](laya-systemone-api.md) | Laya choice, score and true/false probability requests |
| [OpenAI Decisions API](openai-decisions-api.md) | Laya predicate, choice and score answers through the official OpenAI SDK |

## Endpoint index

Availability depends on the serving mode and configured compatible models. This table covers endpoints described in the public documentation; App model-management and EnginePool model-control routes use different paths.

| Method | Endpoint | Purpose | Availability | Reference |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | HTTP responsiveness | App / EnginePool / text CLI / DFlash2 | [Service and management API](service-api.md#get-health) |
| `GET` | `/healthz` | Runtime health | App / EnginePool / CLI | [Service and management API](service-api.md#get-healthz) |
| `GET` | `/v1/models` | Model discovery | App / EnginePool / DFlash2 / System One CLI | [Service and management API](service-api.md#get-v1models) |
| `POST` | `/v1/responses` | Responses inference | App / EnginePool / CLI; compatible model | [Text and vision API](text-vision-api.md#openai-responses) |
| `POST` | `/v1/chat/completions` | Chat Completions inference | App / EnginePool / CLI; compatible model | [Text and vision API](text-vision-api.md#openai-chat-completions) |
| `POST` | `/v1/messages` | Messages inference | App / EnginePool / CLI; compatible model | [Text and vision API](text-vision-api.md#anthropic-messages) |
| `POST` | `/v1/embeddings` | Text, image, audio and combined vectors | App / EnginePool; EmbeddingGemma 2 | [Embedding API](text-embeddings.md#api) |
| `POST` | `/v1/audio/speech` | Speech synthesis | App / EnginePool; audio model | [Speech synthesis API](audio-speech-api.md#request) |
| `POST` | `/v1/audio/transcriptions` | Speech recognition; aligned timestamps or SRT/WebVTT output | App / EnginePool; ASR model | [Speech recognition API](audio-transcriptions-api.md) |
| `POST` | `/v1/audio/alignments` | Align supplied text with audio; JSON or SRT/WebVTT (IronMLX extension) | App / EnginePool; ForcedAligner model | [Independent forced alignment](audio-transcriptions-api.md#independent-forced-alignment-ironmlx-extension) |
| `POST` | `/v1/audio/diarizations` | Speaker activity; JSON, RTTM or file SSE (IronMLX extension) | App / EnginePool; Nemotron 3 Diarization | [API](audio-transcriptions-api.md) |
| `GET` | `/v1/realtime` | Realtime transcription with optional aligned speaker labels | App / EnginePool; ASR WebSocket sessions | [Realtime](audio-transcriptions-api.md) |
| `POST` | `/v1/realtime/calls` | Establish WebRTC audio and DataChannel transport; optional speaker labels | App / EnginePool; ASR sessions | [Realtime API](audio-transcriptions-api.md) |
| `POST` | `/v1/realtime/calls/{call_id}/hangup` | Close and cancel a WebRTC call | App / EnginePool | [Realtime API](audio-transcriptions-api.md) |
| `GET` | `/v1/audio/voices` | List enabled voices | App / EnginePool; configured voice service | [Speech synthesis API](audio-speech-api.md) |
| `GET` | `/admin/api/audio/voices` | List all voices, including disabled profiles | App / EnginePool; configured voice service | [Speech synthesis API](audio-speech-api.md) |
| `POST` | `/v1/audio/voices` | Create a voice | App / EnginePool; configured voice service | [Speech synthesis API](audio-speech-api.md) |
| `PATCH` | `/v1/audio/voices/{id}` | Update a voice | App / EnginePool; configured voice service | [Speech synthesis API](audio-speech-api.md) |
| `DELETE` | `/v1/audio/voices/{id}` | Delete a voice | App / EnginePool; configured voice service | [Speech synthesis API](audio-speech-api.md) |
| `GET` | `/v1/audio/voices/{id}/preview` | Reference recording preview | App / EnginePool; configured voice service | [Speech synthesis API](audio-speech-api.md) |
| `POST` | `/v1/images/generations` | Text-to-image generation | App / EnginePool; image model | [Image generation API](image-generation-api.md) |
| `POST` | `/v1/images/edits` | Single-image conditional editing | App / EnginePool; image model | [Image generation API](image-generation-api.md) |
| `POST` | `/v1/systemone` | Typed decision inference | App / EnginePool / System One CLI; decision model | [System One API](laya-systemone-api.md#endpoints) |
| `POST` | `/v1/decisions` | OpenAI Decisions inference | App / EnginePool / System One CLI; decision model | [OpenAI Decisions API](openai-decisions-api.md#request) |
| `POST` | `/v1/models/{model_id}/load` | Load a model | EnginePool | [EnginePool control](engine-pool.md#http-apis) |
| `POST` | `/v1/models/{model_id}/unload` | Unload a model | EnginePool | [EnginePool control](engine-pool.md#http-apis) |
| `GET` | `/admin/api/models/loaded` | Loaded models and metrics | App | [Service and management API](service-api.md#management-endpoints) |
| `GET` | `/admin/api/log-level` | Read the log level | Loopback only | [Service and management API](service-api.md#log-level-management-loopback-only) |
| `POST` | `/admin/api/log-level` | Update the log level | Loopback only | [Service and management API](service-api.md#log-level-management-loopback-only) |
| `POST` | `/admin/api/models/register` | Register local model resources | App model-management daemon | [Audio resource registration example](audio-speech-api.md#register-local-resources) |
| `POST` | `/admin/api/models/load` | Load a local model | App model-management daemon | [Audio resource registration example](audio-speech-api.md#register-local-resources) |

For LAN authentication, see [Service and management API](service-api.md#service-address-authentication-and-conventions). Standalone System One authentication and discovery shapes are defined in its [CLI section](laya-systemone-api.md#optional-standalone-cli).

## Agent integrations

Follow [Hermes Agent](hermes-agent.md), [oh-my-pi](oh-my-pi.md), or [DeepSeek Harness (DSH)](dsh.md) for application setup.
