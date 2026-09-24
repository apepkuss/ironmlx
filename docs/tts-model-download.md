# TTS model download and usage

[简体中文](zh-CN/tts-model-download.md) · [Speech API](audio-speech-api.md)

IronMLX.app supports the verified `mlx-community/IndexTTS-2.5-fp16`
resource profile. The App prepares the inference resources automatically; Python,
PyTorch and manually configured dependency paths are not required. Other TTS
layouts and model revisions require separate integration.

## Usage

1. Search for and download `mlx-community/IndexTTS-2.5-fp16` in the Dashboard. The App selects the supported immutable revision.
2. Wait for the main model download, auxiliary resource preparation and integrity checks to finish, then load the model from the model list.
3. Import reference audio and assign a stable voice ID on the Dashboard **Voices** page, or provide the reference audio directly as Base64 in the request.
4. Call `/v1/audio/speech` with either `voice` or `ref_audio`. Request a complete WAV response, or use `stream: true` with `response_format: pcm` for streaming audio.

The App saves the resource configuration and restores previously loaded or pinned
models after restart. Voice profiles are stored under
`~/.ironmlx/audio/voices/` by default; profile metadata refers only to audio files
inside the managed directory. OpenAI-compatible clients can use the stable voice
IDs, while `GET /v1/audio/voices` returns the enabled voices. Initial preparation
requires network access; the complete prepared resources can then be used offline.
Synthesized audio is played by the caller or the
[standalone example client](../examples/speech-client.swift). The Dashboard can
preview the reference recording stored in a voice profile. See the
[speech API](audio-speech-api.md) for request fields and complete examples.

If the main model was downloaded before auxiliary resource support was available,
choose **Prepare resources** from the model list. Preparation can be retried after
an interruption; verified model files and download caches are reused.

## IndexTTS 2.5 repository contract

The supported reference is
[`mlx-community/IndexTTS-2.5-fp16`](https://huggingface.co/mlx-community/IndexTTS-2.5-fp16/tree/65644cd70da15309ffeb74aa03f3686bb04e3eb1):

| File | Purpose |
| --- | --- |
| `config.json`, `config.yaml` | Model configuration |
| `model_manifest.json`, `conversion_report.json` | Component manifest and conversion record |
| `gpt.safetensors` | UnifiedVoice GPT |
| `codec.safetensors` | Speech codec |
| `s2mel.safetensors` | S2Mel |
| `bigvgan.safetensors` | Vocoder |
| `model.safetensors` | w2v-BERT speech frontend, not the GPT weights |
| `multilingual_zh_ja_yue_char_del.tiktoken` | tiktoken vocabulary; no `tokenizer.json` |
| `feat1.pt`, `feat2.pt`, `wav2vec2bert_stats.pt` | Timbre, emotion and feature statistics; extracted through fixed storage mappings without executing pickle |
| `README.md`, `LICENSE*` | Repository documentation and licenses |

Recognition requires `model_family=IndexTTS`, `model_version=2.5` and
`format_version=1` in the manifest, cross-checked against the configuration version
and vocabulary path. The four component filenames come from the manifest, and
their sizes must match the remote inventory for the pinned revision. Missing
components, vocabulary or auxiliary files are rejected before weight download so
an incomplete snapshot is never published.

The App reuses its pinned commit, SHA-256 / Git blob verification, resumable
downloads, disk reservation, download queue, journal and atomic publication flow.
Model files are stored under:

```text
~/.ironmlx/models/huggingface/<owner>--<repo>/snapshots/<commit>/
```

## Automatic resource preparation

`.ironmlx-snapshot.json` records `model_type=indextts2_5`,
`artifact_role=tts` and the complete download inventory. The inference resource
profile uses pinned versions and SHA-256 checks for:

- The five safetensors components, configuration and vocabulary from the main model.
- CAMPPlus auxiliary weights and the w2v-BERT configuration, reusing the w2v-BERT weights already present in the main model.
- Natively rebuilt timbre, emotion, statistics and CAMPPlus safetensors.
- WeText 0.1.2 FST data, the UniDic-lite 1.0.8 dictionary, provenance records and licenses.

Auxiliary download caches and prepared resources are stored under
`~/.ironmlx/audio/indextts25/`. The App verifies pinned inputs and outputs, prepares
them in a temporary directory, then publishes the complete directory. It does not
modify the model snapshot or execute Python or pickle from downloaded packages.
Resumable download caches remain available for retries, while cancellation or
failure never publishes a ready state.

Model-file integrity and inference-resource readiness are checked independently.
Missing or changed resources are shown as not ready; the model becomes loadable
after preparation succeeds. The native loader verifies the resources again when
loading them.

## Verification

Ordinary regression checks do not require model files:

```sh
swift test -c release --package-path ironmlx-app
node scripts/tests/test-dashboard-tts.mjs
```

Real download acceptance uses an isolated data directory and the Release download
helper explicitly:

```sh
IRONMLX_TEST_DOWNLOAD_INDEXTTS=1 \
IRONMLX_TEST_DOWNLOAD_ROOT=/absolute/path/to/isolated-data \
IRONMLX_TEST_DOWNLOAD_BACKEND=/absolute/path/to/IronMLX.app/Contents/Helpers/ironmlx \
swift test -c release --package-path ironmlx-app --filter indexTTSLiveDownloadUsingAppService
```

Then use the same isolated directory to verify saved App configuration, model
loading, backend restart recovery and the WAV/PCM endpoints:

```sh
IRONMLX_TEST_DOWNLOAD_ROOT=/absolute/path/to/isolated-data \
IRONMLX_AUDIO_APP_BUNDLE=/absolute/path/to/IronMLX.app \
IRONMLX_AUDIO_APP_REFERENCE=/absolute/path/to/reference.wav \
swift test -c release --package-path ironmlx-app --filter audioAppLoadsAndRestoresFromSavedConfigurationWithBundle
```
