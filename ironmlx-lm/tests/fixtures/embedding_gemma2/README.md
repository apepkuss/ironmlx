# EmbeddingGemma 2 reference fixtures

The text and image fixtures cover the previously qualified BF16 and affine 4bit
adapters. Audio fixtures extend that regression set without replacing its stricter
text/image tolerances.

## Audio provenance

`audio-reference.json` records 12 independent upstream outputs for both formats:
three speech clips, three matching text queries, ordered text/audio combinations,
two ordered audio clips, silence, a 440 Hz tone, 161 samples and 801 samples. All
WAVs are mono 16 kHz PCM16. The speech clips were synthesized locally with macOS
Samantha from these project-owned sentences, then explicitly resampled with
`ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le`:

- `sunny.wav`: “The weather is sunny today.”
- `train.wav`: “A train arrives at the station.”
- `window.wav`: “Please open the window.”

The other WAVs were generated mathematically (silence and sine waves).
`sunny.flac` is a lossless conversion and `sunny.mp3` a lossy conversion of the
same WAV, used for transport/decoder tests rather than exact vector goldens.
SHA-256 hashes of all audio files are recorded in `audio-reference.json`.
No third-party recordings or model weights are committed.

References execute original source files from:

- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0),
  MIT, including the Gemma4 Conformer and EmbeddingGemma 2 text encoder/projector.
- [Transformers](https://github.com/huggingface/transformers/tree/cb33194ad6152bd9fad6305378d92db385dd7b32),
  Apache-2.0, including the Gemma4 audio feature extractor and mel/window helpers.

`audio-source-hashes.json` verifies each executed source file against these pins.
The generator stubs unrelated package imports and the generic dataclass config
filter, not the numerical model. Multiple clips use upstream feature extraction,
projection, scatter, text model and pooling in input order.

Checkpoint revisions are BF16
`1a4ffddb7905d3f63486748deabe091a01fb6201` and affine 4bit
`8ab839e127cb8cb305edd71fb7a02c82f2d49ab3`, from `mlx-community`.

## Reproduce audio references

Use Python 3.11 with `mlx==0.32.3`, `numpy==1.26.4`, `tokenizers==0.23.2` and
`Pillow==12.3.0`. NumPy is pinned because its FFT promotion changed in NumPy 2;
this reference computes the spectrum in float64 before the model's float32 mel
features and float32 Conformer intermediates.

Clone the two upstream repositories and check out the revisions above. Download
both checkpoints separately; the generator makes no network requests.

```sh
python scripts/generate-embedding-gemma2-audio-reference.py \
  --mlx-vlm-dir /path/to/mlx-vlm --transformers-dir /path/to/transformers \
  --features-only --check
python scripts/generate-embedding-gemma2-audio-reference.py \
  --mlx-vlm-dir /path/to/mlx-vlm --transformers-dir /path/to/transformers \
  --model-dir /path/to/embeddinggemma-2-bf16 --precision bf16 --check
python scripts/generate-embedding-gemma2-audio-reference.py \
  --mlx-vlm-dir /path/to/mlx-vlm --transformers-dir /path/to/transformers \
  --model-dir /path/to/embeddinggemma-2-4bit --precision 4bit --check
```

Omit `--check` only when deliberately regenerating and reviewing goldens.
`audio-features-reference.json` records the official mask/token count and selected
first/middle/last log-mel rows. The Rust feature test uses absolute tolerance
`1e-6`. Real audio vectors require cosine similarity above `0.9995`, maximum
component error below `0.005`, finite unit norms, exact usage/token counts and
matching cross-modal retrieval top-1. These bounds allow BF16 rounding at the
Conformer/text projection boundary; they do not apply to existing text/image
reference tests. The real-checkpoint test also exercises independent versus
batched inputs, all four dimensions, 30-second audio, rejection of aggregate
limits and an image/audio/text sample.

```sh
IRONMLX_EMBEDDING_BF16_DIR=/path/to/embeddinggemma-2-bf16 \
IRONMLX_EMBEDDING_AFFINE4_DIR=/path/to/embeddinggemma-2-4bit \
  cargo test --release -p ironmlx-lm --test embedding_gemma2_checkpoint \
  -- --ignored --test-threads=1
```
