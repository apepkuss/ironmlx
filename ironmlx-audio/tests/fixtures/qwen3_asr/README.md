# Qwen3-ASR fixtures

The two configuration files are metadata from the published
`mlx-community/Qwen3-ASR-0.6B-8bit` and `mlx-community/Qwen3-ASR-1.7B-8bit`
checkpoints. They contain no model weights.

`whisper-mel-440hz.f32` is a little-endian float32 array of shape `[128,100]`.
It was generated with Transformers 5.20.0 `WhisperFeatureExtractor(feature_size=128)`
from 16,000 samples of `float32(0.25 * sin(2*pi*440*i/16000))`, with
`sampling_rate=16000`, `padding=True`, `truncation=False`, and `return_tensors="np"`.
The test compares every mel value, including reflective STFT boundary frames.

Native full-model acceptance additionally uses the official Qwen English and
Chinese speech samples with both local checkpoints. The numerical reference is
the MLX Audio implementation at commit
`dad2087f790dfe96b07d5aac1405db4b578b411f`; production inference does not import
Python, Transformers, or MLX Audio.
