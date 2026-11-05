"""Generate codec parity fixtures from the pinned mlx-audio implementation.

Requires MLX 0.32.4, mlx-lm 0.28.2, mlx-audio's Mimi/config dependencies and
speech_tokenizer.py from mlx-audio b1db0f21c1767b40b3496927f12df67495f0088b.
This script is an offline verifier; production never imports Python.
"""
import argparse
import hashlib
import json
import types
from pathlib import Path

import mlx.core as mx
from mlx_audio.tts.models.qwen3_tts.config import (
    Qwen3TTSTokenizerConfig,
    Qwen3TTSTokenizerDecoderConfig,
    Qwen3TTSTokenizerEncoderConfig,
)

p = argparse.ArgumentParser()
p.add_argument("snapshot", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--frames", type=int, default=80)
p.add_argument("--chunk-frames", type=int, default=4)
a = p.parse_args()
source_hash = hashlib.sha256(a.source.read_bytes()).hexdigest()
if source_hash != "e15ba6efd871e31b6a450f1405760e0790ee4ab6c846b4be2cb363afd0d617b6":
    raise ValueError("Upstream codec source identity mismatch")
# mlx-audio moved its vendored LM helpers after the older installed Mimi package.
# Resolve those imports to the explicit mlx-lm verification dependency.
module = types.ModuleType("mlx_audio.tts.models.qwen3_tts.reference_codec")
source = a.source.read_text().replace("mlx_audio.lm.", "mlx_lm.")
exec(compile(source, str(a.source), "exec"), module.__dict__)
Qwen3TTSSpeechTokenizer = module.Qwen3TTSSpeechTokenizer
c = json.loads((a.snapshot / "audio_tokenizer/config.json").read_text())

def fields(cls, obj):
    return {k: v for k, v in obj.items() if k in cls.__dataclass_fields__}

cfg = Qwen3TTSTokenizerConfig(
    decoder_config=Qwen3TTSTokenizerDecoderConfig(
        **fields(Qwen3TTSTokenizerDecoderConfig, c["decoder_config"])
    ),
    encoder_config=Qwen3TTSTokenizerEncoderConfig(
        **fields(Qwen3TTSTokenizerEncoderConfig, c["encoder_config"])
    ),
)
m = Qwen3TTSSpeechTokenizer(cfg)
w = m.sanitize(mx.load(str(a.snapshot / "audio_tokenizer/model.safetensors")))
m.load_weights(list(w.items()), strict=False)
for layer in m.encoder_model.quantizer.rvq_first.vq.layers + m.encoder_model.quantizer.rvq_rest.vq.layers:
    layer.codebook.update_in_place()
mx.eval(m.parameters())
m.eval()
t = mx.arange(24000, dtype=mx.float32) / 24000
wave = 0.2 * mx.sin(2 * 3.141592653589793 * 220 * t) + 0.05 * mx.sin(2 * 3.141592653589793 * 571 * t)
encoded = m.encode(wave[None, None, :]).transpose(0, 2, 1)
codes = ((mx.arange(16 * a.frames).reshape(1, a.frames, 16) * 37) % 2047 + 1).astype(mx.int32)
decoded = m.decoder(codes.transpose(0, 2, 1)).reshape(-1)
mx.eval(decoded)
m.decoder.reset_streaming_state()
chunks = []
for start in range(0, a.frames, a.chunk_frames):
    chunk = m.decoder.streaming_step(codes[:, start:start + a.chunk_frames, :].transpose(0, 2, 1)).reshape(-1)
    mx.eval(chunk)
    chunks.append(chunk)
mx.save_safetensors(
    str(a.output),
    dict(wave=wave, encoded=encoded, codes=codes, decoded=decoded, stream=mx.concatenate(chunks)),
    metadata={"source_sha256": source_hash, "mlx": mx.__version__, "snapshot": a.snapshot.name},
)
print(a.output)
