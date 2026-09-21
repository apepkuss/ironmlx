#!/usr/bin/env python3
"""Generate fixed semantic-code decoder and acoustic reference tensors offline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import mlx.core as mx
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description=__doc__)
for name in ("reference", "snapshot", "conditioning", "output"):
    parser.add_argument(f"--{name}", type=Path, required=True)
args = parser.parse_args()
revision = subprocess.check_output(
    ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
).strip()
if (
    revision != "a7666367b8551656a2029ad75f259cb5e4936b3b"
    or os.environ.get("MLX_ENABLE_TF32") != "0"
):
    raise ValueError("requires fixed reference and MLX_ENABLE_TF32=0 at startup")
if mx.__version__ != "0.32.2":
    raise ValueError("acoustic baseline requires MLX 0.32.2")
package = types.ModuleType("mlx_indextts")
package.__path__ = [str(args.reference.resolve() / "mlx_indextts")]
sys.modules["mlx_indextts"] = package
from mlx_indextts.models.codec_v25 import EnhancedCodecV25
from mlx_indextts.models.s2mel import create_s2mel_from_config
from mlx_indextts.models.bigvgan_v2 import BigVGANV2, BigVGANV2Config

cfg = OmegaConf.load(args.snapshot / "config.yaml")
codec = EnhancedCodecV25()
codec.load_weights(str(args.snapshot / "codec.safetensors"), strict=True)
codec.eval()
codes = mx.array([[680, 6750, 992, 417, 52, 52, 4123, 197]], dtype=mx.uint32)
trace = {}
x = codec.quantizer.vq2emb(codes)
trace["codec.quantized"] = x
x = codec.decoder[0].norm(codec.decoder[0].embed(x))
trace["codec.embed"] = x
import mlx.nn as nn

for i, block in enumerate(codec.decoder[0].convnext):
    trace[f"codec.{i}.dwconv"] = block.dwconv(x)
    trace[f"codec.{i}.norm"] = block.norm(trace[f"codec.{i}.dwconv"])
    inner = block.pwconv1(trace[f"codec.{i}.norm"])
    trace[f"codec.{i}.gelu_input"] = inner
    trace[f"codec.{i}.gelu"] = nn.gelu(inner)
    x = block(x)
    trace[f"codec.block{i}"] = x
semantic = codec.decode(codes)
mx.eval(semantic)
print("semantic", semantic.shape, flush=True)
model = create_s2mel_from_config(OmegaConf.to_container(cfg.s2mel, resolve=True))
model.load_weights(str(args.snapshot / "s2mel.safetensors"), strict=True)
model.eval()
reference = mx.load(str(args.conditioning))
condition, *_ = model.length_regulator(
    semantic, mx.array([int(semantic.shape[1] * 1.72)]), n_quantizers=3, f0=None
)
combined = mx.concatenate([reference["prompt_condition"], condition], axis=1)
noise = mx.random.normal((1, 80, combined.shape[1]), key=mx.random.key(2025))
mel = model.cfm.solve_euler(
    noise,
    mx.array([combined.shape[1]]),
    reference["mel"].transpose(0, 2, 1),
    combined,
    reference["style"],
    None,
    mx.linspace(0, 1, 26),
    0.7,
)
mx.eval(mel)
print("mel", mel.shape, flush=True)
generated_mel = mel[:, :, reference["mel"].shape[1] :]
vocoder = BigVGANV2(BigVGANV2Config())
vocoder.load_weights(str(args.snapshot / "bigvgan.safetensors"), strict=True)
vocoder.eval()
audio = vocoder(generated_mel)
mx.eval(audio)
outputs = {
    **trace,
    "codes": codes,
    "semantic": semantic,
    "condition": condition,
    "combined": combined,
    "noise": noise,
    "mel": mel,
    "generated_mel": generated_mel,
    "audio": audio,
    "ref_mel": reference["mel"].transpose(0, 2, 1),
    "style": reference["style"],
}
mx.save_safetensors(str(args.output), outputs, metadata={"reference": revision})
args.output.with_suffix(".json").write_text(
    json.dumps(
        {
            "revision": revision,
            "mlx_version": mx.__version__,
            "shapes": {k: list(v.shape) for k, v in outputs.items()},
            "dtypes": {k: str(v.dtype) for k, v in outputs.items()},
        },
        indent=2,
    )
    + "\n"
)
print("audio", audio.shape, flush=True)
