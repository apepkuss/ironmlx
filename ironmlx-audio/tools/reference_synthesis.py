#!/usr/bin/env python3
"""Generate full GPT-to-waveform baselines with explicit request-local random keys."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

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
    raise ValueError("full acoustic baseline requires MLX 0.32.2")
package = types.ModuleType("mlx_indextts")
package.__path__ = [str(args.reference.resolve() / "mlx_indextts")]
sys.modules["mlx_indextts"] = package
from mlx_indextts.config import IndexTTSConfig
from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25
from mlx_indextts.models.codec_v25 import EnhancedCodecV25
from mlx_indextts.models.s2mel import create_s2mel_from_config
from mlx_indextts.models.bigvgan_v2 import BigVGANV2, BigVGANV2Config

cfg = OmegaConf.load(args.snapshot / "config.yaml")
gpt_cfg = IndexTTSConfig.from_omegaconf(cfg)
gpt_cfg.version = 2.5
gpt = UnifiedVoiceV25(gpt_cfg)
codec = EnhancedCodecV25()
s2mel = create_s2mel_from_config(OmegaConf.to_container(cfg.s2mel, resolve=True))
vocoder = BigVGANV2(BigVGANV2Config())
for model, name in [
    (gpt, "gpt"),
    (codec, "codec"),
    (s2mel, "s2mel"),
    (vocoder, "bigvgan"),
]:
    model.load_weights(str(args.snapshot / f"{name}.safetensors"), strict=True)
    model.eval()
reference = mx.load(str(args.conditioning))
conditioning = reference["conditioning"]
corpus = json.loads(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/text.json").read_text()
)
outputs = {
    "conditioning": conditioning,
    "prompt_condition": reference["prompt_condition"],
    "ref_mel": reference["mel"].transpose(0, 2, 1),
    "style": reference["style"],
}
for case in corpus["cases"][:2]:
    name = case["language"]
    tokens = mx.array([case["token_ids"][0]], dtype=mx.int32)
    outputs[f"{name}.text"] = tokens
    outputs[f"{name}.language"] = mx.array(case["language_id"], dtype=mx.int32)
    current, padding = gpt.prepare_inputs(conditioning, tokens, case["language_id"])
    start = gpt.mel_embedding(
        mx.array([[8192]])
    ) + gpt.mel_pos_embedding.get_fixed_embedding(0)
    current = mx.concatenate([current, start], axis=1)
    padding = mx.concatenate([padding, mx.ones((1, 1), dtype=mx.int32)], axis=1)
    cache, generated, key = None, [], mx.random.key(2025)
    for step in range(1500):
        key, draw = mx.random.split(key)

        # Preserve the pinned 0.31 reference sampler's Gumbel-max algorithm.
        def categorical(values):
            return mx.argmax(values + mx.random.gumbel(values.shape, key=draw), axis=-1)

        with patch.object(mx.random, "categorical", side_effect=categorical):
            sampled, _, cache = gpt.generate_step(
                current,
                cache=cache,
                temperature=0.8,
                top_k=30,
                top_p=0.8,
                repetition_penalty=10.0,
                generated_tokens=generated,
                attention_mask=padding,
            )
        token = sampled.item()
        if token == 8193:
            break
        if not 0 <= token < 8192:
            raise ValueError("invalid semantic token")
        generated.append(token)
        mx.eval(cache)
        current = gpt.mel_embedding(
            mx.array([[token]])
        ) + gpt.mel_pos_embedding.get_fixed_embedding(len(generated))
        padding = mx.concatenate([padding, mx.ones((1, 1), dtype=mx.int32)], axis=1)
    else:
        raise ValueError("generation limit without EOS")
    if not generated:
        raise ValueError("no semantic codes")
    outputs[f"{name}.raw_codes"] = mx.array(generated, dtype=mx.uint32)
    # Same post-EOS silence compression as fixed generate.compress_silence.
    codes = generated
    if generated.count(52) > 30:
        codes, consecutive = [], 0
        for code in generated:
            if code != 52:
                codes.append(code)
                consecutive = 0
            elif consecutive < 10:
                codes.append(code)
                consecutive += 1
    outputs[f"{name}.codes"] = mx.array(codes, dtype=mx.uint32)
    semantic = codec.decode(mx.array([codes], dtype=mx.int32))
    length = max(1, int(semantic.shape[1] * 1.72))
    condition, *_ = s2mel.length_regulator(
        semantic, mx.array([length]), n_quantizers=3, f0=None
    )
    combined = mx.concatenate([reference["prompt_condition"], condition], axis=1)
    key, draw = mx.random.split(key)
    noise = mx.random.normal((1, 80, combined.shape[1]), key=draw)
    mel = s2mel.cfm.solve_euler(
        noise,
        mx.array([combined.shape[1]]),
        outputs["ref_mel"],
        combined,
        reference["style"],
        None,
        mx.linspace(0, 1, 26),
        0.7,
    )
    generated_mel = mel[:, :, outputs["ref_mel"].shape[2] :]
    audio = vocoder(generated_mel)
    mx.eval(audio)
    for component, value in [
        ("semantic", semantic),
        ("condition", condition),
        ("noise", noise),
        ("mel", generated_mel),
        ("audio", audio),
    ]:
        outputs[f"{name}.{component}"] = value
    print(name, "codes", len(generated), "audio", audio.shape, flush=True)
mx.save_safetensors(
    str(args.output), outputs, metadata={"reference": revision, "mlx": mx.__version__}
)
args.output.with_suffix(".json").write_text(
    json.dumps(
        {
            "revision": revision,
            "mlx": mx.__version__,
            "seed": 2025,
            "sampling": "explicit-split-gumbel-max",
            "shapes": {k: list(v.shape) for k, v in outputs.items()},
        },
        indent=2,
    )
    + "\n"
)
