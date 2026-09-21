#!/usr/bin/env python3
"""Generate fixed IndexTTS 2.5 prefill/decode/cache/sampling baselines offline."""

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
package = types.ModuleType("mlx_indextts")
package.__path__ = [str(args.reference.resolve() / "mlx_indextts")]
sys.modules["mlx_indextts"] = package
from mlx_indextts.config import IndexTTSConfig
from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25

cfg = IndexTTSConfig.from_omegaconf(OmegaConf.load(args.snapshot / "config.yaml"))
cfg.version = 2.5
model = UnifiedVoiceV25(cfg)
model.load_weights(str(args.snapshot / "gpt.safetensors"), strict=True)
model.eval()
conditioning = mx.load(str(args.conditioning))["conditioning"]
corpus = json.loads(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/text.json").read_text()
)
outputs = {"conditioning": conditioning}
for case in corpus["cases"][:2]:
    name = case["language"]
    tokens = mx.array([case["token_ids"][0]], dtype=mx.int32)
    outputs[name + ".text"] = tokens
    outputs[name + ".language"] = mx.array(case["language_id"], dtype=mx.int32)
    x, padding = model.prepare_inputs(conditioning, tokens, case["language_id"])
    start = model.mel_embedding(
        mx.array([[8192]])
    ) + model.mel_pos_embedding.get_fixed_embedding(0)
    x = mx.concatenate([x, start], axis=1)
    padding = mx.concatenate([padding, mx.ones((1, 1), dtype=mx.int32)], axis=1)
    cache, generated = None, []
    for step in range(4):
        outputs[f"{name}.{step}.input"] = x
        outputs[f"{name}.{step}.mask"] = padding
        attention = model.generation_attention_mask(
            padding, query_len=x.shape[1], key_len=padding.shape[1]
        )
        hidden, cache = model.gpt(x, mask=attention, cache=cache)
        logits = model.mel_head(model.final_norm(hidden[:, -1:, :]))[:, 0, :]
        key = mx.random.key(2025 + step)

        def categorical(values):
            outputs[f"{name}.{step}.sampling_logits"] = values
            return mx.argmax(values + mx.random.gumbel(values.shape, key=key), axis=-1)

        with patch.object(mx.random, "categorical", side_effect=categorical):
            sampled = model._sample(logits, 0.8, 30, 0.8, 10.0, generated)
        mx.eval(logits, sampled, cache)
        outputs[f"{name}.{step}.logits"] = logits
        outputs[f"{name}.{step}.sample"] = sampled
        outputs[f"{name}.{step}.k0"] = cache[0][0]
        outputs[f"{name}.{step}.v23"] = cache[-1][1]
        token = int(sampled.item())
        print(name, step, token, flush=True)
        if token >= 8192:
            raise ValueError("baseline needs four nonterminal steps")
        generated.append(token)
        x = model.mel_embedding(
            mx.array([[token]])
        ) + model.mel_pos_embedding.get_fixed_embedding(len(generated))
        padding = mx.concatenate([padding, mx.ones((1, 1), dtype=mx.int32)], axis=1)
mx.save_safetensors(str(args.output), outputs, metadata={"reference": revision})
args.output.with_suffix(".json").write_text(
    json.dumps(
        {
            "revision": revision,
            "mlx_version": mx.__version__,
            "sampling": "gumbel-max",
            "shapes": {name: list(tensor.shape) for name, tensor in outputs.items()},
            "dtypes": {name: str(tensor.dtype) for name, tensor in outputs.items()},
        },
        indent=2,
    )
    + "\n"
)
