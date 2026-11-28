#!/usr/bin/env python3
"""Regenerate reference outputs with mlx-lm 0c4c99192230f1943dd468d297f0b8a64ae7b7c2.

Use MLX_ENABLE_TF32=0, matching the IronMLX CLI. The small deterministic
weights are synthetic; no original checkpoint weights are included.
"""
import os
os.environ['MLX_ENABLE_TF32'] = '0'
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm.models import inkling2

root = Path(__file__).parent
config = json.loads((root / 'config.json').read_text())
model = inkling2.Model(inkling2.ModelArgs.from_dict(config))
model.load_weights(str(root / 'model.safetensors'))
def save_reference(model, root):
    ids = mx.arange(1, 13)[None]
    full = model(ids, cache=model.make_cache())
    cache = model.make_cache()
    first = model(ids[:, :5], cache=cache)
    second = model(ids[:, 5:], cache=cache)
    pixels = mx.arange(3 * 8 * 12, dtype=mx.float32).reshape(3, 8, 12) / 200 - 0.5
    mx.save_safetensors(str(root / 'reference.safetensors'), {
        'logits': full, 'chunked': mx.concatenate([first, second], axis=1),
        'pixels': pixels, 'vision': model.encode_image(pixels),
    })

save_reference(model, root)
# Exercise the real checkpoint's two expert formats and per-expert amax.
from mlx_lm.utils import quantize_model
from mlx.utils import tree_flatten

def predicate(path, module):
    if '.mlp.experts.' not in path:
        return False
    if '.layers.7.' in path:
        return {'group_size': 32, 'bits': 8, 'mode': 'mxfp8'}
    return {'group_size': 16, 'bits': 4, 'mode': 'nvfp4', 'global_scale': True}

model, quant_config = quantize_model(model, config, None, None, quant_predicate=predicate)
mixed = root / 'mixed'
mixed.mkdir(exist_ok=True)
# The published checkpoint declares only global affine metadata; expert formats
# are identified by their packed tensors, scales and per-expert global scales.
checkpoint_config = dict(config, quantization={'group_size': 64, 'bits': 4, 'mode': 'affine'})
(mixed / 'config.json').write_text(json.dumps(checkpoint_config, indent=2) + '\n')
(mixed / 'tokenizer_config.json').write_text('{}\n')
mx.save_safetensors(str(mixed / 'model.safetensors'), dict(tree_flatten(model.parameters())))
save_reference(model, mixed)
