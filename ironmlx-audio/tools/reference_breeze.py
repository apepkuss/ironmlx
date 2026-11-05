#!/usr/bin/env python3
"""Generate explicit native Breeze parity fixtures from pinned mlx-audio sources.

Requires Python MLX 0.32.4, mlx-lm 0.28.2, transformers 4.57.3.
The source paths are offline audit inputs; production never imports Python.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizerFast

p = argparse.ArgumentParser()
p.add_argument('--snapshot', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--config-source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
for path, expected in [(a.source, 'bdb56012ec3bc40b1836ff8d7f247dfed01bd6d3b9e28ca489cdc679bdb36706'),
                       (a.config_source, 'c734bb7659d648c071556493b9245cb2f70ddf6584df8cd1299b8930086e1374')]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Upstream source identity mismatch: {path}')
# Only the upstream neural network classes are needed; avoid importing audio IO.
base = types.ModuleType('mlx_audio.tts.models.base')
@dataclass
class BaseModelArgs:
    pass
base.BaseModelArgs = BaseModelArgs
base.GenerationResult = object
sys.modules[base.__name__] = base
utils = types.ModuleType('mlx_audio.utils')
utils.load_audio = None
sys.modules[utils.__name__] = utils
spec = importlib.util.spec_from_file_location('reference_breeze.config', a.config_source)
config = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = config
spec.loader.exec_module(config)
module = types.ModuleType('reference_breeze.model')
module.__package__ = 'reference_breeze'
source = a.source.read_text().replace('mlx_audio.lm.', 'mlx_lm.')
exec(compile(source, str(a.source), 'exec'), module.__dict__)
c = json.loads((a.snapshot/'config.json').read_text())
model = module.Model(config.ModelConfig.from_dict(c))
index = json.loads((a.snapshot/'model.safetensors.index.json').read_text())['weight_map']
w = {}
for f in sorted(set(index.values())):
    w.update(mx.load(str(a.snapshot/f)))
w = model.sanitize(w)
if q := c.get('quantization'):
    nn.quantize(model, group_size=q['group_size'], bits=q['bits'], mode=q['mode'], class_predicate=lambda path, layer: path+'.scales' in w)
model.load_weights(list(w.items()))
model.eval()
tokenizer_config = json.loads((a.snapshot/'tokenizer_config.json').read_text())
for k in ('tokenizer_class','model_max_length','clean_up_tokenization_spaces','added_tokens_decoder'):
    tokenizer_config.pop(k,None)
model.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(a.snapshot/'tokenizer.json'), **tokenizer_config)
result={}
for label,text,instruct in [('en','Hello.','A clear and calm English narrator.'),('zh','你好。','一位温柔清晰的中文女声。')]:
    target=f'[S0]<ins_bos>{instruct}<ins_eos>{text}'
    ids=model._text_ids(target)[None,:]
    x=model.text_encoder(ids)
    prompt=model.text_encoder_proj(x)
    cache=model.backbone_model.make_cache()
    hidden=model.backbone_model(input_embeddings=prompt,cache=cache)[:,-1:,:]
    logits=model.lm_head(hidden)
    first=int(model._mask_reserved_codec_logits(logits).argmax(-1).item())
    result[f'{label}.ids']=ids
    result[f'{label}.text']=x
    result[f'{label}.prompt']=prompt
    result[f'{label}.hidden']=hidden
    result[f'{label}.logits']=logits
    frame=[first]
    dcache=model.depth_decoder.model.make_cache()
    model.depth_decoder.start_frame(hidden[:,0,:],dcache)
    for i in range(15):
        dl=model.depth_decoder.step_logits(dcache,head_idx=i,token_id=frame[-1])
        result[f'{label}.depth.{i}']=dl
        frame.append(int(model._mask_reserved_codec_logits(dl).argmax(-1).item()))
    result[f'{label}.frame']=mx.array(frame,dtype=mx.int32)
    codes=mx.array(frame,dtype=mx.int32)[None,None,:]
    result[f'{label}.next']=model.backbone_model(input_ids=codes,cache=cache)[:,-1:,:]
    mx.eval(result)
a.output.parent.mkdir(parents=True,exist_ok=True)
mx.save_safetensors(str(a.output),result,metadata={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(a.config_source.read_bytes()).hexdigest(),'mlx':mx.__version__,'snapshot':a.snapshot.name})
print(a.output, flush=True)
