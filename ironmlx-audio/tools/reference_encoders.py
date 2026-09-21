#!/usr/bin/env python3
"""Compute fixed CPU PyTorch / MLX reference encoder tensors for native parity tests."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "snapshot", "w2v", "campplus", "features", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case", default="speech_1s")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != "a7666367b8551656a2029ad75f259cb5e4936b3b":
        raise ValueError("unexpected reference revision")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("set MLX_ENABLE_TF32=0 before starting the reference process")
    torch.set_num_threads(4)
    tensors = load_file(str(args.features))
    outputs = {"features": tensors[f"{args.case}.semantic"], "mask": tensors[f"{args.case}.mask"]}
    config = Wav2Vec2BertConfig.from_json_file(str(args.w2v / "config.json"))
    model = Wav2Vec2BertModel.from_pretrained(args.snapshot, config=config,
                                            local_files_only=True, attn_implementation="eager").eval()
    with torch.no_grad():
        result = model(input_features=outputs["features"], attention_mask=outputs["mask"], output_hidden_states=True)
        outputs["hidden17"] = result.hidden_states[17].contiguous()
        stats = torch.load(args.snapshot / "wav2vec2bert_stats.pt", weights_only=True, map_location="cpu")
        outputs["semantic"] = ((outputs["hidden17"] - stats["mean"]) / torch.sqrt(stats["var"])).contiguous()
    del model, result
    gc.collect()
    print("w2v hidden17", outputs["hidden17"].shape, flush=True)

    package = types.ModuleType("mlx_indextts")
    package.__path__ = [str(args.reference.resolve() / "mlx_indextts")]
    sys.modules["mlx_indextts"] = package
    from mlx_indextts.indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    camp = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    camp.load_state_dict(torch.load(args.campplus / "campplus_cn_common.bin", weights_only=True,
                                    map_location="cpu"), strict=True)
    outputs["fbank"] = tensors[f"{args.case}.fbank"]
    with torch.no_grad():
        outputs["style"] = camp(outputs["fbank"]).contiguous()
    del camp
    gc.collect()
    print("camp style", outputs["style"].shape, flush=True)

    import mlx.core as mx
    from mlx_indextts.config import IndexTTSConfig
    from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25
    from mlx_indextts.models.s2mel import create_s2mel_from_config
    cfg = OmegaConf.load(args.snapshot / "config.yaml")
    config = IndexTTSConfig.from_omegaconf(cfg)
    config.version = 2.5
    gpt = UnifiedVoiceV25(config)
    gpt.load_weights(str(args.snapshot / "gpt.safetensors"), strict=True)
    gpt.eval()
    semantic = mx.array(outputs["semantic"].numpy())
    lengths = mx.array([semantic.shape[1]], dtype=mx.int32)
    emotion = gpt.get_emovec(semantic.transpose(0, 2, 1), lengths)
    conditioning = gpt.prepare_conditioning_latents(mx.array(outputs["style"].numpy()), emotion, 1)
    mx.eval(emotion, conditioning)
    outputs["emotion"] = torch.from_numpy(np.array(emotion)).contiguous()
    outputs["conditioning"] = torch.from_numpy(np.array(conditioning)).contiguous()
    del gpt
    gc.collect()
    model = create_s2mel_from_config(OmegaConf.to_container(cfg.s2mel, resolve=True))
    model.load_weights(str(args.snapshot / "s2mel.safetensors"), strict=True)
    model.eval()
    outputs["mel"] = tensors[f"{args.case}.mel"]
    prompt, *_ = model.length_regulator(semantic, ylens=mx.array([outputs["mel"].shape[1]], dtype=mx.int32), n_quantizers=3, f0=None)
    mx.eval(prompt)
    outputs["prompt_condition"] = torch.from_numpy(np.array(prompt)).contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(outputs, str(args.output), metadata={"reference": revision})
    args.output.with_suffix(".json").write_text(json.dumps({"case": args.case,
        "shapes": {name: list(value.shape) for name, value in outputs.items()},
        "dtypes": {name: str(value.dtype) for name, value in outputs.items()},
    }, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
