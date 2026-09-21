#!/usr/bin/env python3
"""Generate spectrogram baselines from the fixed Python reference, offline."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import librosa
import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import save_file
import torch
import torchaudio
from transformers import SeamlessM4TFeatureExtractor

REVISION = "a7666367b8551656a2029ad75f259cb5e4936b3b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "snapshot", "w2v", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--audio", type=Path)
    source_group.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("unexpected reference revision")
    source = args.reference / "mlx_indextts" / "generate_v2.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == "IndexTTSv2")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_init_mel_config")
    # Execute the unchanged upstream method without initializing synthesis networks.
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    model = SimpleNamespace(cfg=OmegaConf.load(args.snapshot / "config.yaml"))
    scope["_init_mel_config"](model)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.w2v, local_files_only=True)
    if args.synthetic:
        time = np.arange(16000, dtype=np.float64) / 16000
        audio = (0.2 * np.sin(2 * np.pi * (170 * time + 1200 * time**2))
                 + np.random.default_rng(2025).normal(0, 0.01, len(time))).astype(np.float32)
        cases = [("synthetic", 16000)]
    else:
        audio, _ = librosa.load(args.audio, sr=16000, mono=True)
        cases = [("speech_1s", 16000), ("speech_odd", 16160), ("speech_5s", 80000)]
    outputs = {}
    for name, length in cases:
        if len(audio) < length:
            raise ValueError("reference speech must contain at least five seconds")
        wave = torch.from_numpy(audio[:length].copy())[None]
        audio_22k = torchaudio.functional.resample(wave, 16000, 22050)
        mel = model.mel_fn(audio_22k).transpose(1, 2)
        fbank = torchaudio.compliance.kaldi.fbank(wave, num_mel_bins=80, dither=0,
                                               sample_frequency=16000)
        fbank -= fbank.mean(dim=0, keepdim=True)
        semantic = extractor(wave.numpy(), sampling_rate=16000, return_tensors="pt")
        outputs.update({
            f"{name}.wave16": wave.contiguous(), f"{name}.wave22": audio_22k.contiguous(),
            f"{name}.mel": mel.contiguous(), f"{name}.fbank": fbank[None].contiguous(),
            f"{name}.semantic": semantic.input_features.contiguous(),
            f"{name}.mask": semantic.attention_mask.to(torch.int32).contiguous(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(outputs, str(args.output), metadata={"reference": REVISION})
    args.output.with_suffix(".json").write_text(json.dumps({
        "reference_revision": revision, "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest() if args.audio else "synthetic-seed-2025",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tensors": {key: list(value.shape) for key, value in outputs.items()},
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
