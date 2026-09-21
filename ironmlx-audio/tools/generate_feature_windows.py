#!/usr/bin/env python3
"""Materialize fixed float32 analysis windows without host libm variation."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if torch.__version__.split("+")[0] != "2.10.0":
    raise ValueError("reference windows require PyTorch 2.10.0")
args.output.write_text(json.dumps({
    "reference": "torch 2.10.0 float32 CPU hann_window; Kaldi Povey power 0.85",
    "hann1024": torch.hann_window(1024, periodic=True).tolist(),
    "povey400": torch.hann_window(400, periodic=False).pow(0.85).tolist(),
}, separators=(",", ":")) + "\n")
