#!/usr/bin/env python3
"""Materialize the fixed 16 kHz to 22.05 kHz reference interpolation kernel."""
import argparse
import json
from pathlib import Path
import torch
import torchaudio
from torchaudio.functional.functional import _get_sinc_resample_kernel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if torch.__version__.split("+")[0] != "2.10.0" or torchaudio.__version__.split("+")[0] != "2.10.0":
    raise ValueError("reference kernel requires torch/torchaudio 2.10.0")
kernel, width = _get_sinc_resample_kernel(16000, 22050, 50, dtype=torch.float32)
phases = []
for row in kernel[:, 0]:
    # Values outside sinc support arise only from clamping at ±6 and have
    # negligible roundoff residue. Keep the full active support, including zeros.
    active = torch.where(row.abs() > 1e-15)[0]
    first, last = active[0].item(), active[-1].item()
    phases.append([first - width, row[first:last + 1].tolist()])
args.output.write_text(json.dumps({"reference": "torch/torchaudio 2.10.0 CPU float32, Hann sinc width 6 rolloff 0.99",
    "source_rate": 16000, "target_rate": 22050, "phases": phases}, separators=(",", ":")) + "\n")
