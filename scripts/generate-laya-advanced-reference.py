#!/usr/bin/env python3
"""Generate device/compile-specific Laya references using upstream laya-mlx.

Use laya-mlx commit 0a859518634112655cb97c745dbf04f5191aaf13 and MLX 0.32.2.
The checkpoint is aac6fef/laya-multilingual-mlx revision
f2b4faf51023039425946074e2cf1361d2db11d5. No files are downloaded by this script.
"""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime-dir', required=True, type=Path)
parser.add_argument('--model-dir', required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.runtime_dir))
from laya_mlx import load

root = Path(__file__).resolve().parents[1] / 'ironmlx-decision/tests/fixtures/laya-reference'
for device in ['gpu', 'cpu']:
    for dtype in ['float16', 'float32']:
        for compiled in [False, True]:
            mode = 'compiled' if compiled else 'eager'
            agent = load(str(args.model_dir), device=device, dtype=dtype,
                         batch_size=2, pad_to_multiple=16, compile=compiled)
            for index in range(4):
                request = json.loads((root / f'{index}.request.json').read_text())
                output = agent.predict(request['state'], request['questions'])
                file = root / f'{index}.{device}.{mode}.{dtype}.reference.json'
                file.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
            print(device, dtype, mode, flush=True)
