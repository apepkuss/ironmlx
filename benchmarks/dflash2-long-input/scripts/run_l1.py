#!/usr/bin/env python3
"""L1 prefill QMM feasibility runs (docs/protocol-l1-qmm-feasibility.md).
--kind identity: one process per (mode, projection, variant) with one steady
  call inside an MLX GPU capture (MTL_CAPTURE_ENABLED=1); the set of named
  compute pipelines the process created is extracted from the capture, and
  the server-style witness lines from stderr. No timing.
--kind wall: six fresh processes, modes P D D P P D, all projections, wall
  time per call (first call apart from 5 warmup + 30 timed).
--kind chain: as wall, but 8 back-to-back calls per eval (3 warmup + 15
  timed); wall / 8 is the sustained per-call time (protocol amendment 1).
--kind gpu: one process per mode under xctrace Metal System Trace, 10 timed
  calls; GPU intervals of the process are exported for attribution. Wall
  times from these processes are not used."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import common as C

BIN = C.REPO / 'target/release/examples/prefill_qmm_feasibility'
PROJECTIONS = ('mlp_gate_up', 'mlp_down', 'gdn_in_proj', 'gdn_out_proj', 'attn_qkv', 'attn_o_proj')
WALL_ORDER = ('production', 'default', 'default', 'production', 'production', 'default')
NAME = re.compile(rb'(affine_[A-Za-z0-9_]+|steel_[A-Za-z0-9_]+|ironmlx_[A-Za-z0-9_]+|gemm_[A-Za-z0-9_]+|'
                  rb'v[nsv]_[A-Za-z0-9_]+|g[0-9]*_[A-Za-z0-9_]+)')


def base(mode, out, extra=()):
    return [str(BIN), '--mode', mode, '--model', str(C.TARGET), '--mlx-metallib', str(C.METALLIB), '--out', str(out),
            *extra]


def pipelines(trace):
    names = set()
    for path in trace.iterdir():
        if path.is_file() and path.stat().st_size < 2_000_000:
            names.update(m.decode() for m in NAME.findall(path.read_bytes()))
    return sorted(n for n in names if n not in ('gemm_k_iterations_aligned', 'gemm_k_iterations', 'gemm_params'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--kind', choices=('identity', 'wall', 'chain', 'gpu'), required=True)
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    data = dict(label=a.label, kind=a.kind, binary=dict(path=str(BIN), sha256=C.digest(BIN)),
                source_sha256=C.digest(C.REPO / 'mlx/examples/prefill_qmm_feasibility.rs'),
                protocol_sha256=C.digest(C.SUITE / 'docs/protocol-l1-qmm-feasibility.md'),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), runs=[], complete=False)
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    save()
    env = C.environment()
    if a.kind == 'identity':
        env['MTL_CAPTURE_ENABLED'] = '1'
        for mode in ('production', 'default'):
            for projection in PROJECTIONS:
                for variant in ('qmm', 'bf16'):
                    name = f'{mode}-{projection}-{variant}'
                    cmd = base(mode, directory / f'{name}.json', ('--only', projection, '--capture-dir',
                                                                   str(directory), '--variant', variant))
                    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    trace = directory / f'{name}.gputrace'
                    data['runs'].append(dict(name=name, mode=mode, projection=projection, variant=variant,
                                             exit_code=done.returncode, stderr=done.stderr,
                                             pipelines=pipelines(trace) if trace.exists() else None))
                    save()
                    print(json.dumps(dict(run=name, exit=done.returncode), ensure_ascii=False), flush=True)
    elif a.kind in ('wall', 'chain'):
        extra = ('--chain', '8', '--warmup', '3', '--runs', '15') if a.kind == 'chain' else ()
        for index, mode in enumerate(WALL_ORDER):
            name = f'p{index}-{mode}'
            done = subprocess.run(base(mode, directory / f'{name}.json', extra), env=env, capture_output=True,
                                  text=True)
            record = dict(name=name, index=index, mode=mode, exit_code=done.returncode, stderr=done.stderr)
            if done.returncode == 0:
                record['result'] = json.loads((directory / f'{name}.json').read_text())
            data['runs'].append(record)
            save()
            print(json.dumps(dict(run=name, exit=done.returncode), ensure_ascii=False), flush=True)
            time.sleep(20)
    else:
        for mode in ('production', 'default'):
            name = f'gpu-{mode}'
            trace = directory / f'{name}.trace'
            cmd = ['xcrun', 'xctrace', 'record', '--template', 'Metal System Trace', '--output', str(trace),
                   '--launch', '--', *base(mode, directory / f'{name}.json', ('--runs', '10', '--warmup', '2'))]
            done = subprocess.run(cmd, env=env, capture_output=True, text=True)
            export = directory / f'{name}.gpu-intervals.xml'
            with export.open('w') as handle:
                subprocess.run(['xcrun', 'xctrace', 'export', '--input', str(trace), '--xpath',
                                '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'],
                               stdout=handle, check=False)
            data['runs'].append(dict(name=name, mode=mode, exit_code=done.returncode, log=done.stdout + done.stderr,
                                     result=json.loads((directory / f'{name}.json').read_text())
                                     if (directory / f'{name}.json').exists() else None))
            save()
            print(json.dumps(dict(run=name, exit=done.returncode), ensure_ascii=False), flush=True)
            time.sleep(20)
    data['complete'] = True
    save()


if __name__ == '__main__':
    main()
