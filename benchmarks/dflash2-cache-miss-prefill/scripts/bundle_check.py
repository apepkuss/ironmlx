#!/usr/bin/env python3
"""App bundle real-model check: the bundle's helper with its own
mlx.metallib, the App's default DFlash2 arguments and the App-like
environment. Short code-1 and knowledge-1 as cold misses then exact repeats,
and the 30K code-1 cold miss. Outputs are compared with the candidate's C-on
session of scenarios-main-v1; hit tokens and saves are recorded."""
import argparse
import json
from pathlib import Path
import time

import common as C

APP = Path('/Users/xin/workspace/perf-dflash2-cache-miss-prefill/dist/IronMLX.app')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    directory = C.SUITE / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    helper = APP / 'Contents/Helpers/ironmlx'
    metallib = APP / 'Contents/Resources/mlx.metallib'
    fx = C.fixtures()
    reference = json.loads((C.SUITE / 'results/scenarios-main-v1/run.json').read_text())
    ref = {f"{r['phase']}:{r['task']}:{r['step']}": r['output_sha256']
           for s in reference['sessions'] if s['name'] == 'C-on' for r in s['rows']}
    data = dict(label=a.label, helper_sha256=C.digest(helper), metallib_sha256=C.digest(metallib), rows=[])
    server = C.Server(directory, 'bundle', helper, True)
    server.cmd[server.cmd.index('--mlx-metallib') + 1] = str(metallib)
    steps = [('short', 'code-1', 'miss'), ('short', 'code-1', 'repeat'), ('short', 'knowledge-1', 'miss'),
             ('short', 'knowledge-1', 'repeat'), ('30k', 'code-1', 'miss')]
    with server:
        data.update(command=server.cmd, env_keys=sorted(server.env),
                    fingerprint=C.healthz()['dflash2'].get('prefix_fingerprint'),
                    m5_profile=C.healthz()['dflash2'].get('m5_profile', {}).get('status'))
        for text in C.WARMUPS:
            server.request(C.user(text), 64)
        for phase, task, step in steps:
            row = server.request(C.user(fx[phase][task]['prompt']), 4096 if phase == 'short' else 1024)
            key = f'{phase}:{task}:{step}'
            summary = dict(step=key, valid=C.valid(row), finish=row['finish_reason'], tokens=row['completion_tokens'],
                           hit=row['counters']['prefix_cache_hit_tokens'], saves=row['counters']['prefix_cache_saves'],
                           ttft_ms=round(row['ttft_s'] * 1000, 1) if row['ttft_s'] else None,
                           equal_to_scenario_c_on=row['output_sha256'] == ref.get(key))
            data['rows'].append(summary | dict(output_sha256=row['output_sha256'], memory=row['memory']))
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            time.sleep(1)
    data['server'] = server.summary
    (directory / 'run.json').write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
