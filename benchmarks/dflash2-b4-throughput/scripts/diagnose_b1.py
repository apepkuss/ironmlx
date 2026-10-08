#!/usr/bin/env python3
"""Diagnosis (not a formal measurement) of B1 short TTFT. Arms A/B, 3 rounds
AB BA AB, two session kinds per arm and round:
  after_b4 - exactly run_formal.py's B4 session prefix (warmup batch, the
             three batch sets) then B1 short tasks;
  fresh    - only a short warmup, then the B1 short tasks.
B1 short: code-1, knowledge-1, code-3, each 3 times in that order. Per request
client TTFT, server prefill time (/healthz dflash2.prefill_us delta) and MLX
active/cache bytes before the request."""
import argparse
import json
from pathlib import Path
import time

import common as C

TASKS = ('code-1', 'knowledge-1', 'code-3')
ORDER = ('AB', 'BA', 'AB')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    binaries = {'A': a.a, 'B': a.b}
    data = dict(label=a.label, purpose='b1-ttft-diagnosis',
                binaries={k: dict(path=str(v), sha256=C.digest(v)) for k, v in binaries.items()}, sessions=[])
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    index = 0
    for round_id, pair in enumerate(ORDER):
        for kind in ('after_b4', 'fresh'):
            for arm in pair:
                record = dict(index=index, round=round_id, kind=kind, arm=arm, rows=[])
                data['sessions'].append(record)
                server = C.Server(directory, f's{index:02d}-{kind}-{arm}', binaries[arm], 4, 8192)
                try:
                    with server:
                        if kind == 'after_b4':
                            for tasks in [C.WARMUP_BATCH] + list(C.BATCH_SETS[round_id % 3:] +
                                                                 C.BATCH_SETS[:round_id % 3]):
                                C.batch([fx['short'][t] for t in tasks])
                                time.sleep(2)
                        else:
                            C.chat(C.WARMUPS['warmup-1'], max_tokens=64)
                            time.sleep(1)
                        for rep in range(3):
                            for task in TASKS:
                                before = C.healthz()
                                row = C.chat(fx['short'][task])
                                after = C.healthz()
                                record['rows'].append(dict(
                                    task=task, rep=rep, ttft_ms=row['ttft_s'] * 1000, valid=C.valid(row),
                                    prefill_ms=(after['dflash2']['prefill_us'] - before['dflash2']['prefill_us']) / 1000,
                                    mlx_before={k: before['memory'].get(k) for k in ('mlx_active_bytes',
                                                                                     'mlx_cache_bytes')},
                                    output_sha256=row['output_sha256']))
                                save()
                                time.sleep(1)
                finally:
                    record['server'] = server.summary
                    save()
                print(json.dumps(dict(session=index, round=round_id, kind=kind, arm=arm), ensure_ascii=False),
                      flush=True)
                index += 1
                time.sleep(10)
    data['complete'] = True
    save()


if __name__ == '__main__':
    main()
