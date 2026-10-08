#!/usr/bin/env python3
"""L2 prefill chunk-size screen (docs/protocol-l2-chunk-screen.md).
--kind perf: 6 Latin-square rounds x {2048, 4096, 8192}, no diagnostics.
--kind correctness: one session per size with the token-id diagnostic.
--kind path: one session per size with the prefill-phase diagnostic.
Each session: fresh server (`--max-sequences 1`, capacity 40960,
`--prefill-chunk-size K`), short warmup, 30K code-1 and knowledge-1
(max_tokens 1024). Never retries; an invalid session is recorded and the run
continues."""
import argparse
import json
from pathlib import Path
import re
import time

import common as C

SIZES = (2048, 4096, 8192)
ORDERS = ((2048, 4096, 8192), (4096, 8192, 2048), (8192, 2048, 4096),
          (2048, 8192, 4096), (4096, 2048, 8192), (8192, 4096, 2048))
TASKS = ('code-1', 'knowledge-1')
EXPECTED = '8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65'
WITNESSES = ('gate-up QMM BM128 active', 'down QMM BM128 active', '2048-row BF16 D256 NAX attention active',
             'realized-layout native fallback', 'masked causal softmax fallback active')


def plan(kind):
    if kind == 'perf':
        return [dict(round=r, size=size) for r, order in enumerate(ORDERS) for size in order]
    return [dict(round=None, size=size) for size in SIZES]


def witnesses(log_path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
    return {w: w in text for w in WITNESSES}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--kind', choices=('perf', 'correctness', 'path'), required=True)
    p.add_argument('--binary', type=Path, required=True)
    a = p.parse_args()
    if C.digest(a.binary) != EXPECTED:
        raise SystemExit('frozen per-item baseline binary required')
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    data = dict(label=a.label, kind=a.kind, protocol_sha256=C.digest(C.SUITE / 'docs/protocol-l2-chunk-screen.md'),
                scripts={q.name: C.digest(q) for q in sorted((C.SUITE / 'scripts').glob('*.py'))},
                binary=dict(path=str(a.binary), sha256=EXPECTED), model_identity=C.model_identity(),
                plan=plan(a.kind), complete=False, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                sessions=[])
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    save()
    for index, planned in enumerate(data['plan']):
        name = f"s{index:02d}-{planned['size']}"
        env = {}
        if a.kind == 'correctness':
            env['IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS'] = str(directory / f'{name}.token-ids.jsonl')
        if a.kind == 'path':
            env['IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES'] = '1'
        record = dict(index=index, **planned, rows=[], invalid=None)
        data['sessions'].append(record)
        server = C.Server(directory, name, a.binary, 1, 40960, env,
                          extra_args=('--prefill-chunk-size', str(planned['size'])))
        try:
            with server:
                record.update(command=server.cmd, env_keys=sorted(server.env))
                C.chat(C.WARMUPS['warmup-1'], max_tokens=64)
                time.sleep(2)
                for task in TASKS:
                    before = C.dflash2_state(C.healthz())
                    begin = time.monotonic()
                    row = C.chat(fx['30k'][task], max_tokens=1024)
                    end = time.monotonic()
                    after = C.dflash2_state(C.healthz())
                    row.pop('output', None)
                    row.update(task=task, valid=C.valid(row),
                               server_prefill_ms=(after['prefill_us'] - before['prefill_us']) / 1000,
                               server_generation_ms=(after['generation_us'] - before['generation_us']) / 1000,
                               mlx_after=after['memory'],
                               memory=server.memory_window(begin, end))
                    record['rows'].append(row)
                    save()
                    if not row['valid']:
                        raise RuntimeError(f'invalid response {task}')
                    time.sleep(5)
                record['lifecycle_peak_bytes'] = server.lifecycle_peak()
                if a.kind == 'correctness':  # wait for the warmup + task records to be complete
                    ids = Path(env['IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS'])
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        raw = ids.read_text() if ids.exists() else ''
                        if raw.endswith('\n') and raw.count('\n') >= 1 + len(TASKS):
                            break
                        time.sleep(0.5)
        except Exception as error:  # invalid session: kept, never retried
            record['invalid'] = f'{type(error).__name__}: {error}'
        finally:
            record['server'] = server.summary
            if server.log_path.exists():
                record['witnesses'] = witnesses(server.log_path)
            save()
        print(json.dumps(dict(session=index, size=planned['size'], round=planned['round'],
                              valid=record['invalid'] is None), ensure_ascii=False), flush=True)
        time.sleep(10)
    data['complete'] = True
    save()


if __name__ == '__main__':
    main()
