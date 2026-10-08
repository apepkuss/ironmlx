#!/usr/bin/env python3
"""B1 short TTFT confirmation (docs/protocol-b1-ttft-confirmation.md).
24 paired rounds, order AB BA BA AB AB BA BA AB repeated three times; each
session is run_formal.py's B4 session unchanged (fresh server, warmup batch,
three batch sets rotated by round % 3, B1 short code-1, knowledge-1,
code-3). Run once; an invalid session is recorded and the run continues
without retry; a binary identity change stops the run."""
import argparse
import json
from pathlib import Path
import time

import common as C
import run_formal as F

ORDER = ('AB', 'BA', 'BA', 'AB', 'AB', 'BA', 'BA', 'AB') * 3
EXPECTED = {'A': '4fc49e9223f62468cebae3837ed7c1867d4047821a29405903502a41e9f5b487',
            'B': '8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65'}


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
    identity = {k: C.digest(v) for k, v in binaries.items()}
    if identity != EXPECTED:
        raise RuntimeError('frozen binaries required')
    protocol = C.SUITE / 'docs/protocol-b1-ttft-confirmation.md'
    data = dict(label=a.label, purpose='b1-ttft-confirmation', protocol_sha256=C.digest(protocol),
                scripts={q.name: C.digest(q) for q in sorted((C.SUITE / 'scripts').glob('*.py'))},
                binaries={k: dict(path=str(v), sha256=identity[k]) for k, v in binaries.items()},
                model_identity=C.model_identity(),
                plan=[dict(kind='b4', round=r, arm=arm) for r, pair in enumerate(ORDER) for arm in pair],
                complete=False, stopped=None, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                sessions=[])
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    save()
    try:
        for index, planned in enumerate(data['plan']):
            for name, info in data['binaries'].items():
                if C.digest(info['path']) != info['sha256']:
                    data['stopped'] = f'binary {name} changed'
                    raise RuntimeError(data['stopped'])
            record = dict(index=index, **planned, batches=[], b1=[], invalid=None)
            data['sessions'].append(record)
            server = C.Server(directory, f"s{index:02d}-{planned['arm']}", binaries[planned['arm']], 4, 8192)
            try:
                with server:
                    record.update(command=server.cmd, env_keys=sorted(server.env))
                    F.b4_session(server, fx, planned['round'], record, save)
                    record['lifecycle_peak_bytes'] = server.lifecycle_peak()
            except Exception as error:  # invalid session: kept, never retried
                record['invalid'] = f'{type(error).__name__}: {error}'
            finally:
                record['server'] = server.summary
                save()
            print(json.dumps(dict(session=index, round=planned['round'], arm=planned['arm'],
                                  valid=record['invalid'] is None, invalid=record['invalid']),
                             ensure_ascii=False), flush=True)
            time.sleep(10)
        data['complete'] = True
    finally:
        data['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save()


if __name__ == '__main__':
    main()
