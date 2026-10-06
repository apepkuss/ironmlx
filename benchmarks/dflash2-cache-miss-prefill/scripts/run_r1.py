#!/usr/bin/env python3
"""R1 confirmation run (docs/protocol-r1-confirmation.md). Sessions are
exactly run_timing.py's sessions (same steps, cache formation, gaps); R1 is
the 30K exact repeat right after the 30K cold miss. One unscored BC
adaptation pair, then 24 scored rounds in the pre-generated order. An invalid
session is recorded and the run continues without retry; a binary identity
change stops the run. Progress lines carry no timing values."""
import argparse
import json
from pathlib import Path
import time

import common as C
import run_timing as T

ORDER_FILE = C.SUITE / 'docs/r1-confirmation-order.json'
ORDER_SHA = 'c8104dab856b77b6b606c368947b98ebdd2f500b0a0f8c8bc0776b5cc7f0998f'
EXPECTED = {'B': 'b9c7b4fb55bc5b6e51757db0361cb08ef8a1bcc1e5f66698f2fc6f264b8bdf85',
            'C': 'f007c386ade02cd7c0168d1029d73eac419fc738d0b5684bca692efbd1ad4074'}


def plan():
    if C.digest(ORDER_FILE) != ORDER_SHA:
        raise RuntimeError('order file differs from the registered one')
    order = json.loads(ORDER_FILE.read_text())
    sessions = [dict(round=None, arm=arm, scored=False) for arm in order['adaptation']]
    for index, pair in enumerate(order['rounds']):
        sessions += [dict(round=index, arm=arm, scored=True) for arm in pair]
    return sessions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    directory = C.SUITE / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    binaries = {'B': a.baseline, 'C': a.candidate}
    identity = {k: C.digest(v) for k, v in binaries.items()}
    if identity != EXPECTED:
        raise RuntimeError('frozen formal binaries required')
    data = dict(label=a.label, purpose='r1-confirmation',
                protocol_sha256=C.digest(C.SUITE / 'docs/protocol-r1-confirmation.md'),
                order_sha256=ORDER_SHA,
                scripts={q.name: C.digest(q) for q in sorted((C.SUITE / 'scripts').glob('*.py'))},
                binaries={k: dict(path=str(v), sha256=identity[k]) for k, v in binaries.items()},
                model_identity=C.model_identity(), plan=plan(), complete=False, stopped=None,
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sessions=[])
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    save()
    steps = T.session_steps(fx)
    try:
        for index, planned in enumerate(data['plan']):
            for name, info in data['binaries'].items():
                if C.digest(info['path']) != info['sha256']:
                    data['stopped'] = f'binary {name} changed'
                    raise RuntimeError(data['stopped'])
            record = dict(index=index, **planned, rows=[], invalid=None)
            data['sessions'].append(record)
            server = C.Server(directory, f's{index:02d}-{planned["arm"]}', binaries[planned['arm']], True)
            try:
                with server:
                    record.update(command=server.cmd, env_keys=sorted(server.env))
                    answers = {}
                    for kind, phase, task, messages, max_tokens in steps:
                        if kind == 'append':
                            messages = (C.user(fx['short'][task]['prompt']) +
                                        [dict(role='assistant', content=answers[task]),
                                         dict(role='user', content=T.Q2)])
                        row = dict(kind=kind, phase=phase, task=task) | server.request(messages, max_tokens)
                        if kind == 'miss' and phase == 'short':
                            answers[task] = row['output']
                        row['audit'] = [] if kind == 'warmup' else T.audit(row)
                        record['rows'].append(row)
                        save()
                        if row['audit']:
                            raise RuntimeError(f'{kind} {phase} {task}: {row["audit"]}')
                        time.sleep(1)
            except Exception as error:  # invalid session: recorded, never retried
                record['invalid'] = f'{type(error).__name__}: {error}'
            finally:
                record['server'] = getattr(server, 'summary', None)
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
