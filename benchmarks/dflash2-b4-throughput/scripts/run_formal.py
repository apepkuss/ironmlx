#!/usr/bin/env python3
"""Formal per-item measurement (docs/protocol.md). A = last accepted
version, B = candidate. B4 sessions: 8 paired rounds AB BA BA AB AB BA BA AB,
each a fresh server (capacity 4, 8192): warmup batch, the three batch sets
(start rotated by round), then the B1 short tasks one at a time. B1 long
sessions: 4 paired rounds AB BA BA AB, fresh server (capacity 4, 40960): one
short warmup, then 30K code-1 once. No diagnostics; never retries; a failed
session stops the run."""
import argparse
import json
from pathlib import Path
import time

import common as C

B4_ORDER = ('AB', 'BA', 'BA', 'AB', 'AB', 'BA', 'BA', 'AB')
LONG_ORDER = ('AB', 'BA', 'BA', 'AB')
B1_SHORT = ('code-1', 'knowledge-1', 'code-3')
LONG_TASK, LONG_MAX_TOKENS, LONG_CAPACITY = 'code-1', 1024, 40960


def plan():
    sessions = [dict(kind='b4', round=r, arm=arm) for r, pair in enumerate(B4_ORDER) for arm in pair]
    sessions += [dict(kind='long', round=r, arm=arm) for r, pair in enumerate(LONG_ORDER) for arm in pair]
    return sessions


def b4_session(server, fx, round_id, record, save):
    sets = C.BATCH_SETS[round_id % 3:] + C.BATCH_SETS[:round_id % 3]
    for kind, tasks in [('warmup', C.WARMUP_BATCH)] + [('scored', s) for s in sets]:
        before = C.dflash2_state(C.healthz())
        begin = time.monotonic()
        rows, metrics = C.batch([fx['short'][t] for t in tasks])
        end = time.monotonic()
        after = C.dflash2_state(C.healthz())
        for row, task in zip(rows, tasks):
            row['task'] = task
            row['valid'] = C.valid(row)
        record['batches'].append(dict(kind=kind, tasks=list(tasks), metrics=metrics, rows=rows, before=before,
                                      after=after, memory=server.memory_window(begin, end)))
        save()
        if not all(r['valid'] for r in rows):
            raise RuntimeError(f'invalid response in batch {tasks}')
        time.sleep(2)
    for task in B1_SHORT:
        begin = time.monotonic()
        row = C.chat(fx['short'][task])
        end = time.monotonic()
        row.update(task=task, valid=C.valid(row), memory=server.memory_window(begin, end))
        record['b1'].append(row)
        save()
        if not row['valid']:
            raise RuntimeError(f'invalid B1 response {task}')
        time.sleep(1)


def long_session(server, fx, record, save):
    warm = C.chat(C.WARMUPS['warmup-1'], max_tokens=64)
    record['warmup'] = dict(valid=warm['error'] is None and warm['done'], ttft_s=warm['ttft_s'])
    time.sleep(1)
    begin = time.monotonic()
    row = C.chat(fx['30k'][LONG_TASK], max_tokens=LONG_MAX_TOKENS)
    end = time.monotonic()
    row.update(task=LONG_TASK, valid=C.valid(row), memory=server.memory_window(begin, end))
    record['b1'].append(row)
    save()
    if not row['valid']:
        raise RuntimeError('invalid 30K response')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    binaries = {'A': a.a, 'B': a.b}
    data = dict(label=a.label, purpose='formal', protocol=str(a.protocol), protocol_sha256=C.digest(a.protocol),
                scripts={q.name: C.digest(q) for q in sorted((C.SUITE / 'scripts').glob('*.py'))},
                binaries={k: dict(path=str(v), sha256=C.digest(v)) for k, v in binaries.items()},
                model_identity=C.model_identity(), plan=plan(), complete=False, failure=None,
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sessions=[])
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    save()
    try:
        for index, planned in enumerate(data['plan']):
            for name, info in data['binaries'].items():
                if C.digest(info['path']) != info['sha256']:
                    raise RuntimeError(f'binary {name} changed during the run')
            record = dict(index=index, **planned, batches=[], b1=[])
            data['sessions'].append(record)
            capacity = 8192 if planned['kind'] == 'b4' else LONG_CAPACITY
            server = C.Server(directory, f"s{index:02d}-{planned['kind']}-{planned['arm']}",
                              binaries[planned['arm']], 4, capacity)
            try:
                with server:
                    record.update(command=server.cmd, env_keys=sorted(server.env))
                    if planned['kind'] == 'b4':
                        b4_session(server, fx, planned['round'], record, save)
                    else:
                        long_session(server, fx, record, save)
                    record['lifecycle_peak_bytes'] = server.lifecycle_peak()
            finally:
                record['server'] = server.summary
                save()
            print(json.dumps(dict(session=index, kind=planned['kind'], round=planned['round'], arm=planned['arm'],
                                  done=True), ensure_ascii=False), flush=True)
            time.sleep(10)
        data['complete'] = True
    except BaseException as error:
        data['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        data['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save()


if __name__ == '__main__':
    main()
