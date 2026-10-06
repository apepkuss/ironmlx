#!/usr/bin/env python3
"""Formal timing (docs/protocol.md section B). One unscored adaptation pair,
then six scored rounds BC, CB, CB, BC, BC, CB. App-default server with the
prefix cache on, App-like environment, no diagnostics. Never retries; a
failed session is recorded and the run stops."""
import argparse
import json
from pathlib import Path
import time

import common as C

ORDERS = (('B', 'C'), ('C', 'B'), ('C', 'B'), ('B', 'C'), ('B', 'C'), ('C', 'B'))
WARM_MISSES = ('code-1', 'code-3', 'knowledge-1', 'knowledge-2', 'knowledge-3')
REPEATS = ('code-1', 'knowledge-1')
Q2 = '请用一句话总结上面的回答。'


def plan():
    sessions = [dict(round=None, arm=arm, scored=False) for arm in ('B', 'C')]
    for index, order in enumerate(ORDERS):
        sessions += [dict(round=index, arm=arm, scored=True) for arm in order]
    return sessions


def session_steps(fx):
    steps = [('first', 'short', 'code-2', C.user(fx['short']['code-2']['prompt']), 4096)]
    steps += [('warmup', 'warmup', f'warmup-{i}', C.user(text), 64) for i, text in enumerate(C.WARMUPS)]
    steps += [('miss', 'short', t, C.user(fx['short'][t]['prompt']), 4096) for t in WARM_MISSES]
    steps += [('repeat', 'short', t, C.user(fx['short'][t]['prompt']), 4096) for t in REPEATS]
    steps += [('append', 'short', 'code-1', None, 4096)]
    steps += [('miss', '30k', 'code-1', C.user(fx['30k']['code-1']['prompt']), 1024),
              ('repeat', '30k', 'code-1', C.user(fx['30k']['code-1']['prompt']), 1024)]
    return steps


def audit(row):
    errors = [] if C.valid(row) else ['invalid response']
    c = row['counters']
    if row['kind'] in ('first', 'miss') and (c['prefix_cache_hit_tokens'] != 0 or c['prefix_cache_misses'] != 1):
        errors.append('measured cold miss is not a zero-hit miss')
    if row['kind'] == 'repeat' and c['prefix_cache_hit_tokens'] != row['prompt_tokens']:
        errors.append('exact repeat did not hit the whole prompt')
    return errors


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
    data = dict(label=a.label, purpose='formal-timing', protocol_sha256=C.digest(C.SUITE / 'docs/protocol.md'),
                scripts={p.name: C.digest(p) for p in sorted((C.SUITE / 'scripts').glob('*.py'))},
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
            record = dict(index=index, **planned, rows=[])
            data['sessions'].append(record)
            server = C.Server(directory, f's{index:02d}-{planned["arm"]}', binaries[planned['arm']], True)
            try:
                with server:
                    record.update(command=server.cmd, env_keys=sorted(server.env), ready_s=server.ready - server.started)
                    answers = {}
                    for kind, phase, task, messages, max_tokens in session_steps(fx):
                        if kind == 'append':
                            messages = (C.user(fx['short'][task]['prompt']) +
                                        [dict(role='assistant', content=answers[task]), dict(role='user', content=Q2)])
                        row = dict(kind=kind, phase=phase, task=task) | server.request(messages, max_tokens)
                        if kind == 'miss' and phase == 'short':
                            answers[task] = row['output']
                        row['audit'] = [] if kind == 'warmup' else audit(row)
                        record['rows'].append(row)
                        save()
                        print(json.dumps(dict(session=index, round=planned['round'], arm=planned['arm'], kind=kind,
                                              phase=phase, task=task, ttft_ms=round((row['ttft_s'] or 0) * 1000, 1),
                                              decode=round(row['decode_tps'] or 0, 1),
                                              hit=row['counters']['prefix_cache_hit_tokens'], audit=row['audit']),
                                         ensure_ascii=False), flush=True)
                        if row['audit']:
                            raise RuntimeError(f'audit failed: {row["audit"]}')
                        time.sleep(1)
                    samples = C.read_samples(server.memory_path, live=True)
                    record['lifecycle_peak_footprint_bytes'] = max(s['tree_footprint_bytes'] for s in samples)
            finally:
                record['server'] = getattr(server, 'summary', None)
                save()
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
