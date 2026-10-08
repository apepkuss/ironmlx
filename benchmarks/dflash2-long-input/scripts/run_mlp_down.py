#!/usr/bin/env python3
"""Formal A/B for the mlp_down QMM path change (docs/protocol-mlp-down.md).
A = accepted B4 build 8d3a42dc, B = candidate.
Part 1, 30K: 12 paired rounds (6 AB, 6 BA); each arm session is a fresh
server (--max-sequences 1, capacity 40960, --prefill-chunk-size 2048): one
short warmup, then 30K code-1 and knowledge-1 (max_tokens 1024) in the
round's task order, 5 s apart.
Part 2, short: 6 paired rounds (AB BA BA AB AB BA); B4-stage sessions
(capacity 4, 8192): warmup batch, the three batch sets (start rotated by
round), then the B1 short tasks one at a time.
--kind correctness (run first): one 30K session and one short session per arm
with the token-id record (IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS); every
request's prompt ids, published ids and end state are compared A vs B.
--kind formal: the paired rounds above with no diagnostics at all; after each
pair, every request's output text hash is compared with the same round's
baseline request.
Never retries. An invalid session, a binary change, a token-id divergence or
an output-hash mismatch stops the run; the evidence is kept."""
import argparse
import json
from pathlib import Path
import time

import common as C

LONG_ORDER = ('AB', 'BA', 'BA', 'AB', 'AB', 'BA', 'BA', 'AB', 'AB', 'BA', 'BA', 'AB')
LONG_TASK_ORDER = (('code-1', 'knowledge-1'), ('code-1', 'knowledge-1'), ('knowledge-1', 'code-1'),
                   ('knowledge-1', 'code-1')) * 3
SHORT_ORDER = ('AB', 'BA', 'BA', 'AB', 'AB', 'BA')
B1_SHORT = ('code-1', 'knowledge-1', 'code-3')
EXPECTED = {'A': '8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65'}
TOKEN_ENV = 'IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS'


def plan(kind):
    if kind == 'correctness':
        return [dict(kind='long', round=0, arm=arm, tasks=list(LONG_TASK_ORDER[0])) for arm in 'AB'] + \
            [dict(kind='short', round=0, arm=arm) for arm in 'AB']
    sessions = [dict(kind='long', round=r, arm=arm, tasks=list(LONG_TASK_ORDER[r]))
                for r, pair in enumerate(LONG_ORDER) for arm in pair]
    sessions += [dict(kind='short', round=r, arm=arm) for r, pair in enumerate(SHORT_ORDER) for arm in pair]
    return sessions


def long_session(server, fx, tasks, record, save):
    warm = C.chat(C.WARMUPS['warmup-1'], max_tokens=64)
    record['warmup'] = dict(valid=warm['error'] is None and warm['done'], ttft_s=warm['ttft_s'])
    time.sleep(2)
    for task in tasks:
        begin = time.monotonic()
        row = C.chat(fx['30k'][task], max_tokens=1024)
        end = time.monotonic()
        row.pop('output', None)
        row.update(task=task, valid=C.valid(row), memory=server.memory_window(begin, end))
        record['b1'].append(row)
        save()
        if not row['valid']:
            raise RuntimeError(f'invalid 30K response {task}')
        time.sleep(5)


def short_session(server, fx, round_id, record, save):
    sets = C.BATCH_SETS[round_id % 3:] + C.BATCH_SETS[:round_id % 3]
    for kind, tasks in [('warmup', C.WARMUP_BATCH)] + [('scored', s) for s in sets]:
        before = C.dflash2_state(C.healthz())
        begin = time.monotonic()
        rows, metrics = C.batch([fx['short'][t] for t in tasks])
        end = time.monotonic()
        after = C.dflash2_state(C.healthz())
        for row, task in zip(rows, tasks):
            row.pop('output', None)
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
        row.pop('output', None)
        row.update(task=task, valid=C.valid(row), memory=server.memory_window(begin, end))
        record['b1'].append(row)
        save()
        if not row['valid']:
            raise RuntimeError(f'invalid B1 response {task}')
        time.sleep(1)


def token_records(path, expected):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        raw = path.read_text() if path.exists() else ''
        if raw.endswith('\n') and raw.count('\n') >= expected:
            break
        time.sleep(0.5)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def keyed(records):
    seen = {}
    out = {}
    for record in records:
        prompt = tuple(record['prompt_token_ids'])
        out[(prompt, seen.get(prompt, 0))] = record
        seen[prompt] = seen.get(prompt, 0) + 1
    return out


def compare_tokens(a_records, b_records):
    """Pair requests by (prompt token ids, occurrence): batched completion order may differ between arms."""
    a_keyed, b_keyed = keyed(a_records), keyed(b_records)
    out = []
    for key in sorted(set(a_keyed) | set(b_keyed), key=lambda k: (len(k[0]), k)):
        ra, rb = a_keyed.get(key), b_keyed.get(key)
        if ra is None or rb is None:
            out.append(dict(prompt_tokens=len(key[0]), occurrence=key[1], status='missing'))
            continue
        same = (ra['published_token_ids'] == rb['published_token_ids']
                and ra['cancelled'] == rb['cancelled'] and ra['failure'] == rb['failure'])
        first = next((j for j, (x, y) in enumerate(zip(ra['published_token_ids'], rb['published_token_ids']))
                      if x != y), None)
        out.append(dict(prompt_tokens=len(key[0]), occurrence=key[1], status='equal' if same else 'differs',
                        tokens=[len(ra['published_token_ids']), len(rb['published_token_ids'])],
                        first_divergence=first))
    return out


def outputs(session):
    """(where, task, occurrence) -> output hash for every request of a session."""
    out, seen = {}, {}
    rows = [(b['kind'] + ':' + ','.join(b['tasks']), r) for b in session['batches'] for r in b['rows']]
    rows += [('b1', r) for r in session['b1']]
    for where, row in rows:
        key = (where, row['task'])
        seen[key] = seen.get(key, 0) + 1
        out[key + (seen[key],)] = row['output_sha256']
    return out


def compare_hashes(session_a, session_b):
    a_out, b_out = outputs(session_a), outputs(session_b)
    return [dict(where=k[0], task=k[1], occurrence=k[2],
                 status='equal' if a_out.get(k) == b_out.get(k) else
                 ('missing' if k not in a_out or k not in b_out else 'differs'))
            for k in sorted(set(a_out) | set(b_out))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--b-sha256', required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--kind', choices=('correctness', 'formal'), required=True)
    a = p.parse_args()
    binaries = {'A': a.a, 'B': a.b}
    expected = dict(EXPECTED, B=a.b_sha256)
    for arm, path in binaries.items():
        if C.digest(path) != expected[arm]:
            raise SystemExit(f'binary {arm} identity mismatch')
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    data = dict(label=a.label, purpose='formal-mlp-down', protocol=str(a.protocol),
                protocol_sha256=C.digest(a.protocol), kind=a.kind,
                token_ids='on' if a.kind == 'correctness' else 'off',
                scripts={q.name: C.digest(q) for q in sorted((C.SUITE / 'scripts').glob('*.py'))},
                binaries={k: dict(path=str(v), sha256=expected[k]) for k, v in binaries.items()},
                model_identity=C.model_identity(), complete=False, failure=None, plan=plan(a.kind),
                token_comparisons=[], hash_comparisons=[], started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sessions=[])
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    save()
    tokens = {}
    try:
        for index, planned in enumerate(data['plan']):
            for name, info in data['binaries'].items():
                if C.digest(info['path']) != info['sha256']:
                    raise RuntimeError(f'binary {name} changed during the run')
            record = dict(index=index, **planned, batches=[], b1=[])
            data['sessions'].append(record)
            name = f"s{index:02d}-{planned['kind']}-r{planned['round']:02d}-{planned['arm']}"
            env = {}
            ids = directory / f'{name}.token-ids.jsonl'
            if a.kind == 'correctness':
                env[TOKEN_ENV] = str(ids)
            if planned['kind'] == 'long':
                server = C.Server(directory, name, binaries[planned['arm']], 1, 40960, env,
                                  extra_args=('--prefill-chunk-size', '2048'))
            else:
                server = C.Server(directory, name, binaries[planned['arm']], 4, 8192, env)
            try:
                with server:
                    record.update(command=server.cmd, env_keys=sorted(server.env))
                    if planned['kind'] == 'long':
                        long_session(server, fx, planned['tasks'], record, save)
                        expected_records = 1 + len(planned['tasks'])
                    else:
                        short_session(server, fx, planned['round'], record, save)
                        expected_records = 4 * 4 + len(B1_SHORT)
                    record['lifecycle_peak_bytes'] = server.lifecycle_peak()
                    if a.kind == 'correctness':
                        tokens[(planned['kind'], planned['round'], planned['arm'])] = token_records(ids, expected_records)
            finally:
                record['server'] = server.summary
                save()
            print(json.dumps(dict(session=index, kind=planned['kind'], round=planned['round'], arm=planned['arm'],
                                  done=True), ensure_ascii=False), flush=True)
            key_a = (planned['kind'], planned['round'], 'A')
            key_b = (planned['kind'], planned['round'], 'B')
            if a.kind == 'correctness' and key_a in tokens and key_b in tokens:
                comparison = compare_tokens(tokens[key_a], tokens[key_b])
                data['token_comparisons'].append(dict(kind=planned['kind'], round=planned['round'],
                                                      comparisons=comparison))
                save()
                if any(c['status'] != 'equal' for c in comparison):
                    raise RuntimeError(f"token-id divergence in {planned['kind']} round {planned['round']}")
            pair = [x for x in data['sessions'] if x['kind'] == planned['kind'] and x['round'] == planned['round']]
            if a.kind == 'formal' and len(pair) == 2:
                comparison = compare_hashes(*sorted(pair, key=lambda x: x['arm']))
                data['hash_comparisons'].append(dict(kind=planned['kind'], round=planned['round'],
                                                     comparisons=comparison))
                save()
                if any(c['status'] != 'equal' for c in comparison):
                    raise RuntimeError(f"output-hash mismatch in {planned['kind']} round {planned['round']}")
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
