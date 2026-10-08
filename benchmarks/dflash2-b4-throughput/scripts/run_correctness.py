#!/usr/bin/env python3
"""Token-id correctness (docs/protocol.md). Per arm: one B4 session (warmup
batch, the three batch sets, then the B1 short tasks one at a time) and one
long session (30K code-1, capacity 40960), both with
IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS. Records are matched between arms by
prompt token ids and occurrence (batched completion order may differ); the
published token ids and the end state must be equal. Draft acceptance per
arm is reported."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import common as C
import run_formal as F


def load_records(path, expected):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        raw = path.read_text() if path.exists() else ''
        if raw.endswith('\n') and raw.count('\n') >= expected:
            break
        time.sleep(0.5)
    return [json.loads(line) for line in path.read_text().splitlines()]


def keyed(records):
    seen = defaultdict(int)
    out = {}
    for record in records:
        prompt = tuple(record['prompt_token_ids'])
        out[(prompt, seen[prompt])] = record
        seen[prompt] += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    data = dict(label=a.label, purpose='token-id-correctness',
                binaries={'A': dict(path=str(a.a), sha256=C.digest(a.a)), 'B': dict(path=str(a.b), sha256=C.digest(a.b))},
                sessions=[], complete=False)
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    records = {}
    for arm, binary in (('A', a.a), ('B', a.b)):
        records[arm] = []
        for kind in ('b4', 'long'):
            path = directory / f'{arm}-{kind}.token-ids.jsonl'
            record = dict(arm=arm, kind=kind, batches=[], b1=[])
            data['sessions'].append(record)
            capacity = 8192 if kind == 'b4' else F.LONG_CAPACITY
            server = C.Server(directory, f'{arm}-{kind}', binary, 4, capacity,
                              {'IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS': str(path)})
            try:
                with server:
                    if kind == 'b4':
                        F.b4_session(server, fx, 0, record, save)
                        expected = 4 * 4 + len(F.B1_SHORT)
                    else:
                        F.long_session(server, fx, record, save)
                        expected = 2
                    records[arm] += load_records(path, expected)
            finally:
                record['server'] = server.summary
                save()
            time.sleep(10)
    a_keyed, b_keyed = keyed(records['A']), keyed(records['B'])
    comparisons = []
    for key in sorted(set(a_keyed) | set(b_keyed), key=lambda k: (len(k[0]), k)):
        ra, rb = a_keyed.get(key), b_keyed.get(key)
        if ra is None or rb is None:
            comparisons.append(dict(prompt_tokens=len(key[0]), occurrence=key[1], status='missing'))
            continue
        same = (ra['published_token_ids'] == rb['published_token_ids'] and ra['cancelled'] == rb['cancelled']
                and ra['failure'] == rb['failure'])
        comparisons.append(dict(prompt_tokens=len(key[0]), occurrence=key[1], status='equal' if same else 'differs',
                                tokens=[len(ra['published_token_ids']), len(rb['published_token_ids'])]))
    acceptance = {}
    for arm in 'AB':
        drafted = sum(r['metrics'].get('drafted_tokens', 0) for r in records[arm])
        accepted = sum(r['metrics'].get('accepted_draft_tokens', 0) for r in records[arm])
        acceptance[arm] = dict(drafted=drafted, accepted=accepted, rate=accepted / drafted if drafted else None)
    data.update(comparisons=comparisons, all_equal=all(c['status'] == 'equal' for c in comparisons),
                acceptance=acceptance, complete=True)
    save()
    print(json.dumps(dict(all_equal=data['all_equal'], compared=len(comparisons),
                          differs=[c for c in comparisons if c['status'] != 'equal'], acceptance=acceptance),
                     indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
