#!/usr/bin/env python3
"""Token-id comparison for scenarios.py runs (docs/protocol.md section A).

Requests are matched by position: every session sends the same request
sequence (warmups, then the chains), and the token-id diagnostic writes one
line per completed request in completion order. Prompt token ids must match
for matched requests; then published token ids and the end state (finish
reason, cancellation, failure) are compared."""
import argparse
import hashlib
import json
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]


def load(directory, session):
    text = (directory / f'{session["name"]}.token-ids.jsonl').read_text()
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Only an unfinished final record (server stopped mid-write) is tolerated.
            if index != len(lines) - 1 or text.endswith('\n'):
                raise
            rows.append(None)
    steps = [f'{r["phase"]}:{r["task"]}:{r["step"]}' for r in session['rows']]
    if len(rows) != len(steps):
        raise ValueError(f'{session["name"]}: {len(rows)} token records for {len(steps)} requests')
    out = {}
    for step, record, row in zip(steps, rows, session['rows']):
        common = dict(hit=row['counters']['prefix_cache_hit_tokens'], saves=row['counters']['prefix_cache_saves'],
                      text=row['output_sha256'], finish=row['finish_reason'])
        if record is None:
            out[step] = common | dict(truncated=True)
        else:
            out[step] = common | dict(prompt=record['prompt_token_ids'], tokens=record['published_token_ids'],
                                      end=(row['finish_reason'], record['cancelled'], record['failure']))
    return out


def compare(a, b, steps):
    result = {}
    for step in steps:
        if step not in a or step not in b:
            result[step] = 'missing'
            continue
        if a[step].get('truncated') or b[step].get('truncated'):
            same = a[step]['text'] == b[step]['text'] and a[step]['finish'] == b[step]['finish']
            result[step] = ('text equal' if same else 'text differs') + ' (token record truncated)'
            continue
        if a[step]['prompt'] != b[step]['prompt']:
            result[step] = 'different prompt'
            continue
        same = a[step]['tokens'] == b[step]['tokens'] and a[step]['end'] == b[step]['end']
        if same:
            result[step] = 'equal'
        else:
            first = next((i for i, (x, y) in enumerate(zip(a[step]['tokens'], b[step]['tokens'])) if x != y),
                         min(len(a[step]['tokens']), len(b[step]['tokens'])))
            result[step] = f'differs at token {first} ({len(a[step]["tokens"])} vs {len(b[step]["tokens"])})'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--pairs', nargs='+', required=True, help='A:B session-name pairs')
    p.add_argument('--self', nargs='*', default=[], help='session:stepA=stepB comparisons inside one session')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    directory = SUITE / 'results' / a.label
    raw = (directory / 'run.json').read_bytes()
    data = json.loads(raw)
    sessions = {s['name']: s for s in data['sessions']}
    loaded = {name: load(directory, s) for name, s in sessions.items() if not s.get('failure')}
    report = dict(label=a.label, input_sha256=hashlib.sha256(raw).hexdigest(),
                  failures={n: s.get('failure') for n, s in sessions.items() if s.get('failure')}, pairs={}, self={},
                  hits={n: {k: (v['hit'], v['saves']) for k, v in rows.items()} for n, rows in loaded.items()})
    for pair in a.pairs:
        x, y = pair.split(':')
        steps = [k for k in loaded[x] if not k.startswith('warmup')]
        report['pairs'][pair] = compare(loaded[x], loaded[y], steps)
    for item in a.self:
        session, rest = item.split(':', 1)
        first, second = rest.split('=')
        rows = loaded[session]
        report['self'][item] = (rows[first]['tokens'] == rows[second]['tokens']
                                and rows[first]['end'] == rows[second]['end'])
    with a.output.open('x') as f:
        json.dump(report, f, indent=1)
        f.write('\n')
    for pair, result in report['pairs'].items():
        print(pair)
        for step, verdict in result.items():
            print(f'  {step:32s} {verdict}')
    for item, verdict in report['self'].items():
        print(item, verdict)
    if report['failures']:
        print('failures', report['failures'])


if __name__ == '__main__':
    main()
