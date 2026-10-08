#!/usr/bin/env python3
"""Analysis of the L2 chunk-size screen (docs/protocol-l2-chunk-screen.md).
perf: per size and task the median over rounds; 4096 and 8192 against 2048
(geometric mean over tasks), 95% interval by paired-round bootstrap (10000
draws, seed 20261007); screening rules as registered.
correctness: per task, prompt / published token ids and end state against
2048. path: realized prefill chunks and witness lines per size."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

import common as C
from run_l2 import WITNESSES

DRAWS, SEED = 10000, 20261007
SIZES, TASKS = (2048, 4096, 8192), ('code-1', 'knowledge-1')
METRICS = {
    'ttft': lambda r: r['ttft_s'],
    'decode_tps': lambda r: r['decode_tps'],
    'e2e': lambda r: r['e2e_s'],
    'server_prefill': lambda r: r['server_prefill_ms'],
    'request_peak': lambda r: r['memory'].get('observed_peak_footprint_bytes'),
    'mlx_peak': lambda r: r['mlx_after']['mlx_peak_bytes'],
}


def ratio(cells, size, value, rounds, keys):
    def point(idx):
        logs = []
        for key in keys:
            b = statistics.median(value(cells[i][size][key]) for i in idx)
            a = statistics.median(value(cells[i][2048][key]) for i in idx)
            logs.append(math.log(b / a))
        return math.exp(sum(logs) / len(logs))
    rng = random.Random(SEED)
    boot = sorted(point(rng.choices(rounds, k=len(rounds))) for _ in range(DRAWS))
    return dict(ratio=point(rounds), ci95=[boot[int(DRAWS * 0.025)], boot[int(DRAWS * 0.975) - 1]])


def perf(data):
    cells, invalid = {}, {s: [] for s in SIZES}
    for s in data['sessions']:
        if s['invalid']:
            invalid[s['size']].append(dict(index=s['index'], round=s['round'], reason=s['invalid']))
            continue
        cell = {r['task']: r for r in s['rows']}
        cell['lifecycle'] = {'lifecycle_peak_bytes': s['lifecycle_peak_bytes']}
        cells.setdefault(s['round'], {})[s['size']] = cell
    rounds = sorted(r for r, c in cells.items() if all(size in c for size in SIZES))
    medians = {}
    for size in SIZES:
        medians[size] = {task: {m: statistics.median(f(cells[r][size][task]) for r in rounds)
                                for m, f in METRICS.items()} for task in TASKS}
        medians[size]['lifecycle_peak_bytes'] = statistics.median(
            cells[r][size]['lifecycle']['lifecycle_peak_bytes'] for r in rounds)
        medians[size]['witnesses'] = sorted({w for r in rounds for w, hit in
                                             (data['sessions_by'][(r, size)].get('witnesses') or {}).items() if hit})
    comparisons = {}
    for size in (4096, 8192):
        out = {m: ratio(cells, size, f, rounds, TASKS) for m, f in METRICS.items()}
        out['lifecycle_peak'] = ratio(cells, size, lambda c: c['lifecycle_peak_bytes'], rounds, ('lifecycle',))
        out['lifecycle_peak_delta_bytes'] = medians[size]['lifecycle_peak_bytes'] - medians[2048]['lifecycle_peak_bytes']
        per_task = {}
        for task in TASKS:
            per_task[task] = {m: dict(ratio=ratio(cells, size, f, rounds, (task,))['ratio'],
                                      faster_rounds=sum(f(cells[r][size][task]) < f(cells[r][2048][task])
                                                        for r in rounds)) for m, f in METRICS.items()}
        out['per_task'] = per_task
        gates = dict(ttft=out['ttft']['ratio'] <= 0.97 and out['ttft']['ci95'][1] < 1.0,
                     e2e=out['e2e']['ratio'] <= 0.98 and out['e2e']['ci95'][1] < 1.0,
                     decode=out['decode_tps']['ci95'][0] >= 0.97,
                     memory=out['lifecycle_peak']['ratio'] <= 1.05)
        out['gates'] = gates
        if len(invalid[size]) > 1 or len(invalid[2048]) > 1:
            out['screen'] = 'evidence-insufficient'
        elif not (gates['ttft'] and gates['e2e']):
            out['screen'] = 'insufficient-benefit'
        else:
            out['screen'] = 'worth-implementing-pending-correctness' if all(gates.values()) else 'gate-failed'
        comparisons[size] = out
    return dict(rounds=rounds, invalid=invalid, medians=medians, comparisons=comparisons)


def correctness(data, directory):
    records = {}
    for s in data['sessions']:
        path = directory / f"s{s['index']:02d}-{s['size']}.token-ids.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        records[s['size']] = [r for r in rows if len(r['prompt_token_ids']) > 20000]
    out = {}
    for size in (4096, 8192):
        comparisons = []
        for index, (ra, rb) in enumerate(zip(records[2048], records[size])):
            same_prompt = ra['prompt_token_ids'] == rb['prompt_token_ids']
            same = (same_prompt and ra['published_token_ids'] == rb['published_token_ids']
                    and ra['cancelled'] == rb['cancelled'] and ra['failure'] == rb['failure'])
            first = next((i for i, (x, y) in enumerate(zip(ra['published_token_ids'], rb['published_token_ids']))
                          if x != y), None)
            comparisons.append(dict(request=index, same_prompt=same_prompt, equal=same,
                                    tokens=[len(ra['published_token_ids']), len(rb['published_token_ids'])],
                                    first_divergence=first))
        out[size] = dict(records=[len(records[2048]), len(records[size])], comparisons=comparisons,
                         all_equal=len(records[2048]) == len(records[size]) == len(TASKS)
                         and all(c['equal'] for c in comparisons))
    finishes = {s['size']: [(r['task'], r['finish_reason'], r['completion_tokens'], r['output_sha256'])
                            for r in s['rows']] for s in data['sessions']}
    return dict(comparisons=out, finishes=finishes)


def path(data, directory):
    out = {}
    for s in data['sessions']:
        text = re.sub(r'\x1b\[[0-9;]*m', '', (directory / f"s{s['index']:02d}-{s['size']}.server.log").read_text())
        chunks = []
        for line in text.splitlines():
            if 'dflash2_prefill_phases' in line:
                fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
                parts = [p.split(':') for p in fields['phases'].strip('"').split(',')]
                chunks.append(dict(prompt_len=int(fields['prompt_len']),
                                   phases=[dict(phase=p[0], size=int(p[1]), ms=int(p[2]) / 1000) for p in parts]))
        out[s['size']] = dict(witnesses={w: w in text for w in WITNESSES}, prefill=[c for c in chunks if c['prompt_len'] > 20000],
                              rows=[dict(task=r['task'], ttft_s=r['ttft_s'], server_prefill_ms=r['server_prefill_ms'])
                                    for r in s['rows']])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    raw = (directory / 'run.json').read_bytes()
    data = json.loads(raw)
    if not data.get('complete'):
        raise SystemExit('incomplete run: no judgement')
    data['sessions_by'] = {(s['round'], s['size']): s for s in data['sessions']}
    result = dict(label=a.label, kind=data['kind'], run_sha256=hashlib.sha256(raw).hexdigest(), protocol_sha256=data['protocol_sha256'],
                  binary=data['binary'])
    if data['kind'] == 'perf':
        result.update(perf(data))
    elif data['kind'] == 'correctness':
        result.update(correctness(data, directory))
    else:
        result.update(path=path(data, directory))
    a.output.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(result.get('comparisons') or result.get('path'), indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
