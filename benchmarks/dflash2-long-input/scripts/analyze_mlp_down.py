#!/usr/bin/env python3
"""Analysis of run_mlp_down.py (docs/protocol-mlp-down.md). All ratios are
candidate / baseline (B / A). Per task (30K tasks, B1 short tasks) and per
B4 batch set: median over rounds per arm, ratio of medians; 95% interval by
paired-round bootstrap (rounds resampled with replacement, 10000 draws,
seed 20261008). Geometric means over tasks are reported but never replace
a per-task gate. Memory: median session lifecycle peak per arm, per part.
Time order and absolute values are listed for drift inspection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import common as C

DRAWS, SEED = 10000, 20261008
LONG_TASKS = ('code-1', 'knowledge-1')
B1_SHORT = ('code-1', 'knowledge-1', 'code-3')


def ratio(cells, keys, value, rounds):
    def point(idx):
        logs = []
        for key in keys:
            b = statistics.median(value(cells[i]['B'][key]) for i in idx)
            a = statistics.median(value(cells[i]['A'][key]) for i in idx)
            logs.append(math.log(b / a))
        return math.exp(sum(logs) / len(logs))
    rng = random.Random(SEED)
    boot = sorted(point(rng.choices(rounds, k=len(rounds))) for _ in range(DRAWS))
    return dict(ratio=point(rounds), ci95=[boot[int(DRAWS * 0.025)], boot[int(DRAWS * 0.975) - 1]],
                A_median=[statistics.median(value(cells[i]['A'][k]) for i in rounds) for k in keys],
                B_median=[statistics.median(value(cells[i]['B'][k]) for i in rounds) for k in keys])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--correctness', required=True, help='label of the token-id correctness run')
    a = p.parse_args()
    raw = (C.REPORTS / 'results' / a.label / 'run.json').read_bytes()
    data = json.loads(raw)
    longs, shorts, order = {}, {}, []
    for s in data['sessions']:
        valid = s.get('lifecycle_peak_bytes') is not None
        order.append(dict(index=s['index'], kind=s['kind'], round=s['round'], arm=s['arm'], valid=valid,
                          values=[(r['task'], round(r['ttft_s'] or 0, 3), round(r['e2e_s'] or 0, 3),
                                   round(r['decode_tps'] or 0, 2)) for r in s['b1']]))
        if not valid:
            continue
        if s['kind'] == 'long':
            cell = {r['task']: r for r in s['b1']}
            cell['lifecycle'] = s['lifecycle_peak_bytes']
            longs.setdefault(s['round'], {})[s['arm']] = cell
        else:
            cell = {}
            for b in s['batches']:
                if b['kind'] == 'scored':
                    cell['set:' + ','.join(b['tasks'])] = b
            for row in s['b1']:
                cell['b1:' + row['task']] = row
            cell['lifecycle'] = s['lifecycle_peak_bytes']
            shorts.setdefault(s['round'], {})[s['arm']] = cell
    long_rounds = sorted(r for r, c in longs.items() if set(c) == {'A', 'B'})
    short_rounds = sorted(r for r, c in shorts.items() if set(c) == {'A', 'B'})
    planned_long = len({(x['round']) for x in data['plan'] if x['kind'] == 'long'})
    planned_short = len({(x['round']) for x in data['plan'] if x['kind'] == 'short'})
    complete = (data.get('complete') and not data.get('failure') and len(long_rounds) == planned_long
                and len(short_rounds) == planned_short)
    result = dict(label=a.label, input_sha256=hashlib.sha256(raw).hexdigest(),
                  analyzer_sha256=C.digest(Path(__file__)), binaries=data['binaries'], failure=data.get('failure'), complete=bool(complete), long_rounds=long_rounds,
                  short_rounds=short_rounds, time_order=order)
    r = {}
    if long_rounds:
        for task in LONG_TASKS:
            for metric, f in (('ttft', lambda x: x['ttft_s']), ('e2e', lambda x: x['e2e_s']),
                              ('decode_tps', lambda x: x['decode_tps'])):
                r[f'long:{task}:{metric}'] = ratio(longs, [task], f, long_rounds)
        r['long:memory'] = ratio(longs, ['lifecycle'], lambda v: v, long_rounds)
    if short_rounds:
        sets = sorted(k for k in shorts[short_rounds[0]]['A'] if k.startswith('set:'))
        for key in sets:
            r[f'b4:{key}:throughput'] = ratio(shorts, [key], lambda b: b['metrics']['throughput_tps'], short_rounds)
        r['b4:geomean:throughput'] = ratio(shorts, sets, lambda b: b['metrics']['throughput_tps'], short_rounds)
        for task in B1_SHORT:
            for metric, f in (('ttft', lambda x: x['ttft_s']), ('e2e', lambda x: x['e2e_s']),
                              ('decode_tps', lambda x: x['decode_tps'])):
                r[f'b1:{task}:{metric}'] = ratio(shorts, [f'b1:{task}'], f, short_rounds)
        r['b4:memory'] = ratio(shorts, ['lifecycle'], lambda v: v, short_rounds)
    result['ratios'] = r
    gates = {}
    if long_rounds:
        for task in LONG_TASKS:
            t = r[f'long:{task}:ttft']
            gates[f'long:{task}:ttft'] = t['ratio'] <= 0.985 and t['ci95'][1] < 1.0
            gates[f'long:{task}:decode'] = r[f'long:{task}:decode_tps']['ratio'] >= 0.99
            gates[f'long:{task}:e2e'] = r[f'long:{task}:e2e']['ratio'] <= 1.00
        gates['long:memory'] = r['long:memory']['ratio'] <= 1.01
    if short_rounds:
        for key in sets:
            gates[f'b4:{key}:throughput'] = r[f'b4:{key}:throughput']['ratio'] >= 0.99
        for task in B1_SHORT:
            gates[f'b1:{task}:ttft'] = r[f'b1:{task}:ttft']['ratio'] <= 1.03
            gates[f'b1:{task}:e2e'] = r[f'b1:{task}:e2e']['ratio'] <= 1.03
            gates[f'b1:{task}:decode'] = r[f'b1:{task}:decode_tps']['ci95'][0] >= 0.97
        gates['b4:memory'] = r['b4:memory']['ratio'] <= 1.01
    correctness = json.loads((C.REPORTS / 'results' / a.correctness / 'run.json').read_text())
    comps = [c for t in correctness.get('token_comparisons') or [] for c in t['comparisons']]
    result['correctness'] = dict(label=a.correctness, complete=correctness.get('complete'),
                                 failure=correctness.get('failure'), requests=len(comps),
                                 equal=sum(c['status'] == 'equal' for c in comps),
                                 binaries=correctness['binaries'])
    gates['token_ids'] = (bool(correctness.get('complete')) and not correctness.get('failure') and bool(comps)
                          and all(c['status'] == 'equal' for c in comps)
                          and correctness['binaries'] == data['binaries'])
    hashes = [c for t in data.get('hash_comparisons') or [] for c in t['comparisons']]
    result['output_hashes'] = dict(requests=len(hashes), equal=sum(c['status'] == 'equal' for c in hashes))
    gates['output_hashes'] = bool(hashes) and all(c['status'] == 'equal' for c in hashes) and bool(complete)
    result['gates'] = gates
    result['all_gates'] = bool(complete) and bool(gates) and all(gates.values())
    with a.output.open('x') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(dict(complete=result['complete'], failure=result['failure'],
                          ratios={k: (round(v['ratio'], 4), [round(x, 4) for x in v['ci95']]) for k, v in r.items()},
                          gates=gates, all_gates=result['all_gates']), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
