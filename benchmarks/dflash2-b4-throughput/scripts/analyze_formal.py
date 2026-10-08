#!/usr/bin/env python3
"""Analysis of run_formal.py (docs/protocol.md). Per batch set (or task) the
median over rounds per arm, B/A ratio, geometric mean over sets; 95%
interval by paired-round bootstrap (10000 draws, seed 20261006). Gates as
registered."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DRAWS, SEED = 10000, 20261006


def geo_ratio(cells, keys, value, rounds):
    def point(idx):
        logs = []
        for key in keys:
            b = statistics.median(value(cells[i]['B'][key]) for i in idx)
            a = statistics.median(value(cells[i]['A'][key]) for i in idx)
            logs.append(math.log(b / a))
        return math.exp(sum(logs) / len(logs))
    rng = random.Random(SEED)
    boot = sorted(point(rng.choices(rounds, k=len(rounds))) for _ in range(DRAWS))
    return dict(ratio=point(rounds), ci95=[boot[int(DRAWS * 0.025)], boot[int(DRAWS * 0.975) - 1]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = (C.REPORTS / 'results' / a.label / 'run.json').read_bytes()
    data = json.loads(raw)
    if not data.get('complete') or data.get('failure'):
        raise SystemExit('incomplete run: no judgement')
    b4, longs = {}, {}
    for s in data['sessions']:
        if s['kind'] == 'b4':
            cell = {}
            for b in s['batches']:
                if b['kind'] == 'scored':
                    key = 'set:' + ','.join(b['tasks'])
                    cell[key] = b
                    for slot, row in enumerate(b['rows']):
                        cell[f'req:{key}:{slot}'] = row
            for row in s['b1']:
                cell['b1:' + row['task']] = row
            cell['lifecycle'] = s['lifecycle_peak_bytes']
            b4.setdefault(s['round'], {})[s['arm']] = cell
        else:
            longs.setdefault(s['round'], {})[s['arm']] = {'long': s['b1'][0], 'lifecycle': s['lifecycle_peak_bytes']}
    rounds = sorted(b4)
    sets = sorted(k for k in b4[rounds[0]]['A'] if k.startswith('set:'))
    reqs = sorted(k for k in b4[rounds[0]]['A'] if k.startswith('req:'))
    b1 = sorted(k for k in b4[rounds[0]]['A'] if k.startswith('b1:'))
    r = dict(
        b4_throughput=geo_ratio(b4, sets, lambda b: b['metrics']['throughput_tps'], rounds),
        b4_wall=geo_ratio(b4, sets, lambda b: b['metrics']['wall_s'], rounds),
        b4_request_ttft=geo_ratio(b4, reqs, lambda row: row['ttft_s'], rounds),
        b4_request_e2e=geo_ratio(b4, reqs, lambda row: row['e2e_s'], rounds),
        b1_decode=geo_ratio(b4, b1, lambda row: row['decode_tps'], rounds),
        b1_ttft=geo_ratio(b4, b1, lambda row: row['ttft_s'], rounds),
        b1_e2e=geo_ratio(b4, b1, lambda row: row['e2e_s'], rounds),
        long_ttft=geo_ratio(longs, ['long'], lambda row: row['ttft_s'], sorted(longs)),
        long_decode=geo_ratio(longs, ['long'], lambda row: row['decode_tps'], sorted(longs)),
    )
    life = {arm: statistics.median(b4[i][arm]['lifecycle'] for i in rounds) for arm in 'AB'}
    r['memory'] = dict(ratio=life['B'] / life['A'], A_gib=life['A'] / 2**30, B_gib=life['B'] / 2**30)
    gates = {
        'b4_throughput': r['b4_throughput']['ratio'] >= 1.03 and r['b4_throughput']['ci95'][0] > 1.0,
        'b4_wall': r['b4_wall']['ratio'] <= 0.97 and r['b4_wall']['ci95'][1] < 1.0,
        'b4_request_ttft': r['b4_request_ttft']['ratio'] <= 1.05,
        'b1_decode': r['b1_decode']['ci95'][0] >= 0.97,
        'b1_ttft': r['b1_ttft']['ratio'] <= 1.03,
        'b1_e2e': r['b1_e2e']['ratio'] <= 1.03,
        'long_ttft': r['long_ttft']['ratio'] <= 1.03,
        'memory': r['memory']['ratio'] <= 1.02,
    }
    medians = {arm: {k: statistics.median(b4[i][arm][k]['metrics']['throughput_tps'] for i in rounds) for k in sets}
               for arm in 'AB'}
    walls = {arm: {k: statistics.median(b4[i][arm][k]['metrics']['wall_s'] for i in rounds) for k in sets}
             for arm in 'AB'}
    result = dict(label=a.label, input_sha256=hashlib.sha256(raw).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  binaries=data['binaries'], rounds=len(rounds), long_rounds=len(longs), ratios=r, gates=gates,
                  all_gates=all(gates.values()), throughput_medians=medians, wall_medians=walls,
                  per_round=[{arm: {k: round(b4[i][arm][k]['metrics']['throughput_tps'], 2) for k in sets}
                              for arm in 'AB'} for i in rounds])
    with a.output.open('x') as f:
        json.dump(result, f, indent=1)
        f.write('\n')
    print(json.dumps(dict(ratios={k: (round(v['ratio'], 4), [round(x, 4) for x in v.get('ci95', [])])
                                  for k, v in r.items()}, gates=gates, all_gates=all(gates.values()),
                          throughput_medians=medians), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
