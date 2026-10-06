#!/usr/bin/env python3
"""Analysis of the formal timing run (docs/protocol.md section B). Per task
and arm the median over the six scored rounds; ratio C/B per task; overall
the geometric mean over tasks; 95% interval by paired-round bootstrap
(10000 draws, seed 20261006)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

SUITE = Path(__file__).resolve().parents[1]
WARM = ('code-1', 'code-3', 'knowledge-1', 'knowledge-2', 'knowledge-3')
DRAWS, SEED = 10000, 20261006


def values(data):
    """cells[arm][round][(kind, phase, task)] = row; lifecycle[arm][round]."""
    cells, lifecycle = {}, {}
    for s in data['sessions']:
        if not s['scored']:
            continue
        cells.setdefault(s['arm'], {})[s['round']] = {(r['kind'], r['phase'], r['task']): r for r in s['rows']}
        lifecycle.setdefault(s['arm'], {})[s['round']] = s['lifecycle_peak_footprint_bytes']
    return cells, lifecycle


def metric(row, name):
    if name == 'ttft':
        return row['ttft_s']
    if name == 'decode':
        return row['decode_tps']
    if name == 'e2e':
        return row['e2e_s']
    if name == 'peak':
        return row['memory']['observed_peak_footprint_bytes']
    raise KeyError(name)


def ratio(cells, keys, name, rounds):
    def point(idx):
        logs = []
        for key in keys:
            c = statistics.median(metric(cells['C'][i][key], name) for i in idx)
            b = statistics.median(metric(cells['B'][i][key], name) for i in idx)
            logs.append(math.log(c / b))
        return math.exp(sum(logs) / len(logs))
    rng = random.Random(SEED)
    boot = sorted(point(rng.choices(rounds, k=len(rounds))) for _ in range(DRAWS))
    return dict(ratio=point(rounds), ci95=[boot[int(DRAWS * 0.025)], boot[int(DRAWS * 0.975) - 1]])


def medians(cells, keys, name, rounds, scale=1.0):
    return {arm: {key[2]: statistics.median(metric(cells[arm][i][key], name) for i in rounds) * scale
                  for key in keys} for arm in ('B', 'C')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = (SUITE / 'results' / a.label / 'run.json').read_bytes()
    data = json.loads(raw)
    if not data.get('complete') or data.get('failure'):
        raise SystemExit('incomplete run: no judgement')
    cells, lifecycle = values(data)
    rounds = sorted(cells['B'])
    warm = [('miss', 'short', t) for t in WARM]
    repeats = [('repeat', 'short', t) for t in ('code-1', 'knowledge-1')]
    first = [('first', 'short', 'code-2')]
    miss30 = [('miss', '30k', 'code-1')]
    repeat30 = [('repeat', '30k', 'code-1')]
    append = [('append', 'short', 'code-1')]
    r = dict(
        warm_ttft=ratio(cells, warm, 'ttft', rounds),
        warm_decode=ratio(cells, warm, 'decode', rounds),
        warm_e2e=ratio(cells, warm, 'e2e', rounds),
        hit_ttft=ratio(cells, repeats, 'ttft', rounds),
        miss30_ttft=ratio(cells, miss30, 'ttft', rounds),
        repeat30_ttft=ratio(cells, repeat30, 'ttft', rounds),
        miss30_peak=ratio(cells, miss30, 'peak', rounds),
        first_ttft=ratio(cells, first, 'ttft', rounds),
        append_ttft=ratio(cells, append, 'ttft', rounds),
    )
    life = {arm: statistics.median(lifecycle[arm][i] for i in rounds) for arm in ('B', 'C')}
    r['lifecycle_peak'] = dict(ratio=life['C'] / life['B'], B_gib=life['B'] / 2**30, C_gib=life['C'] / 2**30)
    hit_tokens = {arm: {key[2] + '/' + key[1] + '/' + key[0]: sorted({cells[arm][i][key]['counters']['prefix_cache_hit_tokens']
                                                            for i in rounds})
                        for key in repeats + repeat30 + append} for arm in ('B', 'C')}
    outputs_stable = {arm: all(len({cells[arm][i][key]['output_sha256'] for i in rounds}) == 1
                               for key in warm + first + miss30) for arm in ('B', 'C')}
    gates = {
        'primary_warm_ttft': r['warm_ttft']['ratio'] <= 0.95 and r['warm_ttft']['ci95'][1] < 1.0,
        'warm_decode': r['warm_decode']['ci95'][0] >= 0.97,
        'warm_e2e': r['warm_e2e']['ci95'][1] <= 1.03,
        'hit_ttft': r['hit_ttft']['ratio'] <= 1.10 and hit_tokens['B'] == hit_tokens['C'],
        'miss30_ttft': r['miss30_ttft']['ci95'][1] <= 1.03,
        'repeat30_ttft': r['repeat30_ttft']['ratio'] <= 1.10,
        'memory': r['lifecycle_peak']['ratio'] <= 1.02 and r['miss30_peak']['ratio'] <= 1.02,
    }
    result = dict(label=a.label, input_sha256=hashlib.sha256(raw).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rounds=len(rounds), ratios=r, gates=gates, hit_tokens=hit_tokens, outputs_stable=outputs_stable,
                  medians=dict(warm_ttft_ms=medians(cells, warm, 'ttft', rounds, 1000),
                               warm_decode_tps=medians(cells, warm, 'decode', rounds),
                               warm_e2e_s=medians(cells, warm, 'e2e', rounds),
                               hit_ttft_ms=medians(cells, repeats, 'ttft', rounds, 1000),
                               first_ttft_ms=medians(cells, first, 'ttft', rounds, 1000),
                               append_ttft_ms=medians(cells, append, 'ttft', rounds, 1000),
                               miss30_ttft_s=medians(cells, miss30, 'ttft', rounds),
                               repeat30_ttft_ms=medians(cells, repeat30, 'ttft', rounds, 1000),
                               miss30_peak_gib=medians(cells, miss30, 'peak', rounds, 1 / 2**30)))
    with a.output.open('x') as f:
        json.dump(result, f, indent=1)
        f.write('\n')
    print(json.dumps(dict(ratios={k: (round(v['ratio'], 4), [round(x, 4) for x in v.get('ci95', [])])
                                  for k, v in r.items()}, gates=gates, hit_tokens=hit_tokens,
                          outputs_stable=outputs_stable), indent=1))


if __name__ == '__main__':
    main()
