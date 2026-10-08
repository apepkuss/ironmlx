#!/usr/bin/env python3
"""Analysis of run_b1_confirmation.py (docs/protocol-b1-ttft-confirmation.md).
Primary: B1 short TTFT B/A over the new rounds (per-task ratio of medians,
geometric mean, paired-round bootstrap 95%, 10000 draws, seed 20261006),
criterion point <= 1.03. Also: per-task ratios, code-3 paired differences,
reference metrics, and the 32-round sensitivity analysis with
results/item1-formal-v1 (reference only)."""
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

DRAWS, SEED, MIN_ROUNDS = 10000, 20261006, 20
TASKS = ('code-1', 'knowledge-1', 'code-3')


def cells_from(data, offset=0):
    cells = {}
    for s in data['sessions']:
        if s['kind'] != 'b4' or s.get('invalid'):
            continue
        cell = {f"b1:{r['task']}": r for r in s['b1']}
        for b in s['batches']:
            if b['kind'] == 'scored':
                cell['set:' + ','.join(b['tasks'])] = b
        cells.setdefault(s['round'] + offset, {})[s['arm']] = cell
    return {r: c for r, c in cells.items() if set(c) == {'A', 'B'}}


def ratio(cells, keys, value, rounds):
    def point(idx):
        logs = [math.log(statistics.median(value(cells[i]['B'][k]) for i in idx) /
                         statistics.median(value(cells[i]['A'][k]) for i in idx)) for k in keys]
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
    cells = cells_from(data)
    rounds = sorted(cells)
    invalid = [dict(index=s['index'], round=s['round'], arm=s['arm'], reason=s['invalid'])
               for s in data['sessions'] if s.get('invalid')]
    ttft = lambda row: row['ttft_s']  # noqa: E731
    b1_keys = [f'b1:{t}' for t in TASKS]
    result = dict(label=a.label, input_sha256=hashlib.sha256(raw).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  complete=data.get('complete'), stopped=data.get('stopped'), planned_rounds=24,
                  valid_rounds=len(rounds), invalid_sessions=invalid)
    if data.get('stopped') or not data.get('complete') or len(rounds) < MIN_ROUNDS:
        result['criterion'] = 'undetermined'
    else:
        primary = ratio(cells, b1_keys, ttft, rounds)
        result['primary_b1_ttft'] = primary
        result['criterion'] = 'met' if primary['ratio'] <= 1.03 else 'not met'
    if len(rounds) >= 2:
        result['per_task_ttft'] = {t: ratio(cells, [f'b1:{t}'], ttft, rounds) for t in TASKS}
        result['per_task_median_ms'] = {t: {arm: statistics.median(cells[r][arm][f'b1:{t}']['ttft_s'] * 1000
                                                                   for r in rounds) for arm in 'AB'}
                                        for t in TASKS}
        diffs = [(cells[r]['B']['b1:code-3']['ttft_s'] - cells[r]['A']['b1:code-3']['ttft_s']) * 1000
                 for r in rounds]
        a3 = sorted(cells[r]['A']['b1:code-3']['ttft_s'] * 1000 for r in rounds)
        b3 = sorted(cells[r]['B']['b1:code-3']['ttft_s'] * 1000 for r in rounds)
        result['code3'] = dict(
            paired_diff_ms=[round(x, 1) for x in diffs], median_paired_diff_ms=statistics.median(diffs),
            rounds_b_slower=sum(x > 0 for x in diffs), rounds=len(diffs),
            a_ms=[round(x, 1) for x in a3], b_ms=[round(x, 1) for x in b3],
            ratio_without_each_arm_max=statistics.median(b3[:-1]) / statistics.median(a3[:-1]))
        sets = sorted(k for k in cells[rounds[0]]['A'] if k.startswith('set:'))
        result['reference'] = dict(
            b1_decode=ratio(cells, b1_keys, lambda row: row['decode_tps'], rounds),
            b1_e2e=ratio(cells, b1_keys, lambda row: row['e2e_s'], rounds),
            b4_throughput=ratio(cells, sets, lambda b: b['metrics']['throughput_tps'], rounds),
            b4_wall=ratio(cells, sets, lambda b: b['metrics']['wall_s'], rounds))
        old = json.loads((C.REPORTS / 'results/item1-formal-v1/run.json').read_text())
        merged = dict(cells_from(old, offset=1000))
        merged.update(cells)
        merged_rounds = sorted(merged)
        result['sensitivity_32_rounds'] = dict(
            rounds=len(merged_rounds), b1_ttft=ratio(merged, b1_keys, ttft, merged_rounds),
            per_task={t: ratio(merged, [f'b1:{t}'], ttft, merged_rounds) for t in TASKS})
    with a.output.open('x') as f:
        json.dump(result, f, indent=1)
        f.write('\n')
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
