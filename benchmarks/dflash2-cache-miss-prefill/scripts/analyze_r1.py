#!/usr/bin/env python3
"""Analysis of the R1 confirmation run (docs/protocol-r1-confirmation.md).
Scored rounds with both sessions valid; C/B = ratio of the arms' R1 TTFT
medians (criterion: <= 1.10); paired-round bootstrap 95% interval (10000
draws, seed 20261006). Also per-round raw values, C - B differences and the
slow-peak share (server prefill > 60 ms). Correctness: R1 full hit, R1 output
equal to the same session's 30K miss output, equal across sessions and arms."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

SUITE = Path(__file__).resolve().parents[1]
DRAWS, SEED, SLOW_MS, MIN_ROUNDS = 10000, 20261006, 60.0, 20


def r1(session):
    rows = {(r['kind'], r['phase'], r['task']): r for r in session['rows']}
    return rows.get(('repeat', '30k', 'code-1')), rows.get(('miss', '30k', 'code-1'))


def interval(values_fn, rounds):
    rng = random.Random(SEED)
    boot = sorted(values_fn(rng.choices(rounds, k=len(rounds))) for _ in range(DRAWS))
    return [boot[int(DRAWS * 0.025)], boot[int(DRAWS * 0.975) - 1]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = (SUITE / 'results' / a.label / 'run.json').read_bytes()
    data = json.loads(raw)
    scored = {}
    invalid = []
    for s in data['sessions']:
        if not s['scored']:
            continue
        if s['invalid']:
            invalid.append(dict(index=s['index'], round=s['round'], arm=s['arm'], reason=s['invalid']))
        scored.setdefault(s['round'], {})[s['arm']] = s
    planned_rounds = sorted({x['round'] for x in data['plan'] if x['scored']})
    valid_rounds = [r for r in planned_rounds
                    if r in scored and all(arm in scored[r] and not scored[r][arm]['invalid'] for arm in 'BC')]
    cells = {r: {arm: r1(scored[r][arm]) for arm in 'BC'} for r in valid_rounds}
    ttft = {r: {arm: cells[r][arm][0]['ttft_s'] * 1000 for arm in 'BC'} for r in valid_rounds}
    prefill = {r: {arm: cells[r][arm][0]['server_prefill_ms'] for arm in 'BC'} for r in valid_rounds}

    def ratio(idx):
        return statistics.median(ttft[i]['C'] for i in idx) / statistics.median(ttft[i]['B'] for i in idx)

    def diff(idx):
        return statistics.median(ttft[i]['C'] - ttft[i]['B'] for i in idx)
    result = dict(label=a.label, input_sha256=hashlib.sha256(raw).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  complete=data.get('complete'), stopped=data.get('stopped'),
                  planned_rounds=len(planned_rounds), valid_rounds=len(valid_rounds), invalid_sessions=invalid)
    if data.get('stopped') or len(valid_rounds) < MIN_ROUNDS:
        result['criterion'] = 'undetermined'
    else:
        point = ratio(valid_rounds)
        result.update(
            ratio=point, ratio_ci95=interval(ratio, valid_rounds),
            criterion='met' if point <= 1.10 else 'not met',
            median_ttft_ms={arm: statistics.median(ttft[r][arm] for r in valid_rounds) for arm in 'BC'},
            median_difference_ms=diff(valid_rounds), median_difference_ci95_ms=interval(diff, valid_rounds),
            slow_share={arm: [sum(prefill[r][arm] > SLOW_MS for r in valid_rounds), len(valid_rounds)]
                        for arm in 'BC'})
    correctness = dict(full_hit=all(cells[r][arm][0]['counters']['prefix_cache_hit_tokens']
                                    == cells[r][arm][0]['prompt_tokens'] for r in valid_rounds for arm in 'BC'),
                       r1_equals_session_miss=all(cells[r][arm][0]['output_sha256'] == cells[r][arm][1]['output_sha256']
                                                  for r in valid_rounds for arm in 'BC'),
                       r1_outputs={arm: sorted({cells[r][arm][0]['output_sha256'] for r in valid_rounds})
                                   for arm in 'BC'})
    correctness['r1_equal_across_sessions_and_arms'] = (
        len(set(correctness['r1_outputs']['B']) | set(correctness['r1_outputs']['C'])) == 1)
    result['correctness'] = correctness
    result['per_round'] = [dict(round=r, order=''.join(sorted(scored[r], key=lambda arm: scored[r][arm]['index'])),
                                B_ttft_ms=ttft[r]['B'], C_ttft_ms=ttft[r]['C'],
                                B_prefill_ms=prefill[r]['B'], C_prefill_ms=prefill[r]['C'],
                                C_minus_B_ms=ttft[r]['C'] - ttft[r]['B']) for r in valid_rounds]
    with a.output.open('x') as f:
        json.dump(result, f, indent=1)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_round'}, indent=1, ensure_ascii=False))
    for row in result['per_round']:
        print(row['round'], row['order'], 'B %.1f C %.1f | prefill B %.1f C %.1f | C-B %+.1f' % (
            row['B_ttft_ms'], row['C_ttft_ms'], row['B_prefill_ms'], row['C_prefill_ms'], row['C_minus_B_ms']))


if __name__ == '__main__':
    main()
