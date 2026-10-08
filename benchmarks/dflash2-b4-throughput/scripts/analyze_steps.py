#!/usr/bin/env python3
"""Summarize diagnose_b4.py runs: batch throughput and, when a step log is
present, the per-batch step decomposition. Step-log lines are mapped to
batches by the scheduler step counter (batch_count) read from /healthz before
and after each batch. In `stages` mode the ragged and tree window stage times
are synchronized and attributable; in `steplog` mode phase times show where
the host waited."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load_steps(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # an unfinished final line at shutdown
    return rows


def summarize_steps(steps):
    phases = defaultdict(int)
    widths = Counter()
    notes = defaultdict(int)
    ragged = dict(windows=0, built=0, rows=Counter(), cache_build_us=0, window_us=0,
                  stages=defaultdict(int), emitted=0, accepted=0)
    tree = dict(windows=Counter(), stages=defaultdict(int), emitted=0)
    total = 0
    for step in steps:
        total += step['total_us']
        widths[step['width']] += 1
        for name, us in step['phases']:
            phases[name] += us
        for name, value in step['notes']:
            notes[name] += value
        if step.get('ragged'):
            t = step['ragged']['timing']
            ragged['windows'] += 1
            ragged['built'] += int(step['ragged']['built'])
            ragged['rows'][t['rows']] += 1
            ragged['cache_build_us'] += t['cache_build_us']
            ragged['window_us'] += t['window_us']
            ragged['emitted'] += sum(t['emitted'])
            ragged['accepted'] += sum(t['accepted'])
            for name, us in t['stages']:
                ragged['stages'][name] += us
        for w in step.get('tree_windows') or []:
            tree['windows'][w['kind']] += 1
            tree['emitted'] += w['emitted']
            for name, us in w['stages']:
                tree['stages'][f"{w['kind']}:{name}"] += us
    ragged['rows'] = dict(ragged['rows'])
    ragged['stages'] = dict(ragged['stages'])
    tree['windows'] = dict(tree['windows'])
    tree['stages'] = dict(tree['stages'])
    return dict(steps=len(steps), total_us=total, phases=dict(phases), widths=dict(widths), notes=dict(notes),
                ragged=ragged, tree=tree)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('labels', nargs='+')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    report = {}
    for label in a.labels:
        directory = C.REPORTS / 'results' / label
        data = json.loads((directory / 'run.json').read_text())
        out = dict(mode=data['mode'], binary_sha256=data['binary_sha256'], batches=[])
        for s in data['sessions']:
            steps = load_steps(directory / f"s{s['index']}.steps.jsonl")
            for b in s['batches']:
                lo, hi = b['before']['scheduler']['batch_count'], b['after']['scheduler']['batch_count']
                entry = dict(session=s['index'], kind=b['kind'], tasks=b['tasks'],
                             throughput_tps=b['metrics']['throughput_tps'], wall_s=b['metrics']['wall_s'],
                             tokens=b['metrics']['output_tokens'], valid=all(r['valid'] for r in b['rows']),
                             step_range=[lo, hi],
                             ttft_ms=[round(r['ttft_s'] * 1000, 1) for r in b['rows']],
                             decode_tps=[round(r['decode_tps'] or 0, 1) for r in b['rows']],
                             e2e_s=[round(r['e2e_s'], 2) for r in b['rows']],
                             ragged_counters=[b['before']['ragged_linear'], b['after']['ragged_linear']],
                             peak_gib=(b['memory'].get('observed_peak_footprint_bytes') or 0) / 2**30)
                if steps:
                    entry['steps'] = summarize_steps(steps[lo:hi])
                out['batches'].append(entry)
        scored = [b for b in out['batches'] if b['kind'] == 'scored']
        out['scored_throughput_median'] = statistics.median(b['throughput_tps'] for b in scored)
        out['scored_wall_median'] = statistics.median(b['wall_s'] for b in scored)
        report[label] = out
        print(f"== {label} ({data['mode']}) 吞吐中位数 {out['scored_throughput_median']:.1f} tok/s，"
              f"整批完成中位数 {out['scored_wall_median']:.2f} s")
        for b in scored:
            line = f"  会话{b['session']} {','.join(b['tasks'])}: {b['throughput_tps']:.1f} tok/s {b['wall_s']:.2f} s 有效={b['valid']}"
            if 'steps' in b:
                st = b['steps']
                total = st['total_us'] or 1
                share = {k: round(100 * v / total, 1) for k, v in sorted(st['phases'].items(), key=lambda x: -x[1])}
                line += f"\n    步数 {st['steps']} 步宽 {st['widths']} 阶段占比% {share}"
                line += f"\n    ragged 窗口 {st['ragged']['windows']}（新建组 {st['ragged']['built']}，行数 {st['ragged']['rows']}）阶段 {st['ragged']['stages']}"
                line += f"\n    树窗口 {st['tree']['windows']} 阶段 {st['tree']['stages']} 备注 {st['notes']}"
            print(line)
    if a.output:
        with a.output.open('x') as f:
            json.dump(report, f, indent=1)
            f.write('\n')


if __name__ == '__main__':
    main()
