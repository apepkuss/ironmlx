#!/usr/bin/env python3
"""Analysis of the L1 runs (docs/protocol-l1-qmm-feasibility.md).
wall: per projection, mode and variant the per-process steady medians and
first-call times (all processes kept), and numerics.
gpu: GPU intervals of the bench process from the Metal System Trace, split
into variant blocks by the 300 ms idle gaps and into calls by the 30 ms
gaps; per call the summed GPU busy time.
Combined with the L2 2048 TTFT, an optimistic per-prefill saving bound per
projection (calls per 30K prefill x GPU time difference)."""
import argparse
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import common as C

CALLS = {'mlp_gate_up': 64, 'mlp_down': 64, 'gdn_in_proj': 48, 'gdn_out_proj': 48, 'attn_qkv': 16, 'attn_o_proj': 16}
CHUNKS = 15  # full 2048-row chunks of a 30732-token prompt


def wall(data):
    out = {}
    for run in data['runs']:
        if run['exit_code'] != 0:
            out.setdefault('_failures', []).append(dict(name=run['name'], stderr=run['stderr'][-2000:]))
            continue
        for p in run['result']['projections']:
            cell = out.setdefault(p['name'], dict(k=p['k'], n=p['n'], weight_bytes=p['weight_bytes']))
            for variant in ('qmm', 'bf16'):
                v = p[variant]
                key = f"{run['mode']}:{variant}"
                entry = cell.setdefault(key, dict(steady_medians_ms=[], first_call_ms=[], numerics=[]))
                entry['steady_medians_ms'].append(v['steady']['median_ms'])
                entry['first_call_ms'].append(v['first_call_ms'])
                entry['numerics'].append(v['numerics'])
                if variant == 'bf16':
                    entry.setdefault('prepare_ms', []).append(v['prepare_ms'])
                    entry['prepared_bytes'] = v['prepared_bytes']
    for name, cell in out.items():
        if name.startswith('_'):
            continue
        for key, entry in cell.items():
            if isinstance(entry, dict):
                entry['median_of_medians_ms'] = statistics.median(entry['steady_medians_ms'])
                entry['range_ms'] = [min(entry['steady_medians_ms']), max(entry['steady_medians_ms'])]
                entry['hashes'] = sorted({n['hash'] for n in entry['numerics']})
                entry['max_abs_err'] = max(n['max_abs_err'] for n in entry['numerics'])
                entry['mean_abs_err'] = max(n['mean_abs_err'] for n in entry['numerics'])
                entry['nonfinite'] = max(n['nonfinite'] for n in entry['numerics'])
    return out


def intervals(path):
    """(start_ns, duration_ns) of the bench process's top-level GPU intervals (xctrace id/ref rows)."""
    root = ET.parse(path).getroot()
    ids = {}
    out = []
    for row in root.iter('row'):
        tags = {}
        for el in row:
            if 'ref' in el.attrib:
                el = ids[el.attrib['ref']]
            else:
                for sub in el.iter():
                    if 'id' in sub.attrib:
                        ids[sub.attrib['id']] = sub
            tags[el.tag] = el
        process = tags.get('process')
        if process is None or 'prefill_qmm_feasibility' not in process.attrib.get('fmt', ''):
            continue
        if int(tags['metal-nesting-level'].text) != 0:
            continue
        out.append((int(tags['start-time'].text), int(tags['duration'].text)))
    return sorted(out)


def gpu(data, directory):
    """Calls are the bench's command buffers; variant blocks are split by the 300 ms idle gaps."""
    out = {}
    for run in data['runs']:
        mode = run['mode']
        names = [p['name'] for p in run['result']['projections']]
        rows = intervals(directory / f"gpu-{mode}.gpu-intervals.xml")
        blocks, current = [], []
        for i, (start, duration) in enumerate(rows):
            if i and start - (rows[i - 1][0] + rows[i - 1][1]) > 200e6:
                blocks.append(current)
                current = []
            current.append(duration / 1e6)
        blocks.append(current)
        out[mode] = dict(command_buffers=len(rows), block_sizes=[len(b) for b in blocks], blocks=blocks,
                         projections=names)
    return out


def chain(data, ttft_ms):
    """Sustained per-call time (protocol amendment 1) and optimistic TTFT bounds."""
    per = {}
    for run in data['runs']:
        if run['exit_code'] != 0:
            per.setdefault('_failures', []).append(run['name'])
            continue
        for p in run['result']['projections']:
            cell = per.setdefault(p['name'], dict(k=p['k'], n=p['n'], runs={}))
            for variant in ('qmm', 'bf16'):
                cell['runs'].setdefault(f"{run['mode']}:{variant}", {})[run['index']] = p[variant]['median_ms']
    pairs = ((0, 1), (3, 2), (4, 5))  # (production index, default index)
    out = {}
    for name, cell in per.items():
        if name.startswith('_'):
            out[name] = cell
            continue
        flops = 2 * 2048 * cell['k'] * cell['n']
        entry = dict(k=cell['k'], n=cell['n'], calls_per_prefill=CALLS[name] * CHUNKS)
        for key, runs in cell['runs'].items():
            values = list(runs.values())
            med = statistics.median(values)
            entry[key] = dict(per_process_ms=runs, median_ms=med, range_ms=[min(values), max(values)],
                              tflops=flops / med / 1e9)
        for variant in ('qmm', 'bf16'):
            prod, dflt = cell['runs'][f'production:{variant}'], cell['runs'][f'default:{variant}']
            entry[f'{variant}_production_over_default'] = [prod[a] / dflt[b] for a, b in pairs]
        best = min(entry['production:qmm']['median_ms'], entry['default:qmm']['median_ms'])
        entry['qmm_choice_saving_ms_per_prefill'] = (entry['production:qmm']['median_ms'] - best) * entry['calls_per_prefill']
        entry['bf16_reference_gap_ms_per_prefill'] = (entry['production:qmm']['median_ms']
                                                      - min(entry['production:bf16']['median_ms'],
                                                            entry['default:bf16']['median_ms'])) * entry['calls_per_prefill']
        entry['production_qmm_ms_per_prefill'] = entry['production:qmm']['median_ms'] * entry['calls_per_prefill']
        out[name] = entry
    shapes = [v for k, v in out.items() if not k.startswith('_')]
    out['_totals'] = dict(
        ttft_reference_ms=ttft_ms,
        production_qmm_ms_per_prefill=sum(v['production_qmm_ms_per_prefill'] for v in shapes),
        qmm_choice_saving_ms_per_prefill=sum(v['qmm_choice_saving_ms_per_prefill'] for v in shapes),
        bf16_reference_gap_ms_per_prefill=sum(max(0.0, v['bf16_reference_gap_ms_per_prefill']) for v in shapes))
    t = out['_totals']
    for key in ('production_qmm_ms_per_prefill', 'qmm_choice_saving_ms_per_prefill', 'bf16_reference_gap_ms_per_prefill'):
        t[key.replace('_ms_per_prefill', '_ttft_fraction')] = t[key] / ttft_ms
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    data = json.loads((directory / 'run.json').read_text())
    if not data.get('complete'):
        raise SystemExit('incomplete run: no judgement')
    result = dict(label=a.label, kind=data['kind'], binary=data['binary'], protocol_sha256=data['protocol_sha256'])
    if data['kind'] == 'wall':
        result['wall'] = wall(data)
    elif data['kind'] == 'chain':
        l2 = json.loads((C.SUITE / 'evidence/l2-perf-v1-analysis.json').read_text())
        ttft = statistics.mean(l2['medians']['2048'][t]['ttft'] for t in ('code-1', 'knowledge-1')) * 1000
        result['chain'] = chain(data, ttft)
    elif data['kind'] == 'gpu':
        result['gpu'] = gpu(data, directory)
    a.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
