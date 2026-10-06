#!/usr/bin/env python3
"""30K hit TTFT bimodality diagnosis (docs/protocol-addendum-30k-hit.md §2).
Eight sessions Bd Cd Cd Bd Bd Cd Cd Bd with the diagnostic binaries. Each
session repeats run_timing.py's steps exactly, then three more 30K exact
repeats (1 s apart) and one after a 10 s pause. Server logs carry one
admission-pressure line and one prefill-phase line per DFlash2 request; both
are matched to requests by order. No pass/fail judgement."""
import argparse
import json
from pathlib import Path
import re
import time

import common as C
import run_timing as T

ORDER = ('Bd', 'Cd', 'Cd', 'Bd', 'Bd', 'Cd', 'Cd', 'Bd')
ENV = {'IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES': '1'}


def parse(log_path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
    admissions, prefill = [], []
    for line in text.splitlines():
        fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
        if 'dflash2_admission_pressure' in line:
            admissions.append({k: fields.get(k) for k in ('level', 'current_bytes', 'soft_watermark_bytes',
                                                          'ceiling_bytes')})
        elif 'dflash2_prefill_phases' in line:
            phases = []
            for part in fields['phases'].strip('"').split(','):
                name, size, us, active, cache, footprint = part.split(':')
                phases.append(dict(phase=name, size=int(size), us=int(us), active_mib=int(active),
                                   cache_mib=int(cache), footprint_mib=int(footprint)))
            prefill.append(dict(prompt_len=int(fields['prompt_len']), hit_tokens=int(fields['hit_tokens']),
                                execution=fields.get('execution'), phases=phases))
    return admissions, prefill


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--baseline-diag', type=Path, required=True)
    p.add_argument('--candidate-diag', type=Path, required=True)
    a = p.parse_args()
    directory = C.SUITE / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    binaries = {'Bd': a.baseline_diag, 'Cd': a.candidate_diag}
    data = dict(label=a.label, purpose='30k-hit-diagnosis', order=ORDER,
                protocol_sha256=C.digest(C.SUITE / 'docs/protocol-addendum-30k-hit.md'),
                binaries={k: dict(path=str(v), sha256=C.digest(v)) for k, v in binaries.items()},
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sessions=[], complete=False)
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    steps = T.session_steps(fx)
    item = fx['30k']['code-1']
    steps += [('repeat-extra', '30k', 'code-1', C.user(item['prompt']), 1024)] * 3
    steps += [('pause', None, None, None, None), ('repeat-late', '30k', 'code-1', C.user(item['prompt']), 1024)]
    for index, arm in enumerate(ORDER):
        record = dict(index=index, arm=arm, rows=[])
        data['sessions'].append(record)
        server = C.Server(directory, f's{index:02d}-{arm}', binaries[arm], True, ENV)
        try:
            with server:
                answers = {}
                for kind, phase, task, messages, max_tokens in steps:
                    if kind == 'pause':
                        time.sleep(10)
                        continue
                    if kind == 'append':
                        messages = (C.user(fx['short'][task]['prompt']) +
                                    [dict(role='assistant', content=answers[task]), dict(role='user', content=T.Q2)])
                    row = dict(kind=kind, phase=phase, task=task) | server.request(messages, max_tokens)
                    row.pop('output', None) if kind != 'miss' else None
                    if kind == 'miss' and phase == 'short':
                        answers[task] = row['output']
                    record['rows'].append(row)
                    time.sleep(1)
        finally:
            record['server'] = getattr(server, 'summary', None)
            admissions, prefill = parse(server.log_path)
            record['admissions'], record['prefill_phases'] = admissions, prefill
            save()
        hits = [(r['kind'], round(r['server_prefill_ms'], 1)) for r in record['rows'] if r['phase'] == '30k']
        print(json.dumps(dict(session=index, arm=arm, thirty_k=hits), ensure_ascii=False), flush=True)
        time.sleep(10)
    data['complete'] = True
    save()


if __name__ == '__main__':
    main()
