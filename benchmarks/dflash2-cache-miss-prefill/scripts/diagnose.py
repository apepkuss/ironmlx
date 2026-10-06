#!/usr/bin/env python3
"""Diagnosis only (not a formal measurement): split DFlash2 prefill time into
lookup, per-chunk forward, prefix-cache save and first-logits phases with the
default-off IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES timing (clock reads at
existing synchronization points only).

Session `on`: prefix cache on (App default). Two warmups, the six short
prompts as misses, the same six again (exact repeats), then 30K code-1 miss,
its repeat and 30K knowledge-1 miss. Session `off`: no prefix cache, two
warmups, the six short prompts and 30K code-1. max_tokens 64 throughout (only
TTFT and prefill phases are of interest)."""
import argparse
import json
from pathlib import Path
import re
import time

import common as C

SHORT = ('code-1', 'code-2', 'code-3', 'knowledge-1', 'knowledge-2', 'knowledge-3')


def plan(session):
    rows = [('warmup', 'warmup', text) for text in C.WARMUPS]
    rows += [('short', f'miss:{t}', t) for t in SHORT]
    if session == 'on':
        rows += [('short', f'repeat:{t}', t) for t in SHORT]
        rows += [('30k', 'miss:code-1', 'code-1'), ('30k', 'repeat:code-1', 'code-1'),
                 ('30k', 'miss:knowledge-1', 'knowledge-1')]
    else:
        rows += [('30k', 'miss:code-1', 'code-1')]
    return rows


def phases(log_path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
    out = []
    for line in text.splitlines():
        if 'dflash2_prefill_phases' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
        parsed = []
        for part in fields['phases'].strip('"').split(','):
            name, size, us = part.split(':')
            parsed.append(dict(phase=name, size=int(size), us=int(us)))
        out.append(dict(execution=fields.get('execution'), prompt_len=int(fields['prompt_len']),
                        hit_tokens=int(fields['hit_tokens']), phases=parsed))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--binary', type=Path, required=True)
    a = p.parse_args()
    directory = C.SUITE / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    data = dict(label=a.label, purpose='diagnosis', binary=str(a.binary), binary_sha256=C.digest(a.binary),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sessions=[])
    out = directory / 'run.json'
    for session in ('on', 'off'):
        record = dict(session=session, rows=[])
        data['sessions'].append(record)
        with C.Server(directory, session, a.binary, session == 'on',
                      {'IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES': '1'}) as server:
            record['command'] = server.cmd
            record['state_after_start'] = C.counters(C.healthz())
            for phase, name, key in plan(session):
                content = key if phase == 'warmup' else fx[phase][key]['prompt']
                row = server.request(C.user(content), 64)
                row.update(phase=phase, name=name)
                row.pop('output', None)
                record['rows'].append(row)
                print(json.dumps(dict(session=session, name=f'{phase}:{name}', ttft_ms=round((row['ttft_s'] or 0) * 1000, 1),
                                      prefill_ms=row['server_prefill_ms'], counters=row['counters'],
                                      error=row['error'])), flush=True)
                out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
                time.sleep(1)
        record['server'] = server.summary
        record['prefill_phases'] = phases(server.log_path)
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
        time.sleep(5)
    data['complete'] = True
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
