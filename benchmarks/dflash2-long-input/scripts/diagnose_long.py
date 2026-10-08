#!/usr/bin/env python3
"""Long-input diagnosis (not a formal measurement). Fresh server per session,
`--max-sequences 1`, capacity 40960. Each session: one short warmup, then the
listed 30K tasks once each (max_tokens 1024), 5 s apart. Modes:
  plain   - no diagnostics (TTFT / Decode / E2E / memory);
  phases  - IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES (per-chunk prefill time,
            clock reads at existing sync points) plus the step log with
            IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES (decode window stages);
  stages  - IRONMLX_DIAGNOSTIC_PREFILL_STAGES (per-layer prefill stage times;
            adds an evaluation at every stage boundary).
Per request: client TTFT / Decode / E2E, server prefill and generation time
(/healthz deltas), footprint peak; server-log diagnostic lines are parsed."""
import argparse
import json
from pathlib import Path
import re
import time

import common as C

MODES = {
    'plain': {},
    'phases': {'IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES': '1', 'IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES': '1'},
    'stages': {'IRONMLX_DIAGNOSTIC_PREFILL_STAGES': '1'},
}


def parse_log(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
    phases, stages = [], []
    for line in text.splitlines():
        if 'dflash2_prefill_phases' in line:
            fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
            parts = [p.split(':') for p in fields['phases'].strip('"').split(',')]
            phases.append(dict(prompt_len=int(fields['prompt_len']),
                               phases=[dict(phase=p[0], size=int(p[1]), us=int(p[2])) for p in parts]))
        elif line.startswith('prefill_stage '):
            fields = dict(re.findall(r'(\w+)=(\S+)', line))
            stages.append(dict(layer=int(fields['layer']), kind=fields['kind'], stage=fields['stage'],
                               us=int(fields['elapsed_us'])))
    return phases, stages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--mode', choices=tuple(MODES), required=True)
    p.add_argument('--sessions', type=int, default=1)
    p.add_argument('--tasks', default='code-1,knowledge-1')
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    tasks = a.tasks.split(',')
    data = dict(label=a.label, purpose='long-input-diagnosis', mode=a.mode, tasks=tasks, binary=str(a.binary),
                binary_sha256=C.digest(a.binary), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                sessions=[], complete=False)
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    for index in range(a.sessions):
        env = dict(MODES[a.mode])
        if a.mode == 'phases':
            env['IRONMLX_DIAGNOSTIC_DFLASH2_STEP_LOG'] = str(directory / f's{index}.steps.jsonl')
        record = dict(index=index, rows=[])
        data['sessions'].append(record)
        server = C.Server(directory, f's{index}', a.binary, 1, 40960, env)
        try:
            with server:
                record.update(command=server.cmd, env_keys=sorted(server.env))
                C.chat(C.WARMUPS['warmup-1'], max_tokens=64)
                time.sleep(2)
                for task in tasks:
                    before = C.dflash2_state(C.healthz())
                    begin = time.monotonic()
                    row = C.chat(fx['30k'][task], max_tokens=1024)
                    end = time.monotonic()
                    after = C.dflash2_state(C.healthz())
                    row.pop('output', None)
                    row.update(task=task, valid=C.valid(row), before=before, after=after,
                               server_prefill_ms=(after['prefill_us'] - before['prefill_us']) / 1000,
                               server_generation_ms=(after['generation_us'] - before['generation_us']) / 1000,
                               memory=server.memory_window(begin, end))
                    record['rows'].append(row)
                    save()
                    print(json.dumps(dict(session=index, task=task, valid=row['valid'],
                                          ttft_s=round(row['ttft_s'] or 0, 2), decode=round(row['decode_tps'] or 0, 1),
                                          tokens=row['completion_tokens']), ensure_ascii=False), flush=True)
                    time.sleep(5)
                record['lifecycle_peak_bytes'] = server.lifecycle_peak()
        finally:
            record['server'] = server.summary
            record['prefill_phases'], record['prefill_stages'] = parse_log(server.log_path)
            save()
        time.sleep(10)
    data['complete'] = True
    save()


if __name__ == '__main__':
    main()
