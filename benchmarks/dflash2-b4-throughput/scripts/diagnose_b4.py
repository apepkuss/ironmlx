#!/usr/bin/env python3
"""B4 diagnosis (not a formal measurement). Fresh server per session with
request capacity 4 and capacity 8192; the fixed warmup batch, then the three
batch sets (rotated by session). Modes:
  plain   - no diagnostics (real throughput and path counters);
  steplog - IRONMLX_DIAGNOSTIC_DFLASH2_STEP_LOG only (host-wait phases);
  stages  - step log plus IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES=1
            (synchronized stage attribution; slows the run);
  draftcheck - IRONMLX_DIAGNOSTIC_DFLASH2_RAGGED_DRAFT_CHECK=1 (batched vs
            per-row draft comparison in every ragged window).
Per batch: throughput, completion time, per-request TTFT/decode/E2E, the
DFlash2 counters before/after and the scheduler step range (batch_count),
which maps step-log lines to batches."""
import argparse
import json
from pathlib import Path
import time

import common as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--mode', choices=('plain', 'steplog', 'stages', 'draftcheck'), required=True)
    p.add_argument('--sessions', type=int, default=1)
    p.add_argument('--env', action='append', default=[], help='extra diagnostic KEY=VALUE')
    a = p.parse_args()
    directory = C.REPORTS / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    data = dict(label=a.label, purpose='b4-diagnosis', mode=a.mode, extra_env=a.env, binary=str(a.binary),
                binary_sha256=C.digest(a.binary), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                sessions=[], complete=False)
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    for index in range(a.sessions):
        env = {}
        steplog = directory / f's{index}.steps.jsonl'
        if a.mode in ('steplog', 'stages'):
            env['IRONMLX_DIAGNOSTIC_DFLASH2_STEP_LOG'] = str(steplog)
        if a.mode == 'stages':
            env['IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES'] = '1'
        if a.mode == 'draftcheck':
            env['IRONMLX_DIAGNOSTIC_DFLASH2_RAGGED_DRAFT_CHECK'] = '1'
        for item in a.env:
            key, value = item.split('=', 1)
            if not key.startswith('IRONMLX_DIAGNOSTIC_'):
                raise SystemExit('only diagnostic variables may be added')
            env[key] = value
        record = dict(index=index, batches=[])
        data['sessions'].append(record)
        server = C.Server(directory, f's{index}', a.binary, 4, 8192, env)
        try:
            with server:
                record.update(command=server.cmd, env_keys=sorted(server.env))
                sets = C.BATCH_SETS[index % 3:] + C.BATCH_SETS[:index % 3]
                for kind, tasks in [('warmup', C.WARMUP_BATCH)] + [('scored', s) for s in sets]:
                    before = C.dflash2_state(C.healthz())
                    begin = time.monotonic()
                    rows, metrics = C.batch([fx['short'][t] for t in tasks])
                    end = time.monotonic()
                    after = C.dflash2_state(C.healthz())
                    for row, task in zip(rows, tasks):
                        row['task'] = task
                        row['valid'] = C.valid(row)
                    record['batches'].append(dict(kind=kind, tasks=list(tasks), metrics=metrics, rows=rows,
                                                  before=before, after=after,
                                                  memory=server.memory_window(begin, end)))
                    save()
                    print(json.dumps(dict(session=index, kind=kind, tasks=tasks,
                                          tps=round(metrics['throughput_tps'], 1), wall=round(metrics['wall_s'], 2),
                                          valid=all(r['valid'] for r in rows)), ensure_ascii=False), flush=True)
                    time.sleep(2)
                record['lifecycle_peak_bytes'] = server.lifecycle_peak()
                time.sleep(2)
        finally:
            record['server'] = server.summary
            save()
        time.sleep(10)
    data['complete'] = True
    save()


if __name__ == '__main__':
    main()
