#!/usr/bin/env python3
"""Measure first-miss Flash prefix latency against an isolated running server.

Records every nonempty SSE event and macOS process counters; no cache clearing,
model changes, pressure injection or performance acceptance is implicit. Run
modes separately with pinned binaries/settings, and retain warmups and outliers.
"""
import argparse
import ctypes
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import urllib.request
from pathlib import Path


RUSAGE_FIELDS = [
    'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
    'wired_size', 'resident_size', 'phys_footprint', 'proc_start_abstime',
    'proc_exit_abstime', 'child_user_time', 'child_system_time',
    'child_pkg_idle_wkups', 'child_interrupt_wkups', 'child_pageins',
    'child_elapsed_abstime', 'diskio_bytesread', 'diskio_byteswritten',
]


class RusageInfo(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in RUSAGE_FIELDS
    ]


def process_counters(pid):
    if pid is None or sys.platform != 'darwin':
        return None
    result = RusageInfo()
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    if lib.proc_pid_rusage(pid, 2, ctypes.byref(result)) != 0:
        return {'error': ctypes.get_errno()}
    return {name: getattr(result, name) for name in RUSAGE_FIELDS}


def counter_delta(before, after):
    if not before or not after or 'error' in before or 'error' in after:
        return None
    return {key: after[key] - before[key] for key in [
        'user_time', 'system_time', 'pageins', 'diskio_bytesread',
        'diskio_byteswritten', 'phys_footprint',
    ]}


def get_health(url):
    with urllib.request.urlopen(url + '/healthz', timeout=30) as response:
        return json.load(response)


def drain(url, timeout=120):
    deadline = time.monotonic() + timeout
    while True:
        health = get_health(url)
        if health.get('prefix_persistence', {}).get('pending_jobs', 0) == 0:
            return health
        if time.monotonic() >= deadline:
            raise TimeoutError('prefix writer failed to drain')
        time.sleep(.02)


def decode_events(lines, start, now=time.perf_counter):
    events, content, reasoning = [], '', ''
    usage, finish, done = None, None, False
    for line in lines:
        if not line.startswith(b'data: '):
            continue
        raw = line[6:].strip()
        if raw == b'[DONE]':
            done = True
            break
        event = json.loads(raw)
        if event.get('usage'):
            usage = event['usage']
        nonempty = False
        for choice in event.get('choices', []):
            delta = choice.get('delta', {})
            content += delta.get('content') or ''
            reasoning += delta.get('reasoning_content') or ''
            nonempty |= bool(delta.get('content') or delta.get('reasoning_content'))
            finish = choice.get('finish_reason') or finish
        if nonempty:
            events.append(now() - start)
    end = now() - start
    if not done or not usage or not finish or not events:
        raise ValueError('incomplete SSE response: require content, usage, finish and DONE')
    gaps = [b - a for a, b in zip(events, events[1:])]
    return dict(ttft_s=events[0], total_s=end, decode_s=end - events[0],
                event_seconds=events, max_content_gap_s=max(gaps, default=0),
                content=content, reasoning=reasoning, usage=usage, finish_reason=finish)


def stream(url, model, prompt, max_tokens):
    request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(dict(
        model=model, messages=[dict(role='user', content=prompt)], temperature=0,
        max_tokens=max_tokens, stream=True, stream_options={'include_usage': True},
        chat_template_kwargs={'enable_thinking': False},
    )).encode(), headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        return decode_events(response, start)


def summary(rows):
    result = {'samples': len(rows)}
    for name in ['ttft_s', 'decode_s', 'max_content_gap_s']:
        values = sorted(row[name] for row in rows)
        if not values:
            continue
        item = dict(median=statistics.median(values), minimum=values[0], maximum=values[-1])
        if len(values) >= 20:
            item['p95_nearest_rank'] = values[math.ceil(.95 * len(values)) - 1]
        result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:19086')
    parser.add_argument('--model', default='flash-qualification')
    parser.add_argument('--pid', type=int)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=500)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--nonce', required=True, help='Use a fresh nonce for each fresh store.')
    parser.add_argument('--label', required=True, help='Record the externally selected route/mode.')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.repeat, args.max_tokens) < 1 or args.warmups < 0:
        parser.error('positive sample/prompt/token counts required; warmups must be nonnegative')
    result = dict(label=args.label, binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  platform=platform.platform(), started_unix_s=time.time(), arguments=vars(args).copy(),
                  initial_health=get_health(args.url), warmups=[], samples=[])
    result['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in result['arguments'].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmups + args.samples):
        before = drain(args.url)
        c0 = process_counters(args.pid)
        prompt = (f'Experiment code {args.nonce}-{index}.\n' +
                  'The project uses Rust for local inference. ' * args.repeat +
                  '\nWrite the numbers 1 to 100 separated by spaces and nothing else.')
        row = stream(args.url, args.model, prompt, args.max_tokens)
        c1 = process_counters(args.pid)
        after = drain(args.url)
        row.update(index=index, before=before, after=after, process_before=c0,
                   process_after=c1, process_delta=counter_delta(c0, c1))
        key = 'warmups' if index < args.warmups else 'samples'
        result[key].append(row)
        result['summary'] = summary(result['samples'])
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(dict(label=args.label, phase=key, index=index,
                              ttft_s=row['ttft_s'], decode_s=row['decode_s'],
                              max_content_gap_s=row['max_content_gap_s'],
                              process_delta=row['process_delta'])), flush=True)


if __name__ == '__main__':
    main()
