#!/usr/bin/env python3
"""Four-app B1 fixed-256-token feasibility preflight (docs/protocol.md).

Not a performance ranking. Same frozen apps, models and drafter as
four-app-short-v1 (identity, start-up profiles, footprint sampler and owned
process cleanup reused from its runner), with BF16 KV everywhere (Splash:
explicit --kv-format bf16). oMLX 0.7.0 has no ignore_eos or equivalent, so
no request uses ignore_eos; every target request asks max_tokens=256.
Per app one fresh server: the two frozen warmups (unchanged request
settings), the six short tasks once each (max_tokens 256), then one
"answer only OK" probe (max_tokens 256). Serial, port 18480. Raw SSE lines
saved. Never retries; failures are recorded and collection continues."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

SUITE = Path(__file__).resolve().parents[1]
REPO = SUITE.parents[1]
sys.path.insert(0, str(REPO / 'benchmarks/four-app-short-v1/scripts'))
import plan as P  # noqa: E402
import run as R  # noqa: E402
from footprint import FootprintCollector, read_samples  # noqa: E402
from run_aligned import cleanup_owned  # noqa: E402

REPORTS = REPO / 'reports/four-app-b1-fixed256-preflight'
MAX_TOKENS = 256
PROBE = dict(id='probe-ok', category='probe', prompt='只回答 OK')
APPS = ('A', 'B', 'C', 'D')


def profiles(directory):
    prof = R.profiles(directory)
    prof['C']['command'] = prof['C']['command'] + ['--kv-format', 'bf16']
    return prof


def chat(app, profile, item, tokenizer, max_tokens):
    """Streaming request; the body equals the frozen client's except max_tokens."""
    body = dict(model=profile['model'], messages=[dict(role='user', content=item['prompt'])], stream=True,
                stream_options=dict(include_usage=True), temperature=0, top_p=1, max_tokens=max_tokens,
                chat_template_kwargs=dict(enable_thinking=False))
    if not profile['omit_effort']:
        body['reasoning_effort'] = 'none'
    data = json.dumps(body).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{P.PORT}/v1/chat/completions', data=data,
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    raw, events, usage, finish, error, status, done, reasoning, native_reply = [], [], None, None, None, None, False, [], None
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            status = response.status
            for line in response:
                raw.append(line.decode(errors='replace'))
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    done = True
                    continue
                if not payload:
                    continue
                obj = json.loads(payload)
                if obj.get('error'):
                    raise ValueError(str(obj['error']))
                if obj.get('tensorfold'):
                    native_reply = dict(tensorfold=obj['tensorfold'], speculative=obj.get('speculative'))
                if obj.get('usage'):
                    usage = obj['usage']
                for choice in obj.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    delta = choice.get('delta') or {}
                    if delta.get('reasoning_content') or delta.get('reasoning'):
                        reasoning.append(delta.get('reasoning_content') or delta['reasoning'])
                    if delta.get('content'):
                        events.append(dict(t=time.perf_counter() - started, text=delta['content']))
    except Exception as exc:  # recorded, never retried
        error = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, urllib.error.HTTPError):
            status = exc.code
            error += ': ' + exc.read().decode(errors='replace')
    output = ''.join(e['text'] for e in events)
    details = (usage or {}).get('prompt_tokens_details') or (usage or {}).get('input_tokens_details') or {}
    return dict(id=item['id'], request_sha256=hashlib.sha256(data).hexdigest(),
                request_settings={k: v for k, v in body.items() if k != 'messages'}, status=status, error=error,
                done=done, finish_reason=finish, usage=usage,
                native_output_tokens=(usage or {}).get('completion_tokens'),
                common_output_tokens=len(tokenizer.encode(output, add_special_tokens=False).ids),
                input_tokens=(usage or {}).get('prompt_tokens'), cached_tokens=details.get('cached_tokens', 0),
                reasoning=reasoning, native_reply=native_reply, ttft_s=events[0]['t'] if events else None,
                e2e_s=time.perf_counter() - started, output=output,
                output_sha256=hashlib.sha256(output.encode()).hexdigest(), raw_sse=raw)


def process_tree(root):
    rows = subprocess.getoutput('ps -axo pid=,ppid=,args=').splitlines()
    table = {}
    for row in rows:
        parts = row.strip().split(None, 2)
        if len(parts) >= 2:
            table[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) > 2 else '')
    keep, frontier = {root}, [root]
    while frontier:
        parent = frontier.pop()
        for pid, (ppid, _) in table.items():
            if ppid == parent and pid not in keep:
                keep.add(pid)
                frontier.append(pid)
    return [dict(pid=pid, ppid=table[pid][0], args=table[pid][1]) for pid in sorted(keep) if pid in table]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--protocol', type=Path, required=True)
    a = p.parse_args()
    directory = REPORTS / 'results' / a.label
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    prof = profiles(directory)
    fixtures = json.loads(P.FIXTURE.read_text())
    data = dict(schema=1, label=a.label, purpose='four-app-b1-fixed256-preflight', max_tokens=MAX_TOKENS,
                ignore_eos_used=False, apps=P.NAMES, profiles=prof, protocol_sha256=R.digest(a.protocol),
                scripts={str(s): R.digest(s) for s in sorted(SUITE.glob('scripts/*.py'))},
                reused_scripts={str(s): R.digest(s) for s in sorted((REPO / 'benchmarks/four-app-short-v1/scripts').glob('*.py'))},
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sessions=[], complete=False,
                failure=None)
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
    data['frozen'] = R.freeze(prof)
    save()
    client = R.load_client()
    tokenizer = client.Tokenizer.from_file(str(P.TARGET / 'tokenizer.json'))
    try:
        for index, app in enumerate(APPS):
            R.verify(data['frozen'], prof)
            profile = prof[app]
            name = f's{index}-{app}'
            log_path, memory_path = directory / f'{name}.server.log', directory / f'{name}.footprint.jsonl'
            session = dict(index=index, app=app, name=name, command=profile['command'],
                           env_keys=sorted(R.server_env(profile)), environment_before=R.environment(), requests=[])
            data['sessions'].append(session)
            save()
            with log_path.open('x') as log:
                process = subprocess.Popen(profile['command'], cwd=profile['cwd'], env=R.server_env(profile),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                footprint = FootprintCollector(memory_path, P.PROBE, process.pid, interval=0.05)
                footprint.start()
                started = time.monotonic()
                try:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'{app} exited: {process.returncode}')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{P.PORT}/v1/models', timeout=1) as r:
                                if r.status == 200:
                                    break
                        except (OSError, TimeoutError):
                            pass
                        if time.monotonic() - started > 600:
                            raise TimeoutError(app + ' startup')
                        time.sleep(0.5)
                    session['ready_s'] = time.monotonic() - started
                    session['process_tree_at_ready'] = process_tree(process.pid)
                    plan = [('warmup', w, 4096) for w in client.WARMUPS] + \
                        [('target', t, MAX_TOKENS) for t in fixtures] + [('probe', PROBE, MAX_TOKENS)]
                    for phase, item, limit in plan:
                        before = R.native_state(app)
                        row = chat(app, profile, item, tokenizer, limit)
                        after = R.native_state(app)
                        row.update(phase=phase, native_before=before, native_after=after)
                        session['requests'].append(row)
                        save()
                        print(json.dumps(dict(app=app, phase=phase, id=item['id'], native=row['native_output_tokens'],
                                              common=row['common_output_tokens'], finish=row['finish_reason'],
                                              cached=row['cached_tokens'], error=row['error']), ensure_ascii=False),
                              flush=True)
                        time.sleep(1)
                    session['healthz'] = R.get_json('/healthz') if app == 'A' else None
                finally:
                    session['footprint_collector'] = footprint.stop()
                    try:
                        session['remaining_owned_processes'] = cleanup_owned(process, footprint.probe)
                    except BaseException as error:
                        session['cleanup_error'] = f'{type(error).__name__}: {error}'
                        raise
                    finally:
                        session.update(server_exit_code=process.returncode, log_sha256=R.digest(log_path),
                                       footprint_sha256=R.digest(memory_path), environment_after=R.environment())
                        save()
            time.sleep(10)
        data['complete'] = True
    except BaseException as error:
        data['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        data['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save()


if __name__ == '__main__':
    main()
