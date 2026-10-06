#!/usr/bin/env python3
"""Correctness, cache-behaviour and fallback sessions (no timing judgement).

Every session runs a fresh server with the token-id diagnostic
(IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS) and the prefill-phase diagnostic, two
unrelated warmups, then its steps. A request chain for prompt P is: miss
[P]; exact repeat [P]; append [P, A, Q2]; fork [P, A, Q3], where A is the
arm's own answer to the miss. The 30K chain uses max_tokens 1024 (capacity
32768) and adds a document fork: the same reference material with a
different final question. Short requests use max_tokens 4096.

Sessions (`--set main`): B-on / C-on (baseline / candidate, App default
prefix cache), B-off / C-off (no prefix cache, reference).
Sessions (`--set docfork`): B-on, C-on, B-off, C-off with only the 30K
code-1 chain (re-records the document fork's token ids).
Sessions (`--set fallback`): profile off (B and C), candidate with the
single-prefill setting forced off, candidate with a 1 GiB prefix cache,
memory limit 40 GB (B and C), and a candidate cancellation session (a 30K
cold miss abandoned after 3 s, then a short miss and its repeat)."""
import argparse
import json
from pathlib import Path
import threading
import time
import urllib.request

import common as C

Q2 = '请用一句话总结上面的回答。'
Q3 = '请指出上面回答中最容易出错的一点。'
FORK_QUESTION = '请概括上述资料中出现最多的租户前缀，并说明你的判断依据。'
SHORT = ('code-1', 'knowledge-1', 'code-3')
BIN = C.SUITE / 'binaries'


def main_sessions(base, cand):
    chains = [('short', t) for t in SHORT] + [('30k', 'code-1')]
    return [dict(name=f'{arm}-{mode}', binary=binary, cache_on=mode == 'on', chains=chains)
            for arm, binary in (('B', base), ('C', cand)) for mode in ('on', 'off')]


def docfork_sessions(base, cand):
    chains = [('30k', 'code-1')]
    return [dict(name=f'{arm}-{mode}', binary=binary, cache_on=mode == 'on', chains=chains)
            for arm, binary in (('B', base), ('C', cand)) for mode in ('on', 'off')]


def fallback_sessions(base, cand):
    short = [('short', 'code-1'), ('short', 'knowledge-1')]
    return [
        dict(name='B-on-profile-off', binary=base, cache_on=True, chains=short,
             extra_args=['--m5-dflash2-profile', 'off']),
        dict(name='C-on-profile-off', binary=cand, cache_on=True, chains=short,
             extra_args=['--m5-dflash2-profile', 'off']),
        dict(name='C-on-single-prefill-off', binary=cand, cache_on=True, chains=short,
             extra_env={'IRONMLX_EXPERIMENTAL_DFLASH2_SINGLE_PREFILL': '0'}, allow_experimental=True),
        dict(name='C-on-cache-1g', binary=cand, cache_on=True, prefix_bytes=1 << 30,
             chains=short + [('30k', 'code-1')]),
        dict(name='B-on-memory-limit-40g', binary=base, cache_on=True, extra_args=['--memory-limit-total-gb', '40'],
             chains=[('short', 'code-1'), ('30k', 'code-1'), ('30k', 'knowledge-1'), ('short', 'knowledge-1')],
             simple=True),
        dict(name='C-on-memory-limit-40g', binary=cand, cache_on=True, extra_args=['--memory-limit-total-gb', '40'],
             chains=[('short', 'code-1'), ('30k', 'code-1'), ('30k', 'knowledge-1'), ('short', 'knowledge-1')],
             simple=True),
        dict(name='C-on-cancel', binary=cand, cache_on=True, chains=[('short', 'code-1')], cancel_first=True),
    ]


def cancel_30k(fx, port=C.PORT):
    """Start a 30K cold miss and drop the connection after 3 s."""
    body = dict(model='benchmark', messages=C.user(fx['30k']['knowledge-1']['prompt']), stream=True,
                temperature=0, top_p=1, max_tokens=1024, chat_template_kwargs=dict(enable_thinking=False))
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    result = {}

    def run():
        try:
            response = urllib.request.urlopen(request, timeout=5)
            result['status'] = response.status
            response.close()
        except Exception as exc:  # expected: timeout during prefill
            result['error'] = f'{type(exc).__name__}: {exc}'
    thread = threading.Thread(target=run)
    started = time.monotonic()
    thread.start()
    thread.join(3.2)
    result['abandoned_after_s'] = time.monotonic() - started
    thread.join(10)
    return result


def run_chain(server, fx, phase, task, rows, simple=False):
    item = fx[phase][task]
    max_tokens = 4096 if phase == 'short' else 1024
    steps = [('miss', C.user(item['prompt']))]
    if not simple:
        steps.append(('repeat', C.user(item['prompt'])))
    for step, messages in steps:
        rows.append(dict(phase=phase, task=task, step=step) | server.request(messages, max_tokens))
    if simple or not C.valid(rows[-1]):
        return
    answer = rows[-2]['output']
    for step, question in (('append', Q2), ('fork', Q3)):
        messages = C.user(item['prompt']) + [dict(role='assistant', content=answer), dict(role='user', content=question)]
        rows.append(dict(phase=phase, task=task, step=step) | server.request(messages, max_tokens))
    if phase == '30k':
        prompt = item['prompt'][:item['prompt'].rfind(item['original_question'])] + FORK_QUESTION
        rows.append(dict(phase=phase, task=task, step='doc-fork') | server.request(C.user(prompt), max_tokens))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--set', choices=('main', 'fallback', 'docfork'), required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    directory = C.SUITE / 'results' / a.label
    directory.mkdir(parents=True, exist_ok=False)
    fx = C.fixtures()
    sessions = dict(main=main_sessions, fallback=fallback_sessions, docfork=docfork_sessions)[a.set](
        a.baseline, a.candidate)
    data = dict(label=a.label, set=a.set, purpose='correctness-and-cache-behaviour',
                binaries={'B': dict(path=str(a.baseline), sha256=C.digest(a.baseline)),
                          'C': dict(path=str(a.candidate), sha256=C.digest(a.candidate))},
                questions=dict(append=Q2, fork=Q3, doc_fork=FORK_QUESTION),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sessions=[])
    out = directory / 'run.json'

    def save():
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    for spec in sessions:
        name = spec['name']
        token_path = directory / f'{name}.token-ids.jsonl'
        env = {'IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS': str(token_path),
               'IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES': '1'} | spec.get('extra_env', {})
        record = dict(name=name, binary=str(spec['binary']), rows=[])
        server = None
        data['sessions'].append(record)
        try:
            server = C.Server(directory, name, spec['binary'], spec['cache_on'], env,
                              extra_args=spec.get('extra_args', ()),
                              prefix_bytes=spec.get('prefix_bytes', C.PREFIX_BYTES),
                              allow_experimental=spec.get('allow_experimental', False))
            with server:
                record.update(command=server.cmd, env_keys=sorted(server.env), state_after_start=C.healthz())
                for text in C.WARMUPS:
                    record['rows'].append(dict(phase='warmup', task='warmup', step='warmup') |
                                          server.request(C.user(text), 64))
                if spec.get('cancel_first'):
                    record['cancel'] = cancel_30k(fx)
                    time.sleep(5)
                    record['state_after_cancel'] = C.counters(C.healthz())
                for phase, task in spec['chains']:
                    run_chain(server, fx, phase, task, record['rows'], spec.get('simple', False))
                    save()
                for row in record['rows']:
                    print(json.dumps(dict(session=name, phase=row['phase'], task=row['task'], step=row['step'],
                                          status=row['status'], finish=row['finish_reason'],
                                          tokens=row['completion_tokens'], prompt=row['prompt_tokens'],
                                          hit=row['counters']['prefix_cache_hit_tokens'],
                                          saves=row['counters']['prefix_cache_saves'],
                                          ttft_ms=round((row['ttft_s'] or 0) * 1000, 1),
                                          error=row['error']), ensure_ascii=False), flush=True)
                record['final_state'] = C.healthz()
                # Let the token-id diagnostic finish its last record before shutdown.
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    raw = token_path.read_text() if token_path.exists() else ''
                    if raw.endswith('\n') and raw.count('\n') >= len(record['rows']):
                        break
                    time.sleep(0.5)
        except Exception as error:  # recorded; the next session still runs
            record['failure'] = f'{type(error).__name__}: {error}'
            print(json.dumps(dict(session=name, failure=record['failure'])), flush=True)
        record['server'] = getattr(server, 'summary', None)
        record['token_ids_file'] = token_path.name
        save()
        time.sleep(5)
    data['complete'] = True
    save()


if __name__ == '__main__':
    main()
