#!/usr/bin/env python3
"""Check a running isolated Flash server: real B1/B2/B4, paging health and cancellation.

The server owns its memory policy. This client does not alter model files or
claim that application reads represent physical SSD traffic.
"""
import argparse
import base64
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:19086')
    parser.add_argument('--model', default='flash-qualification')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--stream-timing', action='store_true')
    args = parser.parse_args()
    assert args.rounds > 0 and args.max_tokens > 0

    def health():
        start = time.monotonic()
        with urllib.request.urlopen(args.url + '/healthz', timeout=30) as response:
            root = json.load(response)
        models = root.get('models', [])
        model = next((m for m in models if m['id'] == args.model), root)
        return root, model, (time.monotonic() - start) * 1000

    root, initial, _ = health()
    assert root['status'] == 'healthy', root
    assert initial['paging']['mode'] == 'ssd', 'this test must exercise paging'
    scheduler = initial.get('scheduler')
    if isinstance(scheduler, dict):
        assert scheduler.get('b_max', 4) >= 4
    polls, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            try:
                _, model, ms = health()
                polls.append({'ms': ms, 'paging': model['paging']})
            except Exception as error:
                polls.append({'error': str(error)})
            stop.wait(0.1)

    def request(prompt, stream=False, maximum=None):
        return urllib.request.Request(args.url + '/v1/chat/completions', data=json.dumps({
            'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': maximum or args.max_tokens, 'stream': stream,
            'chat_template_kwargs': {'enable_thinking': False},
            **({'stream_options': {'include_usage': True}} if stream and args.stream_timing else {}),
        }).encode(), headers={'Content-Type': 'application/json'})

    def complete(prompt):
        start = time.monotonic()
        if args.stream_timing:
            message = {'role': 'assistant', 'content': ''}
            first, finish, usage = None, None, None
            with urllib.request.urlopen(request(prompt, stream=True), timeout=600) as response:
                for raw in response:
                    if not raw.startswith(b'data: ') or raw.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(raw[6:])
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        for key in ['content', 'reasoning_content']:
                            if delta.get(key):
                                if first is None:
                                    first = time.monotonic() - start
                                message[key] = message.get(key, '') + delta[key]
                        if choice.get('finish_reason'):
                            finish = choice['finish_reason']
            elapsed = time.monotonic() - start
            assert usage is not None and finish is not None, 'missing terminal SSE usage/finish'
            return {'message': message, 'finish_reason': finish,
                    'completion_tokens': usage['completion_tokens'], 'seconds': elapsed,
                    'first_token_seconds': first, 'decode_seconds': elapsed - (first or elapsed)}
        with urllib.request.urlopen(request(prompt), timeout=600) as response:
            result = json.load(response)
        return {'message': result['choices'][0]['message'],
                'finish_reason': result['choices'][0]['finish_reason'],
                'completion_tokens': result['usage']['completion_tokens'],
                'seconds': time.monotonic() - start}

    def comparable(rows):
        return [{k: v for k, v in row.items() if not k.endswith('seconds')} for row in rows]

    prompts = ['请只回答：一加一等于几？', 'Explain recursion briefly.',
               'Write a Python function that adds two numbers.', '请用一句话介绍你自己。']
    worker = threading.Thread(target=poll)
    worker.start()
    result = {'initial': initial['paging'], 'batches': {}}
    failure = None
    try:
        reference = [complete(p) for p in prompts]
        result['reference'] = reference
        result['rounds'] = []
        for round_index in range(args.rounds):
            for width in [2, 4]:
                h_before, m_before, _ = health()
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=width) as executor:
                    rows = list(executor.map(complete, prompts[:width]))
                batch = {'rows': rows, 'seconds': time.monotonic() - started}
                h_after, m_after, _ = health()
                result['batches'][str(width)] = batch
                result['rounds'].append(dict(round=round_index, width=width, **batch,
                    dflash2_before=h_before.get('dflash2', m_before.get('dflash2')),
                    dflash2_after=h_after.get('dflash2', m_after.get('dflash2'))))
                print(json.dumps({'round': round_index, 'width': width, 'seconds': batch['seconds']}), flush=True)
                assert comparable(rows) == comparable(reference[:width]), f'B{width} differs from B1'
        if args.images:
            image = Path(__file__).resolve().parents[1] / 'ironmlx-lm/tests/fixtures/embedding_gemma2/images/red-circle.png'
            prompt = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()}},
                      {'type': 'text', 'text': 'Describe the image briefly.'}]
            result['image'] = complete(prompt)
        with urllib.request.urlopen(request(prompts[1], stream=True, maximum=128), timeout=600) as response:
            while True:
                line = response.readline()
                assert line, 'SSE ended before the first event'
                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                    result['cancel_first_event'] = json.loads(line[6:])
                    break
        deadline = time.monotonic() + 120
        while True:
            _, model, _ = health()
            scheduler = model.get('scheduler')
            active = model.get('active_requests', scheduler.get('b_active', 0) if isinstance(scheduler, dict) else 0)
            if not active:
                break
            assert time.monotonic() < deadline, 'cancelled request did not release its slot'
            time.sleep(0.1)
        result['recovery'] = complete(prompts[0])
        assert comparable([result['recovery']]) == comparable(reference[:1])
        if args.baseline:
            previous = json.loads(args.baseline.read_text())
            assert comparable(reference) == comparable(previous['reference']), 'route/control output differs'
            if args.images:
                assert comparable([result['image']]) == comparable([previous['image']])
    except Exception as error:
        failure = error
        result['failure'] = repr(error)
    finally:
        stop.set()
        worker.join(timeout=35)
        _, model, _ = health()
        result['final'] = model['paging']
        result['polls'] = polls
        valid = sorted(p['ms'] for p in polls if 'ms' in p)
        if valid:
            result['health_p50_ms'] = valid[len(valid)//2]
            result['health_p95_ms'] = valid[min(len(valid)-1, int(len(valid)*0.95))]
        result['poll_errors'] = sum('error' in p for p in polls)
        for cache in ['experts', 'ple']:
            stats = result['final'][cache]
            assert stats['charged_bytes'] <= stats['budget_bytes'], f'{cache} exceeded its cache budget'
            assert stats['peak_charged_bytes'] >= stats['charged_bytes'], f'{cache} invalid peak accounting'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if failure:
        raise failure
    assert result['poll_errors'] == 0
    print(json.dumps({'status': 'pass', 'output': str(args.output),
                      'health_p95_ms': result.get('health_p95_ms')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
