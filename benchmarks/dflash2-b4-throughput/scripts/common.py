"""Shared pieces for the DFlash2 B4 throughput round.

Server: an ironmlx binary with the product defaults for the Qwen3.8-27B
DFlash2 model (M5 profile automatic), request capacity 4 (`--max-sequences
4`) and capacity 8192 for B4 (the earlier B4 protocol), or capacity 1 for B1
checks; no prefix cache; App-like sanitized environment. Requests are
streamed chat completions, greedy, thinking off.

Large artifacts (binaries, raw results, footprint samples) live under
reports/dflash2-b4-throughput/ (git-ignored); this directory keeps scripts,
protocols, reports and small evidence.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

SUITE = Path(__file__).resolve().parents[1]
REPO = SUITE.parents[1]
REPORTS = REPO / 'reports/dflash2-b4-throughput'
sys.path.insert(0, str(SUITE / 'tools'))
from footprint import FootprintCollector, read_samples, window  # noqa: E402

PORT = 18490
TARGET = Path('/Users/xin/.ironmlx/models/huggingface/mlx-community--Qwen3.8-27B-4bit/snapshots/'
              '3e6447f082e89cc7f0bc6e5441afd38dfce760ff')
DRAFT = Path('/Users/xin/.ironmlx/models/huggingface/z-lab--Qwen3.8-27B-DFlash2/snapshots/'
             '50307d4c4cde6860d4eee73e2547cd786fe8e8a4')
METALLIB = Path('/Users/xin/.local/mlx/lib/mlx.metallib')
PROBE = REPORTS / 'tools/footprint-probe.dylib'
PYTHON = Path('/Users/xin/workspace/b1-rival-benchmark/artifacts/tensorfold/venv/bin/python')
FIXTURES = {'short': SUITE / 'fixtures/short-prompts.json', '30k': SUITE / 'fixtures/30k-prompts.json'}
FIXTURE_SHA = {'short': '138468e163f568167c5579dbf9baaf4032a4ddb1979bfd0c11773c66135e84f6',
               '30k': 'a7ecb63f6a65504cc659bb3c01fcbfebadec61deb2b08ce99dbc1f1f5515a1f3'}
# B4 workload of b1-b4-concurrency-three-app-v1: three batch sets of four
# short tasks with different answer lengths, one fixed warmup batch.
BATCH_SETS = (('code-1', 'code-2', 'code-3', 'knowledge-1'),
              ('knowledge-2', 'knowledge-3', 'code-1', 'code-2'),
              ('code-3', 'knowledge-1', 'knowledge-2', 'knowledge-3'))
WARMUPS = {'warmup-1': '请用三句话说明良好 API 错误信息应包含哪些内容。',
           'warmup-2': '写一个 Python 函数，判断整数是否为偶数，并给出一个调用示例。'}
WARMUP_BATCH = ('warmup-1', 'warmup-2', 'warmup-1', 'warmup-2')
MAX_TOKENS = 4096


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fixtures():
    out = {}
    for phase, path in FIXTURES.items():
        if digest(path) != FIXTURE_SHA[phase]:
            raise RuntimeError(f'{phase} fixture identity differs')
        out[phase] = {row['id']: row['prompt'] for row in json.loads(path.read_text())}
    out['short'].update(WARMUPS)
    return out


def model_identity():
    rows = {}
    for root in (TARGET, DRAFT):
        for path in sorted(root.iterdir()):
            real = path.resolve()
            entry = dict(size=real.stat().st_size)
            if real.stat().st_size < (64 << 20):
                entry['sha256'] = digest(real)
            rows[str(path)] = entry
    return rows


def command(binary, sequences, capacity, port=PORT, extra_args=()):
    return [str(binary), '--mlx-metallib', str(METALLIB), 'serve', '--host', '127.0.0.1', '--port', str(port),
            '--network-mode', 'local', '--model', str(TARGET), '--model-id', 'benchmark',
            '--dflash2-model-dir', str(DRAFT), '--max-sequences', str(sequences),
            '--max-cache-cap', str(capacity)] + list(extra_args)


def environment(extra=None):
    allowed = ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME')
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    env['IRONMLX_LOG_LEVEL'] = 'warn'
    env.update(extra or {})
    for key in env:
        if key.startswith(('DYLD_', 'MLX_', 'IRONMLX_EXPERIMENTAL')):
            raise RuntimeError(f'{key} is not part of the App environment')
    return env


def port_free(port=PORT):
    with socket.socket() as check:
        return check.connect_ex(('127.0.0.1', port)) != 0


def healthz(port=PORT):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=10) as stream:
        return json.load(stream)


def dflash2_state(state):
    d = state.get('dflash2') or {}
    keep = ('requests', 'windows', 'tree_windows', 'ordinary_windows', 'drafted_tokens', 'accepted_draft_tokens',
            'rollback_count', 'tensor_batch_windows', 'tensor_batch_divergent_splits', 'tensor_batch_groups_created',
            'tensor_batch_max_width', 'prefill_us', 'generation_us', 'window_us', 'draft_build_us',
            'verify_build_us', 'host_sync_us', 'rollback_us', 'block_size', 'tree_max_nodes',
            'current_draft_budget')
    out = {k: d.get(k) for k in keep}
    out['ragged_linear'] = d.get('ragged_linear')
    out['m5_profile_status'] = (d.get('m5_profile') or {}).get('status')
    out['scheduler'] = {k: (state.get('scheduler') or {}).get(k) for k in ('b_max', 'batch_count', 'admit_count')}
    out['memory'] = {k: (state.get('memory') or {}).get(k) for k in ('mlx_active_bytes', 'mlx_cache_bytes',
                                                                      'mlx_peak_bytes')}
    return out


def chat(prompt, max_tokens=MAX_TOKENS, port=PORT, timeout=1800):
    body = dict(model='benchmark', messages=[dict(role='user', content=prompt)], stream=True,
                stream_options=dict(include_usage=True), temperature=0, top_p=1, max_tokens=max_tokens,
                chat_template_kwargs=dict(enable_thinking=False))
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    send = time.perf_counter()
    times, texts, usage, finish, error, done, status = [], [], None, None, None, False, None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    continue
                if not data:
                    continue
                payload = json.loads(data)
                if payload.get('error'):
                    raise ValueError(str(payload['error']))
                if payload.get('usage'):
                    usage = payload['usage']
                for choice in payload.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    content = (choice.get('delta') or {}).get('content')
                    if content:
                        times.append(time.perf_counter() - send)
                        texts.append(content)
    except Exception as exc:  # recorded, never retried
        error = f'{type(exc).__name__}: {exc}'
    end = time.perf_counter()
    output = ''.join(texts)
    tokens = (usage or {}).get('completion_tokens')
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    return dict(status=status, error=error, done=done, finish_reason=finish, usage=usage,
                completion_tokens=tokens, prompt_tokens=(usage or {}).get('prompt_tokens'),
                send_s=send, end_s=end, ttft_s=times[0] if times else None, decode_s=span,
                decode_tps=(tokens - 1) / span if tokens and tokens > 1 and span > 0 else None,
                e2e_s=end - send, output=output, output_sha256=hashlib.sha256(output.encode()).hexdigest())


def valid(row):
    return (row['status'] == 200 and row['error'] is None and row['done'] and row['ttft_s'] is not None
            and row['finish_reason'] == 'stop' and isinstance(row['completion_tokens'], int)
            and 1 < row['completion_tokens'] < MAX_TOKENS and row['decode_s'] > 0)


def batch(prompts, port=PORT):
    """Release the requests together, one thread each (barrier)."""
    barrier = threading.Barrier(len(prompts))
    rows = [None] * len(prompts)

    def worker(i, prompt):
        barrier.wait()
        rows[i] = dict(chat(prompt, port=port), slot=i)
    threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    start = min(r['send_s'] for r in rows)
    end = max(r['end_s'] for r in rows)
    tokens = sum(r['completion_tokens'] or 0 for r in rows)
    return rows, dict(output_tokens=tokens, wall_s=end - start, throughput_tps=tokens / (end - start),
                      send_spread_s=max(r['send_s'] for r in rows) - start)


class Server:
    """One owned server process (own process group); only it is cleaned up."""

    def __init__(self, directory, name, binary, sequences, capacity, extra_env=None, port=PORT, extra_args=()):
        self.directory, self.name, self.port = directory, name, port
        self.cmd = command(binary, sequences, capacity, port, extra_args)
        self.env = environment(extra_env)
        self.log_path = directory / f'{name}.server.log'
        self.memory_path = directory / f'{name}.footprint.jsonl'
        self.process = self.collector = None
        self.summary = None

    def __enter__(self):
        if not port_free(self.port):
            raise RuntimeError(f'Port {self.port} occupied; refusing to interfere')
        self.log = self.log_path.open('x')
        self.process = subprocess.Popen(self.cmd, env=self.env, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        self.started = time.monotonic()
        try:
            self.collector = FootprintCollector(self.memory_path, PROBE, self.process.pid, interval=0.05)
            self.collector.start()
            deadline = self.started + 900
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(f'server exited: {self.process.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/v1/models', timeout=1) as r:
                        if r.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup')
                time.sleep(0.5)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.ready = time.monotonic()
        return self

    def memory_window(self, begin, end):
        try:
            out = window(read_samples(self.memory_path, live=True), begin, end)
            out.pop('peak_processes', None)
            return out
        except ValueError as error:
            return dict(error=str(error))

    def lifecycle_peak(self):
        samples = read_samples(self.memory_path, live=True)
        return max(s['tree_footprint_bytes'] for s in samples) if samples else None

    def __exit__(self, *exc):
        sampler = self.collector.stop() if self.collector else None
        remaining = []
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        if self.process:
            try:
                os.killpg(self.process.pid, 0)
                remaining.append(self.process.pid)
            except ProcessLookupError:
                pass
        self.log.close()
        self.summary = dict(sampler=sampler, exit_code=self.process.returncode if self.process else None,
                            remaining_group=remaining, log_sha256=digest(self.log_path))
        return False
