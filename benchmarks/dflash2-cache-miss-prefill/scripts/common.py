"""Shared pieces for the DFlash2 cache-miss prefill suite.

Server: an ironmlx binary started with the arguments the App builds for the
Qwen3.8-27B DFlash2 model under default App settings (block 8, draft 4-bit,
capacity 32768, 8 GiB prefix cache, server-default sequences and chunk size),
in an App-like sanitized environment. Requests: OpenAI chat completions,
streamed, greedy, thinking disabled. Per request: client TTFT / Decode / E2E,
server prefill time and prefix-cache counter deltas from /healthz, and the
process-tree footprint peak from a 50 ms sampler.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / 'tools'))
from footprint import FootprintCollector, read_samples, window  # noqa: E402

PORT = 18490
TARGET = Path('/Users/xin/.ironmlx/models/huggingface/mlx-community--Qwen3.8-27B-4bit/snapshots/'
              '3e6447f082e89cc7f0bc6e5441afd38dfce760ff')
DRAFT = Path('/Users/xin/.ironmlx/models/huggingface/z-lab--Qwen3.8-27B-DFlash2/snapshots/'
             '50307d4c4cde6860d4eee73e2547cd786fe8e8a4')
METALLIB = Path('/Users/xin/.local/mlx/lib/mlx.metallib')
PREFIX_BYTES = 8589934592
CACHE_CAP = 32768
PROBE = SUITE / 'tools/footprint-probe.dylib'
PYTHON = Path('/Users/xin/workspace/b1-rival-benchmark/artifacts/tensorfold/venv/bin/python')
FIXTURES = {'short': SUITE / 'fixtures/short-prompts.json', '30k': SUITE / 'fixtures/30k-prompts.json'}
FIXTURE_SHA = {'short': '138468e163f568167c5579dbf9baaf4032a4ddb1979bfd0c11773c66135e84f6',
               '30k': 'a7ecb63f6a65504cc659bb3c01fcbfebadec61deb2b08ce99dbc1f1f5515a1f3'}
WARMUPS = ['请用三句话说明良好 API 错误信息应包含哪些内容。',
           '写一个 Python 函数，判断整数是否为偶数，并给出一个调用示例。']
COUNTERS = ('requests', 'prefill_us', 'generation_us', 'prefix_cache_hits', 'prefix_cache_misses',
            'prefix_cache_saves', 'prefix_cache_hit_tokens', 'prefix_cache_evictions')


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
        out[phase] = {row['id']: row for row in json.loads(path.read_text())}
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


def command(binary, cache_on, port=PORT, extra_args=(), prefix_bytes=PREFIX_BYTES):
    args = [str(binary), '--mlx-metallib', str(METALLIB), 'serve', '--host', '127.0.0.1', '--port', str(port),
            '--network-mode', 'local', '--model', str(TARGET), '--model-id', 'benchmark',
            '--dflash2-model-dir', str(DRAFT), '--dflash2-block-size', '8', '--dflash2-draft-bits', '4',
            '--max-cache-cap', str(CACHE_CAP)]
    if cache_on:
        args += ['--prefix-lru-cache-max-bytes', str(prefix_bytes)]
    return args + list(extra_args)


def environment(extra=None, allow_experimental=False):
    allowed = ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME')
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    env['IRONMLX_LOG_LEVEL'] = 'warn'
    env.update(extra or {})
    for key in env:
        if key.startswith(('DYLD_', 'MLX_')) or (key.startswith('IRONMLX_EXPERIMENTAL') and not allow_experimental):
            raise RuntimeError(f'{key} is not part of the App environment')
    return env


def port_free(port=PORT):
    with socket.socket() as check:
        return check.connect_ex(('127.0.0.1', port)) != 0


def healthz(port=PORT):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=10) as stream:
        return json.load(stream)


def counters(state):
    d = state['dflash2']
    return {k: d.get(k, 0) for k in COUNTERS} | {
        'prefix_cache_entries': d.get('prefix_cache_entries'), 'prefix_cache_bytes': d.get('prefix_cache_bytes'),
        'prefix_cache_enabled': d.get('prefix_cache_enabled')}


def delta(before, after):
    return {k: after[k] - before[k] for k in COUNTERS}


def chat(messages, max_tokens, port=PORT, timeout=1800):
    body = dict(model='benchmark', messages=messages, stream=True, stream_options=dict(include_usage=True),
                temperature=0, top_p=1, max_tokens=max_tokens, chat_template_kwargs=dict(enable_thinking=False))
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
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
                        times.append(time.perf_counter() - started)
                        texts.append(content)
    except Exception as exc:  # recorded, never retried
        error = f'{type(exc).__name__}: {exc}'
    e2e = time.perf_counter() - started
    output = ''.join(texts)
    tokens = (usage or {}).get('completion_tokens')
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    return dict(status=status, error=error, done=done, finish_reason=finish, usage=usage,
                completion_tokens=tokens, prompt_tokens=(usage or {}).get('prompt_tokens'),
                ttft_s=times[0] if times else None, decode_s=span,
                decode_tps=(tokens - 1) / span if tokens and tokens > 1 and span > 0 else None,
                e2e_s=e2e, output=output, output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                chunks=len(texts))


def valid(row):
    return (row['status'] == 200 and row['error'] is None and row['done'] and row['ttft_s'] is not None
            and row['finish_reason'] in ('stop', 'length'))


class Server:
    """One owned server process (own process group); only it is cleaned up."""

    def __init__(self, directory, name, binary, cache_on, extra_env=None, port=PORT, extra_args=(),
                 prefix_bytes=PREFIX_BYTES, allow_experimental=False):
        self.directory, self.name, self.port = directory, name, port
        self.cmd = command(binary, cache_on, port, extra_args, prefix_bytes)
        self.env = environment(extra_env, allow_experimental)
        self.log_path = directory / f'{name}.server.log'
        self.memory_path = directory / f'{name}.footprint.jsonl'
        self.process = self.collector = None

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
            self._wait_ready()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_ready(self):
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
        self.ready = time.monotonic()

    def request(self, messages, max_tokens):
        before = counters(healthz(self.port))
        begin = time.monotonic()
        row = chat(messages, max_tokens, self.port)
        end = time.monotonic()
        after_state = healthz(self.port)
        after = counters(after_state)
        row.update(counters=delta(before, after), cache_after={k: after[k] for k in (
            'prefix_cache_entries', 'prefix_cache_bytes', 'prefix_cache_enabled')},
            server_prefill_ms=(after['prefill_us'] - before['prefill_us']) / 1000,
            mlx_after={k: after_state['memory'].get(k) for k in ('mlx_active_bytes', 'mlx_cache_bytes')},
            wrapper_start_s=begin, wrapper_end_s=end)
        try:
            row['memory'] = window(read_samples(self.memory_path, live=True), begin, end)
            row['memory'].pop('peak_processes', None)
        except ValueError as error:
            row['memory'] = dict(error=str(error))
        return row

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


def user(content):
    return [dict(role='user', content=content)]
