#!/usr/bin/env python3
"""Run the request-local or cross-request PromptLookup qualification matrix.

The runner owns ironmlx-specific orchestration while keeping iron-bench engine
neutral. It starts baseline and PromptLookup servers, calibrates deterministic
corpus prompts with the model tokenizer, drives streaming HTTP requests, and
    checks output parity, scheduler health, index lifecycle, source-specific
    economics, and performance gates.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import math
import os
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CORPUS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "prompt_lookup_corpus_v1.json"
)
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".ironmlx"
    / "models"
    / "models--mlx-community--Qwen3.5-2B-4bit"
    / "snapshots"
    / "674aaa7240b91e8012fcad5d791b7dfe5ba90207"
)

WORDS = (
    "amber", "apricot", "aster", "atlas", "birch", "cobalt", "coral",
    "cedar", "cipher", "comet", "delta", "ember", "falcon", "fern",
    "harbor", "indigo", "ivory", "juniper", "lantern", "lilac", "maple",
    "meadow", "nickel", "onyx", "orbit", "pearl", "pine", "prism",
    "quartz", "river", "saffron", "silver", "summit", "sunrise", "teal",
    "thistle", "umber", "velvet", "violet", "willow", "winter", "zephyr",
)


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    category: str
    polarity: str
    generator: str
    description: str


@dataclass(frozen=True)
class LookupConfig:
    name: str
    min_ngram: int = 2
    max_ngram: int = 4
    max_draft_tokens: int = 4
    history_window_tokens: int = 32768
    max_index_entries: int = 65536

    @classmethod
    def parse(cls, raw: str) -> "LookupConfig":
        parts = raw.split(":")
        if len(parts) != 6:
            raise argparse.ArgumentTypeError(
                "lookup config must be name:min_ngram:max_ngram:draft:window:entries"
            )
        name = parts[0].strip()
        if not name:
            raise argparse.ArgumentTypeError("lookup config name must not be empty")
        try:
            values = [int(value) for value in parts[1:]]
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        config = cls(name, *values)
        if config.min_ngram <= 0 or config.min_ngram > config.max_ngram:
            raise argparse.ArgumentTypeError("lookup n-gram range is invalid")
        if min(
            config.max_draft_tokens,
            config.history_window_tokens,
            config.max_index_entries,
        ) <= 0:
            raise argparse.ArgumentTypeError("lookup limits must be positive")
        return config


@dataclass(frozen=True)
class Variant:
    name: str
    cache_mode: str
    lookup: Optional[LookupConfig]
    round_index: int


@dataclass(frozen=True)
class ResolvedPrompt:
    case: CorpusCase
    target_prompt_tokens: int
    max_tokens: int
    context_units: int
    prompt_tokens_local: int


@dataclass
class MatrixConfig:
    root: Path
    model_dir: Path
    out_root: Path
    serve_bin: Path
    tokenizer_bin: Path
    corpus_path: Path = DEFAULT_CORPUS
    model_name: str = "default"
    host: str = "127.0.0.1"
    port: int = 19120
    prompt_tokens: Tuple[int, ...] = (1024, 8192)
    max_tokens: Tuple[int, ...] = (128,)
    concurrency: Tuple[int, ...] = (1, 2)
    max_sequences: Optional[int] = None
    max_cache_cap: Optional[int] = None
    prefill_chunk_size: int = 2048
    runs: int = 3
    warmup_batches: int = 1
    lookup_configs: Tuple[LookupConfig, ...] = (
        LookupConfig(name="default"),
    )
    cross_request: bool = False
    include_prefix_cache: bool = False
    balanced: bool = False
    categories: Tuple[str, ...] = tuple()
    timeout_secs: int = 900
    startup_timeout_secs: int = 180
    mlx_dir: Path = Path.home() / ".local" / "mlx"
    rust_log: str = "info"
    build: bool = False
    dry_run: bool = False
    allow_failures: bool = False
    resume: bool = False
    extra_serve_args: Tuple[str, ...] = field(default_factory=tuple)


class TokenizerSidecar:
    def __init__(self, binary: Path, model_dir: Path):
        self._process = subprocess.Popen(
            [str(binary), "--model-dir", str(model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()

    def tokenize(self, text: str, include_ids: bool = False) -> Dict[str, Any]:
        request = json.dumps({"text": text, "include_ids": include_ids})
        with self._lock:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(
                    "tokenizer sidecar exited with code {}: {}".format(
                        self._process.returncode, stderr
                    )
                )
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
            response = self._process.stdout.readline()
        if not response:
            raise RuntimeError("tokenizer sidecar returned EOF")
        return json.loads(response)

    def count(self, text: str) -> int:
        return int(self.tokenize(text)["token_count"])

    def close(self, raise_on_error: bool = True) -> None:
        error = None
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=5)
        if raise_on_error and self._process.returncode not in (0, None):
            stderr = self._process.stderr.read() if self._process.stderr else ""
            error = RuntimeError(
                "tokenizer sidecar exited with code {}: {}".format(
                    self._process.returncode, stderr
                )
            )
        for stream in (
            self._process.stdin,
            self._process.stdout,
            self._process.stderr,
        ):
            if stream is not None and not stream.closed:
                stream.close()
        if error is not None:
            raise error

    def __enter__(self) -> "TokenizerSidecar":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(raise_on_error=exc_type is None)


def load_corpus(path: Path) -> List[CorpusCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported corpus schema_version")
    cases = []
    seen = set()
    for raw in payload.get("cases") or []:
        case_id = str(raw["id"])
        if case_id in seen:
            raise ValueError("duplicate corpus case id: {}".format(case_id))
        seen.add(case_id)
        polarity = str(raw["polarity"])
        if polarity not in ("positive", "negative", "adversarial"):
            raise ValueError("invalid polarity for {}: {}".format(case_id, polarity))
        cases.append(
            CorpusCase(
                case_id=case_id,
                category=str(raw["category"]),
                polarity=polarity,
                generator=str(raw["generator"]),
                description=str(raw.get("description") or ""),
            )
        )
    if not cases:
        raise ValueError("corpus contains no cases")
    return cases


def stable_words(index: int, count: int, salt: int = 0) -> str:
    return " ".join(
        WORDS[(index * 17 + offset * 29 + salt * 11) % len(WORDS)]
        for offset in range(count)
    )


def copy_payload(case: CorpusCase, request_seed: int, max_tokens: int) -> str:
    marker = "{}-{:04d}".format(case.case_id.replace("_", "-"), request_seed)
    words = [marker, "payload", "begins"]
    words.extend(
        WORDS[(request_seed * 7 + index * 13 + len(case.case_id)) % len(WORDS)]
        for index in range(max_tokens * 3)
    )
    return " ".join(words)


def render_case(
    case: CorpusCase,
    context_units: int,
    request_seed: int,
    max_tokens: int,
) -> Tuple[str, Optional[str]]:
    if context_units <= 0:
        raise ValueError("context_units must be positive")
    generator = case.generator
    payload = copy_payload(case, request_seed, max_tokens)
    target_index = max(0, min(context_units - 1, context_units * 3 // 4))
    marker = "control-{:04d}".format(request_seed)
    units = []

    for index in range(context_units):
        body = stable_words(index, 24, salt=request_seed)
        if generator == "rag":
            value = payload if index == target_index else body
            units.append(
                "Document KB-{:06d}: owner team-{:03d}; resolution payload: {}."
                .format(index, index % 97, value)
            )
        elif generator == "code":
            if index == target_index:
                units.append(
                    "fn requested_copy_{seed:04d}() -> &'static str {{\n"
                    "    \"{payload}\"\n"
                    "}}".format(seed=request_seed, payload=payload)
                )
            else:
                units.append(
                    "fn helper_{index:06d}(input: &str) -> String {{\n"
                    "    format!(\"unit-{index:06d} {body} {{}}\", input)\n"
                    "}}".format(index=index, body=body)
                )
        elif generator == "json":
            value = payload if index == target_index else body
            units.append(
                json.dumps(
                    {
                        "record_id": "REC-{:06d}".format(index),
                        "partition": index % 31,
                        "payload": value,
                    },
                    separators=(",", ":"),
                )
            )
        elif generator == "long_copy":
            value = payload if index == target_index else body
            units.append("Passage {:06d}: {}".format(index, value))
        elif generator == "negative":
            units.append(
                "Research note {:06d} compares {} under condition {:04d}."
                .format(index, body, (index * 37 + request_seed) % 10000)
            )
        elif generator == "adversarial":
            continuation = stable_words(index + request_seed, 12, salt=index % 7)
            if index == target_index:
                continuation = payload
            units.append(
                "service route alpha beta continuation {}: {}"
                .format(index, continuation)
            )
        else:
            raise ValueError("unknown corpus generator: {}".format(generator))

    context = "\n".join(units)
    if generator == "rag":
        query = (
            "Return only the resolution payload from Document KB-{index:06d}. "
            "Copy it verbatim and continue until the output limit."
        ).format(index=target_index)
        expected_prefix = payload.split(" ", 12)[0:12]
    elif generator == "code":
        query = (
            "Return only the string literal contents from requested_copy_{seed:04d}. "
            "Copy verbatim and continue until the output limit."
        ).format(seed=request_seed)
        expected_prefix = payload.split(" ", 12)[0:12]
    elif generator == "json":
        query = (
            "Return only the payload value from REC-{index:06d}. "
            "Copy it verbatim and continue until the output limit."
        ).format(index=target_index)
        expected_prefix = payload.split(" ", 12)[0:12]
    elif generator == "long_copy":
        query = (
            "Return only Passage {index:06d}. Copy it verbatim and continue "
            "until the output limit."
        ).format(index=target_index)
        expected_prefix = payload.split(" ", 12)[0:12]
    elif generator == "negative":
        query = (
            "Begin with {marker}, then explain in original wording why green leaves "
            "reflect more green light than red light. Do not quote the notes."
        ).format(marker=marker)
        expected_prefix = [marker]
    else:
        query = (
            "Return only the continuation at service route entry {index}. The many "
            "identical prefixes have conflicting continuations, so use that exact entry."
        ).format(index=target_index)
        expected_prefix = payload.split(" ", 12)[0:12]

    prompt = "Context:\n{}\n\nTask:\n{}".format(context, query)
    return prompt, " ".join(expected_prefix)


def resolve_prompt(
    case: CorpusCase,
    target_prompt_tokens: int,
    max_tokens: int,
    token_counter,
) -> ResolvedPrompt:
    if target_prompt_tokens <= 0:
        raise ValueError("target_prompt_tokens must be positive")

    cache: Dict[int, int] = {}

    def count(units: int) -> int:
        if units not in cache:
            prompt, _ = render_case(case, units, request_seed=0, max_tokens=max_tokens)
            cache[units] = int(token_counter(prompt))
        return cache[units]

    low = 1
    high = 1
    while count(high) < target_prompt_tokens:
        low = high
        high *= 2
        if high > 1_000_000:
            raise RuntimeError("unable to bracket prompt token target")

    candidates = {low, high}
    while low + 1 < high:
        mid = (low + high) // 2
        candidates.add(mid)
        if count(mid) < target_prompt_tokens:
            low = mid
        else:
            high = mid
    candidates.update((low, high))
    units = min(candidates, key=lambda value: (abs(count(value) - target_prompt_tokens), value))
    return ResolvedPrompt(
        case=case,
        target_prompt_tokens=target_prompt_tokens,
        max_tokens=max_tokens,
        context_units=units,
        prompt_tokens_local=count(units),
    )


def build_variants(config: MatrixConfig) -> List[Variant]:
    cache_modes = ["off"]
    if config.include_prefix_cache:
        cache_modes.append("prefix")
    variants = []
    for cache_mode in cache_modes:
        round_index = 0
        variants.append(
            Variant(
                name="baseline_{}".format(cache_mode),
                cache_mode=cache_mode,
                lookup=None,
                round_index=round_index,
            )
        )
        round_index += 1
        for lookup in config.lookup_configs:
            variants.append(
                Variant(
                    name="lookup_{}_{}".format(lookup.name, cache_mode),
                    cache_mode=cache_mode,
                    lookup=lookup,
                    round_index=round_index,
                )
            )
            round_index += 1
        if config.balanced:
            for lookup in reversed(config.lookup_configs):
                variants.append(
                    Variant(
                        name="lookup_{}_{}".format(lookup.name, cache_mode),
                        cache_mode=cache_mode,
                        lookup=lookup,
                        round_index=round_index,
                    )
                )
                round_index += 1
            variants.append(
                Variant(
                    name="baseline_{}".format(cache_mode),
                    cache_mode=cache_mode,
                    lookup=None,
                    round_index=round_index,
                )
            )
    return variants


def build_serve_command(
    config: MatrixConfig,
    variant: Variant,
    variant_dir: Path,
) -> List[str]:
    max_cache_cap = config.max_cache_cap or (
        max(config.prompt_tokens) + max(config.max_tokens) + 1024
    )
    cmd = [
        str(config.serve_bin),
        "serve",
        "--model",
        str(config.model_dir),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--max-sequences",
        str(config.max_sequences or max(config.concurrency)),
        "--max-cache-cap",
        str(max_cache_cap),
        "--prefill-chunk-size",
        str(config.prefill_chunk_size),
        "--force-scheduler",
    ]
    if variant.lookup is not None:
        lookup = variant.lookup
        cmd.extend(
            [
                "--prompt-lookup",
                "--prompt-lookup-min-ngram",
                str(lookup.min_ngram),
                "--prompt-lookup-max-ngram",
                str(lookup.max_ngram),
                "--prompt-lookup-max-draft-tokens",
                str(lookup.max_draft_tokens),
                "--prompt-lookup-history-window-tokens",
                str(lookup.history_window_tokens),
                "--prompt-lookup-max-index-entries",
                str(lookup.max_index_entries),
            ]
        )
        if config.cross_request:
            cmd.append("--prompt-lookup-cross-request")
    if variant.cache_mode == "prefix":
        cache_dir = variant_dir / "prefix-cache"
        cmd.extend(["--paged-prefix-cache-dir", str(cache_dir)])
    cmd.extend(config.extra_serve_args)
    return cmd


def fetch_json(host: str, port: int, path: str, timeout: float = 10) -> Dict[str, Any]:
    with urllib.request.urlopen(
        "http://{}:{}{}".format(host, port, path), timeout=timeout
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_ready(
    config: MatrixConfig,
    process: subprocess.Popen,
    log_path: Path,
) -> None:
    deadline = time.time() + config.startup_timeout_secs
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "server exited before readiness with code {}. Log tail:\n{}".format(
                    process.returncode, tail_text(log_path, 6000)
                )
            )
        try:
            health = fetch_json(config.host, config.port, "/healthz", timeout=2)
            if health.get("status") in ("healthy", "degraded"):
                return
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(
        "server did not become ready on {}:{} within {}s: {}".format(
            config.host, config.port, config.startup_timeout_secs, last_error
        )
    )


def wait_idle(config: MatrixConfig) -> Dict[str, Any]:
    deadline = time.time() + 10
    last = None
    while time.time() < deadline:
        last = fetch_json(config.host, config.port, "/healthz")
        scheduler = last.get("scheduler") or {}
        lookup = last.get("prompt_lookup") or {}
        if (
            int(scheduler.get("b_active") or 0) == 0
            and int(scheduler.get("b_queued") or 0) == 0
            and int(lookup.get("index_entries_current") or 0) == 0
        ):
            return last
        time.sleep(0.05)
    raise RuntimeError("server did not return to an idle request-local state: {}".format(last))


def clear_shared_prompt_lookup(config: MatrixConfig) -> Dict[str, Any]:
    request = urllib.request.Request(
        "http://{}:{}/admin/api/prompt-lookup/clear".format(
            config.host, config.port
        ),
        data=b"{}",
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_port_available(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as error:
        raise RuntimeError("benchmark port {}:{} is unavailable: {}".format(host, port, error))
    finally:
        sock.close()


def start_server(
    config: MatrixConfig,
    variant: Variant,
    variant_dir: Path,
) -> Tuple[subprocess.Popen, Any, List[str]]:
    ensure_port_available(config.host, config.port)
    if variant.cache_mode == "prefix":
        cache_dir = variant_dir / "prefix-cache"
        if cache_dir.exists():
            shutil.rmtree(str(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
    command = build_serve_command(config, variant, variant_dir)
    log_path = variant_dir / "server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    home_dir = variant_dir / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home_dir)
    env.setdefault("MLX_DIR", str(config.mlx_dir))
    env["RUST_LOG"] = config.rust_log
    mlx_lib = str(config.mlx_dir / "lib")
    existing_dyld = env.get("DYLD_LIBRARY_PATH")
    env["DYLD_LIBRARY_PATH"] = (
        mlx_lib if not existing_dyld else mlx_lib + os.pathsep + existing_dyld
    )
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        start_new_session=True,
    )
    try:
        wait_ready(config, process, log_path)
    except Exception:
        terminate_process(process)
        log_handle.close()
        raise
    return process, log_handle, command


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def tail_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def health_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "queries",
        "hits",
        "misses",
        "drafted_tokens",
        "accepted_tokens",
        "rejected_tokens",
        "zero_accept_windows",
        "index_evictions",
        "verify_round_us",
        "verify_forward_us",
        "projection_us",
        "exact_batched_verify_windows",
        "sequential_verify_windows",
        "verify_accept_host_sync_count",
        "verify_accept_host_sync_us",
        "rollback_count",
        "rollback_us",
        "mtp_shadow_commit_windows",
        "mtp_shadow_commit_tokens",
        "mtp_shadow_commit_us",
        "hybrid_neural_windows",
        "hybrid_lookup_windows",
        "hybrid_source_switches",
        "hybrid_lookup_miss_fallbacks",
        "hybrid_neural_rebases",
        "hybrid_neural_rebase_us",
        "qualification_query_gate_skips",
        "miss_query_gate_skips",
        "miss_query_reprobes",
        "adaptive_draft_width_reductions",
        "adaptive_draft_width_increases",
        "adaptive_profitability_width_reductions",
        "shared_queries",
        "shared_hits",
        "shared_misses",
        "shared_mtp_certified_published_windows",
        "shared_mtp_certified_published_tokens",
        "shared_mtp_certified_hits",
        "shared_mtp_canonical_validation_windows",
        "shared_mtp_canonical_validation_tokens",
        "shared_mtp_canonical_validation_us",
        "shared_mtp_canonical_validation_mismatches",
        "shared_mtp_canonical_fallbacks",
        "shared_published_requests",
        "shared_published_tokens",
        "shared_evictions",
        "shared_pressure_evictions",
        "shared_clear_count",
        "shared_cleared_entries",
    )
    before_lookup = before.get("prompt_lookup") or {}
    after_lookup = after.get("prompt_lookup") or {}
    lookup = {
        key: max(
            0,
            int(after_lookup.get(key) or 0) - int(before_lookup.get(key) or 0),
        )
        for key in keys
    }
    lookup.update(
        {
            "enabled": bool(after_lookup.get("enabled")),
            "index_entries_current": int(after_lookup.get("index_entries_current") or 0),
            "index_entries_peak": int(after_lookup.get("index_entries_peak") or 0),
            "shared_entries_current": int(
                after_lookup.get("shared_entries_current") or 0
            ),
            "shared_entries_peak": int(after_lookup.get("shared_entries_peak") or 0),
            "shared_estimated_bytes_current": int(
                after_lookup.get("shared_estimated_bytes_current") or 0
            ),
            "shared_estimated_bytes_peak": int(
                after_lookup.get("shared_estimated_bytes_peak") or 0
            ),
        }
    )
    source_keys = (
        "queries",
        "hits",
        "misses",
        "drafted_tokens",
        "accepted_tokens",
        "zero_accept_windows",
        "wasted_verify_tokens",
        "propose_us",
        "verify_us",
        "rollback_us",
    )
    for source in ("local_source", "shared_source"):
        before_source = before_lookup.get(source) or {}
        after_source = after_lookup.get(source) or {}
        lookup[source] = {
            key: max(
                0,
                int(after_source.get(key) or 0) - int(before_source.get(key) or 0),
            )
            for key in source_keys
        }
    before_scheduler = before.get("scheduler") or {}
    after_scheduler = after.get("scheduler") or {}
    scheduler_deltas = {}
    for key in (
        "admit_count",
        "batch_count",
        "admission_queue_full_count",
        "memory_budget_exceeded_count",
    ):
        scheduler_deltas[key + "_delta"] = max(
            0,
            int(after_scheduler.get(key) or 0) - int(before_scheduler.get(key) or 0),
        )
    governor = ((after.get("memory") or {}).get("process_governor") or {})
    return {
        "status": after.get("status"),
        "scheduler": scheduler_deltas,
        "prompt_lookup": lookup,
        "pressure_level": governor.get("pressure_level"),
        "process_current_usage_bytes": int(
            governor.get("current_usage_bytes") or 0
        ),
        "process_effective_ceiling_bytes": int(
            governor.get("effective_ceiling_bytes") or 0
        ),
        "mlx_peak_bytes": int((after.get("memory") or {}).get("mlx_peak_bytes") or 0),
    }


def run_chat_completion(
    config: MatrixConfig,
    prompt: str,
    max_tokens: int,
    next_admission: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    body = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_body = json.dumps(body).encode("utf-8")
    start = time.perf_counter()
    first_token = None
    token_times = []
    content_parts = []
    usage = {}
    finish_reason = None
    connection = http.client.HTTPConnection(
        config.host,
        config.port,
        timeout=config.timeout_secs,
    )
    admission_released = False
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=request_body,
            headers={"content-type": "application/json"},
        )
        response = connection.getresponse()
        if next_admission is not None:
            next_admission.set()
            admission_released = True
        with response:
            if response.status >= 400:
                body_text = response.read().decode("utf-8", errors="replace")
                raise RuntimeError("HTTP {}: {}".format(response.status, body_text))
            request_id = response.headers.get("X-Ironmlx-Request-Id")
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                event = json.loads(payload)
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        now = time.perf_counter()
                        if first_token is None:
                            first_token = now
                        token_times.append(now)
                        content_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")
                if event.get("usage") is not None:
                    usage = event["usage"]
    finally:
        if next_admission is not None and not admission_released:
            next_admission.set()
        connection.close()
    end = time.perf_counter()
    first = first_token if first_token is not None else end
    intervals_ms = [
        (right - left) * 1000.0 for left, right in zip(token_times, token_times[1:])
    ]
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(token_times)
    generation_seconds = max(end - first, 1e-9)
    return {
        "content": "".join(content_parts),
        "request_id": request_id,
        "prompt_tokens_server": usage.get("prompt_tokens"),
        "completion_tokens_server": completion_tokens,
        "finish_reason": finish_reason or "unknown",
        "ttft_ms": (first - start) * 1000.0,
        "e2e_s": end - start,
        "tg_tps": float(completion_tokens) / generation_seconds,
        "itl_ms": intervals_ms,
    }


def run_batch(
    config: MatrixConfig,
    requests: Sequence[Tuple[str, Optional[str], Dict[str, Any]]],
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], float]:
    workers = len(requests)
    admission_turns = [threading.Event() for _ in range(workers)]
    admission_turns[0].set()

    def run_one(index, item):
        prompt, expected_prefix, metadata = item
        admission_turns[index].wait()
        next_admission = (
            admission_turns[index + 1] if index + 1 < workers else None
        )
        result = run_chat_completion(
            config,
            prompt,
            max_tokens,
            next_admission=next_admission,
        )
        result["expected_prefix"] = expected_prefix
        result.update(metadata)
        return result

    batch_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one, index, item)
            for index, item in enumerate(requests)
        ]
        results = [future.result() for future in futures]
    return results, time.perf_counter() - batch_start


def percentile(values: Iterable[float], quantile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    numbers = [float(value) for value in values if value is not None]
    return statistics.median(numbers) if numbers else None


def percent_change(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None or baseline == 0:
        return None
    return (current / baseline - 1.0) * 100.0


def output_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row["cache_mode"],
        row["case_id"],
        row["target_prompt_tokens"],
        row["max_tokens"],
        row["concurrency"],
        row["batch_idx"],
        row["worker_id"],
    )


def attach_output_parity(rows: List[Dict[str, Any]]) -> None:
    baseline_hashes: Dict[Tuple[Any, ...], set] = {}
    for row in rows:
        if row["lookup_name"] is None:
            baseline_hashes.setdefault(output_key(row), set()).add(row["output_token_hash"])
    for row in rows:
        hashes = baseline_hashes.get(output_key(row), set())
        row["baseline_consistent"] = len(hashes) == 1
        row["baseline_match"] = (
            row["output_token_hash"] in hashes if row["lookup_name"] is not None else len(hashes) == 1
        )


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
    cell_health: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["cache_mode"],
            row["lookup_name"],
            row["case_id"],
            row["category"],
            row["polarity"],
            row["target_prompt_tokens"],
            row["max_tokens"],
            row["concurrency"],
            row.get("cross_request_role", "request_local"),
        )
        grouped.setdefault(key, []).append(row)

    health_grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for item in cell_health:
        key = (
            item["variant"],
            item["case_id"],
            item["target_prompt_tokens"],
            item["max_tokens"],
            item["concurrency"],
            item.get("cross_request_role", "request_local"),
        )
        health_grouped.setdefault(key, []).append(item)

    summary = []
    for key, samples in sorted(grouped.items(), key=lambda item: str(item[0])):
        (
            variant,
            cache_mode,
            lookup_name,
            case_id,
            category,
            polarity,
            target_prompt_tokens,
            max_tokens,
            concurrency,
            cross_request_role,
        ) = key
        health_items = health_grouped.get(
            (
                variant,
                case_id,
                target_prompt_tokens,
                max_tokens,
                concurrency,
                cross_request_role,
            ),
            [],
        )
        lookup_deltas = [
            item["delta"]["prompt_lookup"] for item in health_items
        ]
        local_deltas = [item["local_source"] for item in lookup_deltas]
        shared_source_deltas = [item["shared_source"] for item in lookup_deltas]
        verify_sync_count = sum(
            item["verify_accept_host_sync_count"] for item in lookup_deltas
        )
        verify_round_us = sum(item["verify_round_us"] for item in lookup_deltas)
        all_itl = [value for sample in samples for value in sample["itl_ms"]]
        row = {
            "variant": variant,
            "cache_mode": cache_mode,
            "lookup_name": lookup_name,
            "case_id": case_id,
            "category": category,
            "polarity": polarity,
            "target_prompt_tokens": target_prompt_tokens,
            "prompt_tokens_local": int(median(sample["prompt_tokens_local"] for sample in samples) or 0),
            "prompt_tokens_server": median(sample["prompt_tokens_server"] for sample in samples),
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "cross_request_role": cross_request_role,
            "samples": len(samples),
            "output_match_ratio": sum(bool(sample["baseline_match"]) for sample in samples) / len(samples),
            "baseline_consistent_ratio": sum(bool(sample["baseline_consistent"]) for sample in samples) / len(samples),
            "expected_prefix_match_ratio": sum(bool(sample["expected_prefix_match"]) for sample in samples) / len(samples),
            "ttft_ms_median": median(sample["ttft_ms"] for sample in samples),
            "ttft_ms_p95": percentile((sample["ttft_ms"] for sample in samples), 0.95),
            "e2e_s_median": median(sample["e2e_s"] for sample in samples),
            "e2e_s_p95": percentile((sample["e2e_s"] for sample in samples), 0.95),
            "tg_tps_median": median(sample["tg_tps"] for sample in samples),
            "itl_ms_p95": percentile(all_itl, 0.95),
            "aggregate_tps_median": median(sample["batch_aggregate_tps"] for sample in samples),
            "lookup_queries": sum(item["queries"] for item in lookup_deltas),
            "lookup_hits": sum(item["hits"] for item in lookup_deltas),
            "lookup_drafted_tokens": sum(item["drafted_tokens"] for item in lookup_deltas),
            "lookup_accepted_tokens": sum(item["accepted_tokens"] for item in lookup_deltas),
            "lookup_rejected_tokens": sum(item["rejected_tokens"] for item in lookup_deltas),
            "lookup_rollbacks": sum(item["rollback_count"] for item in lookup_deltas),
            "qualification_query_gate_skips": sum(
                item["qualification_query_gate_skips"] for item in lookup_deltas
            ),
            "miss_query_gate_skips": sum(
                item["miss_query_gate_skips"] for item in lookup_deltas
            ),
            "miss_query_reprobes": sum(
                item["miss_query_reprobes"] for item in lookup_deltas
            ),
            "adaptive_draft_width_reductions": sum(
                item["adaptive_draft_width_reductions"] for item in lookup_deltas
            ),
            "adaptive_draft_width_increases": sum(
                item["adaptive_draft_width_increases"] for item in lookup_deltas
            ),
            "adaptive_profitability_width_reductions": sum(
                item["adaptive_profitability_width_reductions"]
                for item in lookup_deltas
            ),
            "local_queries": sum(item["queries"] for item in local_deltas),
            "local_hits": sum(item["hits"] for item in local_deltas),
            "local_misses": sum(item["misses"] for item in local_deltas),
            "local_drafted_tokens": sum(
                item["drafted_tokens"] for item in local_deltas
            ),
            "local_accepted_tokens": sum(
                item["accepted_tokens"] for item in local_deltas
            ),
            "local_zero_accept_windows": sum(
                item["zero_accept_windows"] for item in local_deltas
            ),
            "local_wasted_verify_tokens": sum(
                item["wasted_verify_tokens"] for item in local_deltas
            ),
            "local_propose_us": sum(item["propose_us"] for item in local_deltas),
            "local_verify_us": sum(item["verify_us"] for item in local_deltas),
            "local_rollback_us": sum(item["rollback_us"] for item in local_deltas),
            "shared_queries": sum(item["shared_queries"] for item in lookup_deltas),
            "shared_hits": sum(item["shared_hits"] for item in lookup_deltas),
            "shared_misses": sum(item["shared_misses"] for item in lookup_deltas),
            "shared_mtp_certified_published_windows": sum(
                item["shared_mtp_certified_published_windows"]
                for item in lookup_deltas
            ),
            "shared_mtp_certified_published_tokens": sum(
                item["shared_mtp_certified_published_tokens"]
                for item in lookup_deltas
            ),
            "shared_mtp_certified_hits": sum(
                item["shared_mtp_certified_hits"] for item in lookup_deltas
            ),
            "shared_mtp_canonical_validation_windows": sum(
                item["shared_mtp_canonical_validation_windows"]
                for item in lookup_deltas
            ),
            "shared_mtp_canonical_validation_tokens": sum(
                item["shared_mtp_canonical_validation_tokens"]
                for item in lookup_deltas
            ),
            "shared_mtp_canonical_validation_us": sum(
                item["shared_mtp_canonical_validation_us"]
                for item in lookup_deltas
            ),
            "shared_mtp_canonical_validation_mismatches": sum(
                item["shared_mtp_canonical_validation_mismatches"]
                for item in lookup_deltas
            ),
            "shared_mtp_canonical_fallbacks": sum(
                item["shared_mtp_canonical_fallbacks"]
                for item in lookup_deltas
            ),
            "shared_drafted_tokens": sum(
                item["drafted_tokens"] for item in shared_source_deltas
            ),
            "shared_accepted_tokens": sum(
                item["accepted_tokens"] for item in shared_source_deltas
            ),
            "shared_zero_accept_windows": sum(
                item["zero_accept_windows"] for item in shared_source_deltas
            ),
            "shared_wasted_verify_tokens": sum(
                item["wasted_verify_tokens"] for item in shared_source_deltas
            ),
            "shared_propose_us": sum(
                item["propose_us"] for item in shared_source_deltas
            ),
            "shared_verify_us": sum(
                item["verify_us"] for item in shared_source_deltas
            ),
            "shared_rollback_us": sum(
                item["rollback_us"] for item in shared_source_deltas
            ),
            "shared_published_requests": sum(
                item["shared_published_requests"] for item in lookup_deltas
            ),
            "shared_published_tokens": sum(
                item["shared_published_tokens"] for item in lookup_deltas
            ),
            "shared_evictions": sum(
                item["shared_evictions"] for item in lookup_deltas
            ),
            "shared_pressure_evictions": sum(
                item["shared_pressure_evictions"] for item in lookup_deltas
            ),
            "shared_clear_count": sum(
                item["shared_clear_count"] for item in lookup_deltas
            ),
            "shared_cleared_entries": sum(
                item["shared_cleared_entries"] for item in lookup_deltas
            ),
            "mtp_shadow_commit_windows": sum(
                item["mtp_shadow_commit_windows"] for item in lookup_deltas
            ),
            "mtp_shadow_commit_tokens": sum(
                item["mtp_shadow_commit_tokens"] for item in lookup_deltas
            ),
            "mtp_shadow_commit_us": sum(
                item["mtp_shadow_commit_us"] for item in lookup_deltas
            ),
            "hybrid_neural_windows": sum(
                item["hybrid_neural_windows"] for item in lookup_deltas
            ),
            "hybrid_lookup_windows": sum(
                item["hybrid_lookup_windows"] for item in lookup_deltas
            ),
            "hybrid_source_switches": sum(
                item["hybrid_source_switches"] for item in lookup_deltas
            ),
            "hybrid_lookup_miss_fallbacks": sum(
                item["hybrid_lookup_miss_fallbacks"] for item in lookup_deltas
            ),
            "hybrid_neural_rebases": sum(
                item["hybrid_neural_rebases"] for item in lookup_deltas
            ),
            "hybrid_neural_rebase_us": sum(
                item["hybrid_neural_rebase_us"] for item in lookup_deltas
            ),
            "lookup_verify_round_us": verify_round_us,
            "lookup_verify_round_us_per_window": (
                verify_round_us / verify_sync_count if verify_sync_count else None
            ),
            "lookup_verify_forward_submit_us": sum(
                item["verify_forward_us"] for item in lookup_deltas
            ),
            "lookup_projection_submit_us": sum(
                item["projection_us"] for item in lookup_deltas
            ),
            "lookup_verify_accept_host_sync_count": verify_sync_count,
            "lookup_verify_accept_host_sync_us": sum(
                item["verify_accept_host_sync_us"] for item in lookup_deltas
            ),
            "index_entries_current_max": max(
                (item["index_entries_current"] for item in lookup_deltas), default=0
            ),
            "index_entries_peak_max": max(
                (item["index_entries_peak"] for item in lookup_deltas), default=0
            ),
            "shared_entries_current_max": max(
                (item["shared_entries_current"] for item in lookup_deltas), default=0
            ),
            "shared_entries_peak_max": max(
                (item["shared_entries_peak"] for item in lookup_deltas), default=0
            ),
            "shared_estimated_bytes_current_max": max(
                (item["shared_estimated_bytes_current"] for item in lookup_deltas),
                default=0,
            ),
            "shared_estimated_bytes_peak_max": max(
                (item["shared_estimated_bytes_peak"] for item in lookup_deltas),
                default=0,
            ),
            "process_current_usage_bytes_max": max(
                (
                    item["delta"]["process_current_usage_bytes"]
                    for item in health_items
                ),
                default=0,
            ),
            "process_effective_ceiling_bytes_min": min(
                (
                    item["delta"]["process_effective_ceiling_bytes"]
                    for item in health_items
                    if item["delta"]["process_effective_ceiling_bytes"] > 0
                ),
                default=0,
            ),
            "mlx_peak_bytes_max": max(
                (item["delta"]["mlx_peak_bytes"] for item in health_items),
                default=0,
            ),
            "server_healthy": all(
                item["delta"].get("status") == "healthy"
                and item["delta"]["scheduler"]["admission_queue_full_count_delta"] == 0
                and item["delta"]["scheduler"]["memory_budget_exceeded_count_delta"] == 0
                for item in health_items
            ),
            "scheduler_path_observed": bool(health_items)
            and all(
                item["delta"]["scheduler"]["batch_count_delta"] > 0
                for item in health_items
            ),
        }
        row["lookup_acceptance_ratio"] = (
            row["lookup_accepted_tokens"] / row["lookup_drafted_tokens"]
            if row["lookup_drafted_tokens"]
            else None
        )
        row["local_acceptance_ratio"] = (
            row["local_accepted_tokens"] / row["local_drafted_tokens"]
            if row["local_drafted_tokens"]
            else None
        )
        row["shared_acceptance_ratio"] = (
            row["shared_accepted_tokens"] / row["shared_drafted_tokens"]
            if row["shared_drafted_tokens"]
            else None
        )
        summary.append(row)
    return summary


def build_comparisons(
    config: MatrixConfig,
    summary: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baselines = {}
    for row in summary:
        if row["lookup_name"] is None:
            key = (
                row["cache_mode"],
                row["case_id"],
                row["target_prompt_tokens"],
                row["max_tokens"],
                row["concurrency"],
                row.get("cross_request_role", "request_local"),
            )
            baselines[key] = row
    comparisons = []
    for row in summary:
        if row["lookup_name"] is None:
            continue
        key = (
            row["cache_mode"],
            row["case_id"],
            row["target_prompt_tokens"],
            row["max_tokens"],
            row["concurrency"],
            row.get("cross_request_role", "request_local"),
        )
        baseline = baselines.get(key)
        if baseline is None:
            continue
        comparisons.append(
            {
                "variant": row["variant"],
                "lookup_name": row["lookup_name"],
                "cache_mode": row["cache_mode"],
                "case_id": row["case_id"],
                "category": row["category"],
                "polarity": row["polarity"],
                "target_prompt_tokens": row["target_prompt_tokens"],
                "max_tokens": row["max_tokens"],
                "concurrency": row["concurrency"],
                "cross_request_role": row.get(
                    "cross_request_role", "request_local"
                ),
                "output_match_ratio": row["output_match_ratio"],
                "baseline_consistent_ratio": row["baseline_consistent_ratio"],
                "expected_prefix_match_ratio": row["expected_prefix_match_ratio"],
                "baseline_ttft_ms": baseline["ttft_ms_median"],
                "lookup_ttft_ms": row["ttft_ms_median"],
                "ttft_change_pct": percent_change(row["ttft_ms_median"], baseline["ttft_ms_median"]),
                "ttft_change_ms": (
                    row["ttft_ms_median"] - baseline["ttft_ms_median"]
                    if row["ttft_ms_median"] is not None
                    and baseline["ttft_ms_median"] is not None
                    else None
                ),
                "e2e_change_pct": percent_change(row["e2e_s_median"], baseline["e2e_s_median"]),
                "e2e_p95_change_pct": percent_change(row["e2e_s_p95"], baseline["e2e_s_p95"]),
                "tg_change_pct": percent_change(row["tg_tps_median"], baseline["tg_tps_median"]),
                "aggregate_tps_change_pct": percent_change(
                    row["aggregate_tps_median"], baseline["aggregate_tps_median"]
                ),
                "itl_p95_change_pct": percent_change(row["itl_ms_p95"], baseline["itl_ms_p95"]),
                "lookup_queries": row["lookup_queries"],
                "lookup_hits": row["lookup_hits"],
                "lookup_drafted_tokens": row["lookup_drafted_tokens"],
                "lookup_accepted_tokens": row["lookup_accepted_tokens"],
                "lookup_rejected_tokens": row["lookup_rejected_tokens"],
                "lookup_acceptance_ratio": row["lookup_acceptance_ratio"],
                "lookup_rollbacks": row["lookup_rollbacks"],
                "qualification_query_gate_skips": row.get(
                    "qualification_query_gate_skips", 0
                ),
                "miss_query_gate_skips": row.get("miss_query_gate_skips", 0),
                "miss_query_reprobes": row.get("miss_query_reprobes", 0),
                "adaptive_draft_width_reductions": row.get(
                    "adaptive_draft_width_reductions", 0
                ),
                "adaptive_draft_width_increases": row.get(
                    "adaptive_draft_width_increases", 0
                ),
                "adaptive_profitability_width_reductions": row.get(
                    "adaptive_profitability_width_reductions", 0
                ),
                "local_queries": row.get("local_queries", 0),
                "local_hits": row.get("local_hits", 0),
                "local_misses": row.get("local_misses", 0),
                "local_drafted_tokens": row.get("local_drafted_tokens", 0),
                "local_accepted_tokens": row.get("local_accepted_tokens", 0),
                "local_zero_accept_windows": row.get(
                    "local_zero_accept_windows", 0
                ),
                "local_wasted_verify_tokens": row.get(
                    "local_wasted_verify_tokens", 0
                ),
                "local_propose_us": row.get("local_propose_us", 0),
                "local_verify_us": row.get("local_verify_us", 0),
                "local_rollback_us": row.get("local_rollback_us", 0),
                "shared_queries": row.get("shared_queries", 0),
                "shared_hits": row.get("shared_hits", 0),
                "shared_misses": row.get("shared_misses", 0),
                "shared_drafted_tokens": row.get("shared_drafted_tokens", 0),
                "shared_accepted_tokens": row.get(
                    "shared_accepted_tokens", 0
                ),
                "shared_zero_accept_windows": row.get(
                    "shared_zero_accept_windows", 0
                ),
                "shared_wasted_verify_tokens": row.get(
                    "shared_wasted_verify_tokens", 0
                ),
                "shared_propose_us": row.get("shared_propose_us", 0),
                "shared_verify_us": row.get("shared_verify_us", 0),
                "shared_rollback_us": row.get("shared_rollback_us", 0),
                "shared_published_requests": row.get(
                    "shared_published_requests", 0
                ),
                "shared_published_tokens": row.get(
                    "shared_published_tokens", 0
                ),
                "shared_evictions": row.get("shared_evictions", 0),
                "shared_pressure_evictions": row.get(
                    "shared_pressure_evictions", 0
                ),
                "shared_mtp_certified_published_windows": row.get(
                    "shared_mtp_certified_published_windows", 0
                ),
                "shared_mtp_certified_published_tokens": row.get(
                    "shared_mtp_certified_published_tokens", 0
                ),
                "shared_mtp_certified_hits": row.get(
                    "shared_mtp_certified_hits", 0
                ),
                "shared_mtp_canonical_validation_windows": row.get(
                    "shared_mtp_canonical_validation_windows", 0
                ),
                "shared_mtp_canonical_validation_tokens": row.get(
                    "shared_mtp_canonical_validation_tokens", 0
                ),
                "shared_mtp_canonical_validation_us": row.get(
                    "shared_mtp_canonical_validation_us", 0
                ),
                "shared_mtp_canonical_validation_mismatches": row.get(
                    "shared_mtp_canonical_validation_mismatches", 0
                ),
                "shared_mtp_canonical_fallbacks": row.get(
                    "shared_mtp_canonical_fallbacks", 0
                ),
                "mtp_shadow_commit_windows": row.get(
                    "mtp_shadow_commit_windows", 0
                ),
                "mtp_shadow_commit_tokens": row.get(
                    "mtp_shadow_commit_tokens", 0
                ),
                "mtp_shadow_commit_us": row.get("mtp_shadow_commit_us", 0),
                "hybrid_neural_windows": row.get("hybrid_neural_windows", 0),
                "hybrid_lookup_windows": row.get("hybrid_lookup_windows", 0),
                "hybrid_source_switches": row.get(
                    "hybrid_source_switches", 0
                ),
                "hybrid_lookup_miss_fallbacks": row.get(
                    "hybrid_lookup_miss_fallbacks", 0
                ),
                "hybrid_neural_rebases": row.get("hybrid_neural_rebases", 0),
                "hybrid_neural_rebase_us": row.get(
                    "hybrid_neural_rebase_us", 0
                ),
                "lookup_verify_round_us": row.get("lookup_verify_round_us", 0),
                "lookup_verify_round_us_per_window": row.get(
                    "lookup_verify_round_us_per_window"
                ),
                "lookup_verify_forward_submit_us": row.get(
                    "lookup_verify_forward_submit_us", 0
                ),
                "lookup_projection_submit_us": row.get(
                    "lookup_projection_submit_us", 0
                ),
                "lookup_verify_accept_host_sync_count": row.get(
                    "lookup_verify_accept_host_sync_count", 0
                ),
                "lookup_verify_accept_host_sync_us": row.get(
                    "lookup_verify_accept_host_sync_us", 0
                ),
                "index_entries_current_max": row["index_entries_current_max"],
                "index_entries_peak_max": row["index_entries_peak_max"],
                "process_current_usage_bytes_max": row.get(
                    "process_current_usage_bytes_max", 0
                ),
                "process_effective_ceiling_bytes_min": row.get(
                    "process_effective_ceiling_bytes_min", 0
                ),
                "mlx_peak_bytes_max": row.get("mlx_peak_bytes_max", 0),
                "server_healthy": row["server_healthy"],
                "scheduler_path_controlled": bool(
                    row.get("scheduler_path_observed")
                    and baseline.get("scheduler_path_observed")
                ),
            }
        )
    return comparisons


def evaluate_gates(
    config: MatrixConfig,
    comparisons: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    exercised = [item for item in comparisons if item.get("lookup_queries", 0) > 0]
    fallback = [item for item in comparisons if item.get("lookup_queries", 0) == 0]
    producer_cells = [
        item for item in comparisons if item.get("cross_request_role") == "producer"
    ]
    consumer_cells = [
        item for item in comparisons if item.get("cross_request_role") == "consumer"
    ]
    performance_cells = consumer_cells if config.cross_request else exercised
    correctness = all(
        item["output_match_ratio"] == 1.0
        and item["baseline_consistent_ratio"] == 1.0
        for item in exercised
    ) and bool(exercised)
    fallback_correctness = all(
        item["output_match_ratio"] == 1.0
        and item["baseline_consistent_ratio"] == 1.0
        for item in fallback
    )
    scheduler_controlled = [
        item for item in comparisons if item.get("scheduler_path_controlled", False)
    ]
    scheduler_correctness = all(
        item["output_match_ratio"] == 1.0
        and item["baseline_consistent_ratio"] == 1.0
        for item in scheduler_controlled
    )
    counter_invariants = all(
        item.get("lookup_hits", 0) <= item.get("lookup_queries", 0)
        and item.get("lookup_accepted_tokens", 0)
        <= item.get("lookup_drafted_tokens", 0)
        and item.get("lookup_rejected_tokens", 0)
        == item.get("lookup_drafted_tokens", 0)
        - item.get("lookup_accepted_tokens", 0)
        and item.get("local_hits", 0) <= item.get("local_queries", 0)
        and item.get("local_accepted_tokens", 0)
        <= item.get("local_drafted_tokens", 0)
        and item.get("local_wasted_verify_tokens", 0)
        == item.get("local_drafted_tokens", 0)
        - item.get("local_accepted_tokens", 0)
        and item.get("shared_hits", 0) <= item.get("shared_queries", 0)
        and item.get("shared_accepted_tokens", 0)
        <= item.get("shared_drafted_tokens", 0)
        and item.get("shared_wasted_verify_tokens", 0)
        == item.get("shared_drafted_tokens", 0)
        - item.get("shared_accepted_tokens", 0)
        for item in exercised
    )
    lifecycle = all(item["index_entries_current_max"] == 0 for item in exercised)
    server_health_healthy = all(item["server_healthy"] for item in exercised)
    ttft = all(
        item["ttft_change_ms"] is not None
        and item["baseline_ttft_ms"] is not None
        and item["ttft_change_ms"] <= max(5.0, item["baseline_ttft_ms"] * 0.02)
        for item in exercised
    )
    controls = [
        item
        for item in performance_cells
        if item["polarity"] in ("negative", "adversarial")
    ]
    control_regression = all(
        item["tg_change_pct"] is not None and item["tg_change_pct"] >= -3.0
        for item in controls
    )
    concurrent = [item for item in performance_cells if item["concurrency"] > 1]
    concurrent_p95 = all(
        item["e2e_p95_change_pct"] is not None and item["e2e_p95_change_pct"] <= 3.0
        for item in concurrent
    )
    category_gains = {}
    for category in ("rag", "code", "json", "long_copy"):
        values = [
            item["tg_change_pct"]
            for item in performance_cells
            if item["category"] == category and item["tg_change_pct"] is not None
        ]
        category_gains[category] = statistics.median(values) if values else None
    positive_categories = sum(
        value is not None and value >= 10.0 for value in category_gains.values()
    )
    lookup_limits = {
        lookup.name: lookup.max_index_entries for lookup in config.lookup_configs
    }
    index_bounded = all(
        item["index_entries_peak_max"] <= lookup_limits[item["lookup_name"]]
        for item in comparisons
    )
    full_coverage = (
        set(config.prompt_tokens) >= {1024, 8192, 32768, 65536}
        and set(config.max_tokens) >= {128, 512}
        and set(config.concurrency) >= {1, 2, 4, 8}
        and config.runs >= 5
        and config.balanced
    )
    expected_lookup_dimensions = {
        (prompt_tokens, max_tokens, concurrency)
        for prompt_tokens in config.prompt_tokens
        for max_tokens in config.max_tokens
        for concurrency in config.concurrency
    }
    exercised_lookup_dimensions = {
        (
            item["target_prompt_tokens"],
            item["max_tokens"],
            item["concurrency"],
        )
        for item in exercised
        if all(
            key in item
            for key in ("target_prompt_tokens", "max_tokens", "concurrency")
        )
    }
    lookup_dimension_coverage = expected_lookup_dimensions.issubset(
        exercised_lookup_dimensions
    )
    shared_consumer_path_exercised = bool(
        consumer_cells
        and any(item.get("shared_hits", 0) > 0 for item in consumer_cells)
    )
    canonical_validation_mismatch_free = all(
        item.get("shared_mtp_canonical_validation_mismatches", 0) == 0
        for item in comparisons
    )
    mtp_hybrid = "--mtp-model-dir" in config.extra_serve_args
    mtp_canonical_path_exercised = bool(
        any(
            item.get("shared_mtp_certified_hits", 0) > 0
            and item.get("shared_mtp_canonical_validation_windows", 0) > 0
            for item in consumer_cells
        )
    )
    producer_miss_itl = all(
        item.get("itl_p95_change_pct") is not None
        and item["itl_p95_change_pct"] <= 2.0
        for item in producer_cells
        if item.get("shared_hits", 0) == 0
    )
    gates = {
        "output_token_parity_100pct": correctness,
        "lookup_path_exercised": bool(exercised),
        "lookup_path_exercised_cells": len(exercised),
        "fallback_cells": len(fallback),
        "fallback_output_token_parity_100pct": (
            fallback_correctness if fallback else None
        ),
        "scheduler_path_output_token_parity_100pct": (
            scheduler_correctness if scheduler_controlled else None
        ),
        "lookup_counter_invariants_hold": counter_invariants,
        "request_local_lifecycle_clean": lifecycle,
        "server_health_healthy": server_health_healthy,
        "server_health_degraded_cells": sum(
            not item["server_healthy"] for item in exercised
        ),
        "lookup_index_within_configured_cap": index_bounded,
        "ttft_within_2pct_or_5ms": ttft,
        "negative_decode_regression_within_3pct": control_regression if controls else None,
        "concurrent_p95_e2e_within_3pct": concurrent_p95 if concurrent else None,
        "positive_category_median_tg_change_pct": category_gains,
        "positive_categories_at_least_10pct": positive_categories,
        "positive_category_gate": positive_categories >= 3,
        "cross_request_consumer_path_exercised": (
            shared_consumer_path_exercised if config.cross_request else None
        ),
        "cross_request_producer_miss_itl_within_2pct": (
            producer_miss_itl if producer_cells else None
        ),
        "mtp_canonical_path_exercised": (
            mtp_canonical_path_exercised if mtp_hybrid else None
        ),
        "mtp_canonical_validation_mismatch_free": (
            canonical_validation_mismatch_free if mtp_hybrid else None
        ),
        "full_coverage": full_coverage,
        "lookup_dimension_coverage": lookup_dimension_coverage,
    }
    hard_values = [correctness, counter_invariants, lifecycle, index_bounded, ttft]
    if controls:
        hard_values.append(control_regression)
    if concurrent:
        hard_values.append(concurrent_p95)
    hard_values.append(positive_categories >= 3)
    if config.cross_request:
        hard_values.append(shared_consumer_path_exercised)
    if mtp_hybrid:
        hard_values.extend(
            [mtp_canonical_path_exercised, canonical_validation_mismatch_free]
        )
    if full_coverage:
        hard_values.append(lookup_dimension_coverage)
    gates["status"] = (
        "pass" if full_coverage and all(hard_values) else "fail" if full_coverage else "incomplete"
    )
    return gates


def build_requests(
    resolved: ResolvedPrompt,
    concurrency: int,
    batch_idx: int,
    warmup: bool,
    cross_request: bool = False,
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    requests = []
    episode_idx = batch_idx // 2 if cross_request and not warmup else batch_idx
    cross_request_role = (
        "producer" if batch_idx % 2 == 0 else "consumer"
    ) if cross_request and not warmup else "request_local"
    seed_base = (900000 if warmup else 0) + episode_idx * 1000
    for worker_id in range(concurrency):
        request_seed = seed_base + worker_id
        prompt, expected_prefix = render_case(
            resolved.case,
            resolved.context_units,
            request_seed=request_seed,
            max_tokens=resolved.max_tokens,
        )
        requests.append(
            (
                prompt,
                expected_prefix,
                {
                    "worker_id": worker_id,
                    "request_seed": request_seed,
                    "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "cross_request_role": cross_request_role,
                },
            )
        )
    return requests


def cross_request_role(config: MatrixConfig, batch_idx: int) -> str:
    if not config.cross_request:
        return "request_local"
    return "producer" if batch_idx % 2 == 0 else "consumer"


def run_variant(
    config: MatrixConfig,
    variant: Variant,
    resolved_prompts: Sequence[ResolvedPrompt],
    tokenizer: TokenizerSidecar,
    variant_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    process = None
    log_handle = None
    checkpoint_dir = variant_dir / "batch-checkpoints"
    if checkpoint_dir.exists() and not config.resume:
        raise FileExistsError(
            "batch checkpoints already exist; pass --resume or choose a new --out-root: "
            + str(checkpoint_dir)
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows, health_rows = load_batch_checkpoints(checkpoint_dir)
    seeded_episodes = set()
    completed_batches = {
        (
            row["case_id"],
            row["target_prompt_tokens"],
            row["max_tokens"],
            row["concurrency"],
            row["batch_idx"],
        )
        for row in health_rows
    }
    try:
        process, log_handle, command = start_server(config, variant, variant_dir)
        (variant_dir / "serve-command.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )
        initial_health = fetch_json(config.host, config.port, "/healthz")
        (variant_dir / "initial-healthz.json").write_text(
            json.dumps(initial_health, indent=2), encoding="utf-8"
        )
        for resolved in resolved_prompts:
            for concurrency in config.concurrency:
                pending_batches = [
                    batch_idx
                    for batch_idx in range(
                        config.runs * (2 if config.cross_request else 1)
                    )
                    if batch_key(resolved, concurrency, batch_idx)
                    not in completed_batches
                ]
                if not pending_batches:
                    continue
                for warmup_idx in range(config.warmup_batches):
                    warmup_requests = build_requests(
                        resolved, concurrency, warmup_idx, warmup=True
                    )
                    run_batch(config, warmup_requests, resolved.max_tokens)
                    wait_idle(config)
                for batch_idx in pending_batches:
                    role = cross_request_role(config, batch_idx)
                    episode_key = (
                        resolved.case.case_id,
                        resolved.target_prompt_tokens,
                        resolved.max_tokens,
                        concurrency,
                        batch_idx // 2,
                    )
                    if (
                        variant.lookup is not None
                        and config.cross_request
                        and role == "producer"
                    ):
                        clear_shared_prompt_lookup(config)
                        wait_idle(config)
                    elif (
                        variant.lookup is not None
                        and config.cross_request
                        and role == "consumer"
                        and episode_key not in seeded_episodes
                    ):
                        producer_requests = build_requests(
                            resolved,
                            concurrency,
                            batch_idx - 1,
                            warmup=False,
                            cross_request=True,
                        )
                        run_batch(config, producer_requests, resolved.max_tokens)
                        wait_idle(config)
                        seeded_episodes.add(episode_key)
                    before = wait_idle(config)
                    requests = build_requests(
                        resolved,
                        concurrency,
                        batch_idx,
                        warmup=False,
                        cross_request=config.cross_request,
                    )
                    results, batch_wall = run_batch(
                        config, requests, resolved.max_tokens
                    )
                    total_tokens = sum(
                        int(result["completion_tokens_server"] or 0)
                        for result in results
                    )
                    aggregate_tps = total_tokens / max(batch_wall, 1e-9)
                    batch_rows = []
                    for result in results:
                        tokenized = tokenizer.tokenize(
                            result["content"], include_ids=True
                        )
                        token_ids = tokenized.get("token_ids") or []
                        token_hash = hashlib.sha256(
                            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                        result.update(
                            {
                                "variant": variant.name,
                                "round_index": variant.round_index,
                                "cache_mode": variant.cache_mode,
                                "lookup_name": (
                                    variant.lookup.name if variant.lookup else None
                                ),
                                "case_id": resolved.case.case_id,
                                "category": resolved.case.category,
                                "polarity": resolved.case.polarity,
                                "target_prompt_tokens": resolved.target_prompt_tokens,
                                "prompt_tokens_local": resolved.prompt_tokens_local,
                                "max_tokens": resolved.max_tokens,
                                "concurrency": concurrency,
                                "batch_idx": batch_idx,
                                "batch_wall_s": batch_wall,
                                "batch_aggregate_tps": aggregate_tps,
                                "output_token_count_local": int(
                                    tokenized["token_count"]
                                ),
                                "output_token_ids": token_ids,
                                "output_token_hash": token_hash,
                                "output_text_hash": hashlib.sha256(
                                    result["content"].encode("utf-8")
                                ).hexdigest(),
                                "expected_prefix_match": result["content"]
                                .lstrip()
                                .startswith(result["expected_prefix"] or ""),
                            }
                        )
                        batch_rows.append(result)
                    after = wait_idle(config)
                    health_row = {
                        "variant": variant.name,
                        "round_index": variant.round_index,
                        "case_id": resolved.case.case_id,
                        "target_prompt_tokens": resolved.target_prompt_tokens,
                        "max_tokens": resolved.max_tokens,
                        "concurrency": concurrency,
                        "batch_idx": batch_idx,
                        "cross_request_role": requests[0][2][
                            "cross_request_role"
                        ],
                        "delta": health_delta(before, after),
                    }
                    write_batch_checkpoint(
                        checkpoint_dir,
                        resolved,
                        concurrency,
                        batch_idx,
                        batch_rows,
                        health_row,
                    )
                    rows.extend(batch_rows)
                    health_rows.append(health_row)
                    completed_batches.add(batch_key(resolved, concurrency, batch_idx))
                    if role == "producer":
                        seeded_episodes.add(episode_key)
        final_health = wait_idle(config)
        (variant_dir / "final-healthz.json").write_text(
            json.dumps(final_health, indent=2), encoding="utf-8"
        )
        return rows, health_rows
    finally:
        if process is not None:
            terminate_process(process)
        if log_handle is not None:
            log_handle.close()


def batch_key(
    resolved: ResolvedPrompt, concurrency: int, batch_idx: int
) -> Tuple[str, int, int, int, int]:
    return (
        resolved.case.case_id,
        resolved.target_prompt_tokens,
        resolved.max_tokens,
        concurrency,
        batch_idx,
    )


def batch_checkpoint_path(
    checkpoint_dir: Path,
    resolved: ResolvedPrompt,
    concurrency: int,
    batch_idx: int,
) -> Path:
    return checkpoint_dir / (
        "{}-p{}-m{}-b{}-r{}.json".format(
            resolved.case.case_id,
            resolved.target_prompt_tokens,
            resolved.max_tokens,
            concurrency,
            batch_idx,
        )
    )


def write_batch_checkpoint(
    checkpoint_dir: Path,
    resolved: ResolvedPrompt,
    concurrency: int,
    batch_idx: int,
    rows: Sequence[Dict[str, Any]],
    health_row: Dict[str, Any],
) -> None:
    path = batch_checkpoint_path(checkpoint_dir, resolved, concurrency, batch_idx)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"rows": list(rows), "health": health_row}), encoding="utf-8"
    )
    temporary.replace(path)


def load_batch_checkpoints(
    checkpoint_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    health_rows = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(checkpoint["rows"])
        health_rows.append(checkpoint["health"])
    return rows, health_rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return ("{:.%df}" % digits).format(value)
    return str(value)


def render_markdown(
    config: MatrixConfig,
    comparisons: Sequence[Dict[str, Any]],
    gates: Dict[str, Any],
) -> str:
    lines = [
        "# PromptLookup Production Qualification",
        "",
        "- Model: `{}`".format(config.model_dir),
        "- Prompt targets: `{}`".format(", ".join(map(str, config.prompt_tokens))),
        "- Generation targets: `{}`".format(", ".join(map(str, config.max_tokens))),
        "- Concurrency: `{}`".format(", ".join(map(str, config.concurrency))),
        "- Server max sequences: `{}`".format(
            config.max_sequences or max(config.concurrency)
        ),
        "- Server prefill chunk size: `{}`".format(
            config.prefill_chunk_size
        ),
        "- Runs per cell: `{}`".format(config.runs),
        "- Balanced server order: `{}`".format(config.balanced),
        "- Gate status: **{}**".format(gates["status"]),
        "",
        "## Comparisons",
        "",
        "| lookup | cache | route | case | pp | tg | B | parity | TG change | E2E change | TTFT change | acceptance | rollbacks |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            "| {lookup_name} | {cache_mode} | {route} | {case_id} | {target_prompt_tokens} | "
            "{max_tokens} | {concurrency} | {parity} | {tg} | {e2e} | {ttft} | "
            "{acceptance} | {rollbacks} |".format(
                parity=format_number(row["output_match_ratio"] * 100.0, 1) + "%",
                tg=format_number(row["tg_change_pct"], 1) + "%",
                e2e=format_number(row["e2e_change_pct"], 1) + "%",
                ttft=format_number(row["ttft_change_pct"], 1) + "%",
                acceptance=(
                    format_number(row["lookup_acceptance_ratio"] * 100.0, 1) + "%"
                    if row["lookup_acceptance_ratio"] is not None
                    else "n/a"
                ),
                rollbacks=row["lookup_rollbacks"],
                route=(
                    "scheduler-b1"
                    if row["scheduler_path_controlled"]
                    else "feature-toggle"
                ),
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "```json",
            json.dumps(gates, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_binaries(config: MatrixConfig) -> None:
    env = os.environ.copy()
    env.setdefault("MLX_DIR", str(config.mlx_dir))
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "ironmlx",
            "-p",
            "iron-bench",
        ],
        cwd=str(config.root),
        env=env,
        check=True,
    )


def write_run_plan(
    config: MatrixConfig,
    variants: Sequence[Variant],
) -> None:
    lines = ["#!/bin/sh", "set -eu", ""]
    for variant in variants:
        variant_dir = config.out_root / "round-{:02d}-{}".format(
            variant.round_index, variant.name
        )
        lines.append("# round {} {}".format(variant.round_index, variant.name))
        lines.append(shlex.join(build_serve_command(config, variant, variant_dir)))
        lines.append("")
    path = config.out_root / "server-plan.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def run_matrix(config: MatrixConfig) -> int:
    config.out_root.mkdir(parents=True, exist_ok=True)
    cases = load_corpus(config.corpus_path)
    if config.categories:
        requested = set(config.categories)
        cases = [case for case in cases if case.category in requested]
        missing = requested - {case.category for case in cases}
        if missing:
            raise ValueError("unknown or empty corpus categories: {}".format(sorted(missing)))
    variants = build_variants(config)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            **asdict(config),
            "root": str(config.root),
            "model_dir": str(config.model_dir),
            "out_root": str(config.out_root),
            "serve_bin": str(config.serve_bin),
            "tokenizer_bin": str(config.tokenizer_bin),
            "corpus_path": str(config.corpus_path),
            "mlx_dir": str(config.mlx_dir),
        },
        "variants": [asdict(variant) for variant in variants],
        "cases": [asdict(case) for case in cases],
    }
    (config.out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    if config.dry_run:
        write_run_plan(config, variants)
        print("dry-run wrote {}".format(config.out_root))
        return 0
    if config.build:
        build_binaries(config)
    if not config.serve_bin.is_file():
        raise FileNotFoundError("serve binary not found: {}".format(config.serve_bin))
    if not config.tokenizer_bin.is_file():
        raise FileNotFoundError(
            "tokenizer binary not found: {}".format(config.tokenizer_bin)
        )

    all_rows = []
    all_health = []
    errors = []
    with TokenizerSidecar(config.tokenizer_bin, config.model_dir) as tokenizer:
        resolved = [
            resolve_prompt(case, prompt_tokens, max_tokens, tokenizer.count)
            for case in cases
            for prompt_tokens in config.prompt_tokens
            for max_tokens in config.max_tokens
        ]
        if config.max_cache_cap is None:
            config.max_cache_cap = max(
                item.prompt_tokens_local + item.max_tokens for item in resolved
            ) + 1024
            metadata["config"]["max_cache_cap"] = config.max_cache_cap
            (config.out_root / "metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        write_run_plan(config, variants)
        (config.out_root / "resolved-corpus.json").write_text(
            json.dumps(
                [
                    {
                        **asdict(item),
                        "case": asdict(item.case),
                    }
                    for item in resolved
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        for sequence, variant in enumerate(variants):
            variant_dir = config.out_root / "sequence-{:02d}-round-{:02d}-{}".format(
                sequence, variant.round_index, variant.name
            )
            variant_dir.mkdir(parents=True, exist_ok=True)
            print(
                "[{}] starting {} ({}/{})".format(
                    datetime.now().isoformat(timespec="seconds"),
                    variant.name,
                    sequence + 1,
                    len(variants),
                ),
                file=sys.stderr,
            )
            try:
                rows, health_rows = run_variant(
                    config, variant, resolved, tokenizer, variant_dir
                )
                all_rows.extend(rows)
                all_health.extend(health_rows)
            except Exception as error:
                failure = {
                    "variant": variant.name,
                    "round_index": variant.round_index,
                    "error": str(error),
                }
                errors.append(failure)
                (variant_dir / "error.txt").write_text(str(error), encoding="utf-8")
                checkpoint_rows, checkpoint_health = load_batch_checkpoints(
                    variant_dir / "batch-checkpoints"
                )
                all_rows.extend(checkpoint_rows)
                all_health.extend(checkpoint_health)
                print("ERROR {}: {}".format(variant.name, error), file=sys.stderr)
                if not config.allow_failures:
                    break

    attach_output_parity(all_rows)
    summary = aggregate_rows(all_rows, all_health)
    comparisons = build_comparisons(config, summary)
    gates = evaluate_gates(config, comparisons)
    write_jsonl(config.out_root / "raw-results.jsonl", all_rows)
    write_jsonl(config.out_root / "cell-health.jsonl", all_health)
    (config.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(config.out_root / "summary.csv", summary)
    (config.out_root / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    write_csv(config.out_root / "comparisons.csv", comparisons)
    (config.out_root / "gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )
    (config.out_root / "summary.md").write_text(
        render_markdown(config, comparisons, gates), encoding="utf-8"
    )
    if errors:
        (config.out_root / "errors.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )
    print("wrote {}".format(config.out_root))
    return 1 if errors and not config.allow_failures else 0


def parse_int_list(raw: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("list values must be positive integers")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> MatrixConfig:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--serve-bin", type=Path, default=root / "target/release/ironmlx")
    parser.add_argument(
        "--tokenizer-bin",
        type=Path,
        default=root / "target/release/iron-bench-tokenizer",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--prompt-tokens", type=parse_int_list)
    parser.add_argument("--max-tokens", type=parse_int_list)
    parser.add_argument("--concurrency", type=parse_int_list)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument(
        "--max-cache-cap",
        type=int,
        help="server cache cap; defaults to the resolved prompt length plus generation and headroom",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=2048,
        help="server prefill chunk size; scheduler routing is verified from health counters",
    )
    parser.add_argument("--runs", type=int)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--lookup-config",
        action="append",
        type=LookupConfig.parse,
        dest="lookup_configs",
    )
    parser.add_argument(
        "--cross-request",
        action="store_true",
        help="enable trust-domain-local reuse of completed request histories",
    )
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--include-prefix-cache", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19120)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--mlx-dir", type=Path, default=Path.home() / ".local/mlx")
    parser.add_argument("--rust-log", default="info")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse atomic batch checkpoints under an existing --out-root",
    )
    parser.add_argument("--extra-serve-arg", action="append", default=[])
    args = parser.parse_args(argv)

    if args.profile == "full":
        prompt_tokens = args.prompt_tokens or (1024, 8192, 32768, 65536)
        max_tokens = args.max_tokens or (128, 512)
        concurrency = args.concurrency or (1, 2, 4, 8)
        runs = args.runs if args.runs is not None else 5
        balanced = True
    else:
        prompt_tokens = args.prompt_tokens or (1024, 8192)
        max_tokens = args.max_tokens or (128,)
        concurrency = args.concurrency or (1, 2)
        runs = args.runs if args.runs is not None else 3
        balanced = args.balanced
    if runs <= 0 or args.warmup_batches < 0:
        parser.error("--runs must be positive and --warmup-batches must be non-negative")
    if args.max_sequences is not None and args.max_sequences <= 0:
        parser.error("--max-sequences must be positive")
    if args.max_cache_cap is not None and args.max_cache_cap <= 0:
        parser.error("--max-cache-cap must be positive")
    if args.prefill_chunk_size < 0:
        parser.error("--prefill-chunk-size must be non-negative")
    out_root = args.out_root or (
        Path("/tmp")
        / "ironmlx-prompt-lookup-matrix"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return MatrixConfig(
        root=root,
        model_dir=args.model_dir.resolve(),
        out_root=out_root.resolve(),
        serve_bin=args.serve_bin.resolve(),
        tokenizer_bin=args.tokenizer_bin.resolve(),
        corpus_path=args.corpus.resolve(),
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        prompt_tokens=tuple(prompt_tokens),
        max_tokens=tuple(max_tokens),
        concurrency=tuple(concurrency),
        max_sequences=args.max_sequences,
        max_cache_cap=args.max_cache_cap,
        prefill_chunk_size=args.prefill_chunk_size,
        runs=runs,
        warmup_batches=args.warmup_batches,
        lookup_configs=tuple(args.lookup_configs or (LookupConfig(name="default"),)),
        cross_request=args.cross_request,
        include_prefix_cache=args.include_prefix_cache,
        balanced=balanced,
        categories=tuple(args.category),
        timeout_secs=args.timeout,
        startup_timeout_secs=args.startup_timeout,
        mlx_dir=args.mlx_dir.resolve(),
        rust_log=args.rust_log,
        build=args.build,
        dry_run=args.dry_run,
        allow_failures=args.allow_failures,
        resume=args.resume,
        extra_serve_args=tuple(args.extra_serve_arg),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_matrix(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
