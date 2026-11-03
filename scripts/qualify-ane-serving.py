#!/usr/bin/env python3
"""Opt-in qualification of an isolated ANE HTTP server on --port (default 19086).

Start ironmlx serve with --model-id ane-qualification --max-sequences 2
--prefill-chunk-size 2048 --max-cache-cap 12288, without a prefix cache.
Choose the model/draft and --ane auto/off before running this script. This
checks text/images, B1/B2, multiple chunks and cancellation before first token.
The image fixture and health counters are required; GPU fallback is not ANE
execution. --expect-fallback is reserved for an injected native-worker error.
"""

import argparse
import base64
import concurrent.futures
import http.client
import json
import pathlib
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=19086)
p.add_argument("--output", type=pathlib.Path, required=True)
p.add_argument("--model", default="ane-qualification")
p.add_argument("--family", choices=["dense", "flash"], required=True)
p.add_argument("--mode", choices=["auto", "off"], required=True)
p.add_argument("--expect-fallback", action="store_true")
a = p.parse_args()
base = f"http://127.0.0.1:{a.port}"


def health():
    with urllib.request.urlopen(base + "/healthz", timeout=20) as r:
        return json.load(r)


def jobs():
    return next(
        (
            v["completed_jobs"]
            for v in health()["ane"]["models"]
            if v["model"]
            == ("qwen38-27b" if a.family == "dense" else "qwen38-flash-next")
        )
    )


text = "The quick brown fox jumps over the lazy dog. " * 255
image = base64.b64encode(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "ironmlx-lm/tests/fixtures/embedding_gemma2/images/red-circle.png"
    ).read_bytes()
).decode()


def prompt(i, vision=False):
    t = (
        f"Case {i}. "
        + text
        + (
            " State the image shape and color in one short sentence; ignore the repeated text above."
            if vision
            else f" Answer with one short sentence about the number {i}."
        )
    )
    return (
        [
            dict(
                type="image_url", image_url=dict(url="data:image/png;base64," + image)
            ),
            dict(type="text", text=t),
        ]
        if vision
        else t
    )


prompts = [prompt(1), prompt(2, True)]


def payload(content, stream=False):
    return dict(
        model=a.model,
        messages=[dict(role="user", content=content)],
        temperature=0,
        max_tokens=12,
        stream=stream,
        chat_template_kwargs=dict(enable_thinking=False),
    )


def complete(content):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload(content)).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()) from e
    return dict(
        message=d["choices"][0]["message"],
        finish_reason=d["choices"][0]["finish_reason"],
        usage=d["usage"],
        seconds=time.monotonic() - t,
    )


def comp(x):
    return dict(
        message=x["message"],
        finish_reason=x["finish_reason"],
        completion_tokens=x["usage"]["completion_tokens"],
    )


before = jobs()
b1 = []
b1_jobs = []
for content in prompts:
    n = jobs()
    b1.append(complete(content))
    b1_jobs.append(jobs() - n)
after_b1 = jobs()
print("b1", b1_jobs, flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as ex:
    b2 = list(ex.map(complete, prompts))
after_b2 = jobs()
print("b2", after_b2 - after_b1, flush=True)
long_result = complete(prompt(77).replace(text, text * 3))
long_jobs = jobs() - after_b2
print("multi_chunk", long_jobs, flush=True)
c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
c.request(
    "POST",
    "/v1/chat/completions",
    json.dumps(payload(prompt(91), True)),
    {"Content-Type": "application/json"},
)
r = c.getresponse()
assert r.status == 200
r.close()
c.close()
recovered = complete(prompt(92))
deadline = time.monotonic() + 40
while True:
    h = health()
    s = h["scheduler"]
    if s["b_active"] == 0 and s["b_queued"] == 0:
        break
    assert time.monotonic() < deadline, "cancelled scheduler capacity not released"
    time.sleep(0.2)
result = dict(
    family=a.family,
    mode=a.mode,
    b1=b1,
    b2=b2,
    cancel_recovered=recovered,
    health=h,
    jobs_before=before,
    jobs_after_b1=after_b1,
    b1_jobs=b1_jobs,
    b2_jobs=after_b2 - after_b1,
    multi_chunk_jobs=long_jobs,
    multi_chunk=long_result,
    b1_b2_equal=[comp(x) == comp(y) for (x, y) in zip(b1, b2)],
)
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
assert all(result["b1_b2_equal"]), "B1/B2 results differ"
if a.family == "dense" and a.mode == "auto" and (not a.expect_fallback):
    assert all((n > 0 for n in b1_jobs)), (
        "long text or image prefill did not execute ANE"
    )
    assert after_b2 > after_b1, "concurrent requests did not execute ANE"
    assert long_jobs >= 192, "multiple full chunks did not execute ANE"
else:
    assert jobs() == 0, "off or unadapted Flash executed ANE"
family_state = next(
    (
        x
        for x in h["ane"]["models"]
        if x["model"] == ("qwen38-27b" if a.family == "dense" else "qwen38-flash-next")
    )
)
expected = (
    "runtime_failed"
    if a.expect_fallback
    else "disabled"
    if a.mode == "off"
    else "eligible"
    if a.family == "dense"
    else "model_not_adapted"
)
assert family_state["reason"] == expected, (family_state, expected)
assert h["status"] == "healthy"
print("PASS", a.output, flush=True)
