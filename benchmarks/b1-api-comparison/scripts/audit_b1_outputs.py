#!/usr/bin/env python3
"""Execute reviewed benchmark code in temporary fixtures; keep every failure.

This is a task correctness harness, NOT a security sandbox. Review generated
code before invoking --reviewed. Knowledge/factual review remains manual.
"""

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

INTERVAL_TEST = """
import random
def reference(items):
    points=set()
    # Half-unit grid distinguishes adjacent disjoint integer closed intervals.
    for a,b in items: points.update(range(2*a,2*b+1))
    result=[]
    for p in sorted(points):
        if result and p==result[-1][1]+1: result[-1][1]=p
        else: result.append([p,p])
    return [[a/2,b/2] for a,b in result]
rng=random.Random(20260929)
cases=[[],[[2,3]],[[3,4],[1,3]],[[1,9],[2,3]],[[1,2],[3,4]],[[0,0],[0,0]]]
for _ in range(100):
    cases.append([sorted([rng.randrange(-9,10),rng.randrange(-9,10)]) for _ in range(rng.randrange(20))])
for items in cases: assert merge_intervals([x[:] for x in items])==reference(items),items
print('interval: 106 cases pass')
"""
ASYNC_TEST = """
import asyncio
async def audit():
    active=peak=0
    async def fake(i):
        nonlocal active,peak
        active+=1;peak=max(peak,active)
        try:
            await asyncio.sleep((10-i%10)*0.001)
            return i*7
        finally: active-=1
    globals()['fetch']=fake
    assert await fetch_all([])==[]
    assert await fetch_all(list(range(17)))==[i*7 for i in range(17)]
    assert 1<peak<=4,peak
    class Expected(Exception): pass
    async def bad(i):
        await asyncio.sleep(0.001)
        if i==2: raise Expected('probe')
        return i
    globals()['fetch']=bad
    try: await fetch_all(list(range(9)))
    except Expected: pass
    except ExceptionGroup as exc: assert any(isinstance(e,Expected) for e in exc.exceptions)
    else: raise AssertionError('exception not propagated')
    await asyncio.sleep(0.03)
asyncio.run(audit())
print('async: order, limit, empty input, error propagation pass')
"""
LRU_TEST = """
const check=(x: boolean)=>{if(!x)throw new Error('LRU assertion failed');};
const c=new LRUCache<string,number>(2);
c.put('a',1);c.put('b',2);check(c.get('a')===1);
c.put('c',3);check(c.get('b')==null || c.get('b')===-1);
check(c.get('c')===3);c.put('a',0);check(c.get('a')===0);
c.put('d',4);check(c.get('c')==null || c.get('c')===-1);
check(c.get('a')===0);check(c.get('d')===4);
const one=new LRUCache<object,number>(1);const a={},b={};
one.put(a,11);one.put(a,12);check(one.get(a)===12);
one.put(b,13);check(one.get(a)==null || one.get(a)===-1);check(one.get(b)===13);
console.log('LRU: access/update eviction, capacity-one, object keys, zero value pass');
"""


def code_source(row):
    blocks = re.findall(
        r"```(?:python|py|typescript|ts)?\s*\n(.*?)```", row["output"], re.S
    )
    symbol = {
        "code-1": "merge_intervals",
        "code-2": "class LRUCache",
        "code-3": "def fetch_all",
    }[row["id"]]
    code = next(b for b in blocks if symbol in b)
    if row["id"] == "code-2":
        return code + "\n" + LRU_TEST, ".ts"
    tree = ast.parse(code)
    tree.body = [
        n
        for n in tree.body
        if isinstance(
            n,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    ]
    return ast.unparse(tree) + "\n" + (
        INTERVAL_TEST if row["id"] == "code-1" else ASYNC_TEST
    ), ".py"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--reviewed", action="store_true", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("preserve prior audit: choose a new output")
    results = []
    seen = set()
    for file in args.files:
        data = json.loads(file.read_text())
        for row in data["runs"]:
            if row["category"] != "code":
                continue
            key = (data["label"], row["id"], row["output_sha256"])
            if key in seen:
                continue
            seen.add(key)
            result = dict(file=str(file), id=row["id"], sha256=row["output_sha256"])
            try:
                code, suffix = code_source(row)
                with tempfile.TemporaryDirectory(
                    prefix="b1-output-audit-"
                ) as directory:
                    script = Path(directory) / ("fixture" + suffix)
                    script.write_text(code)
                    command = (
                        [sys.executable, str(script)]
                        if suffix == ".py"
                        else ["node", "--experimental-transform-types", str(script)]
                    )
                    proc = subprocess.run(
                        command,
                        cwd=directory,
                        text=True,
                        capture_output=True,
                        timeout=15,
                    )
                    result.update(
                        passed=proc.returncode == 0,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                    )
            except Exception as exc:
                result.update(passed=False, error=repr(exc))
            results.append(result)
    report = dict(
        code_pass=all(r["passed"] for r in results) and bool(results),
        results=results,
        knowledge_review="required separately",
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
