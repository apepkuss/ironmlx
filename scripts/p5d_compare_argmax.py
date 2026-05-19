#!/usr/bin/env python3
"""Compare ironmlx vs omlx greedy outputs per prompt. Exit 0 if all match."""
import json
import sys
import os

repo_root = os.path.join(os.path.dirname(__file__), "..")
ironmlx_path = os.path.join(repo_root, "reports/p5d-argmax/ironmlx.jsonl")
omlx_path = os.path.join(repo_root, "reports/p5d-argmax/omlx.jsonl")

ironmlx = [json.loads(l) for l in open(ironmlx_path)]
omlx = [json.loads(l) for l in open(omlx_path)]
assert len(ironmlx) == len(omlx), (
    f"length mismatch: ironmlx {len(ironmlx)} vs omlx {len(omlx)}"
)

mismatches = []
errors = []
for a, b in zip(ironmlx, omlx):
    assert a["idx"] == b["idx"], f"idx mismatch: {a['idx']} vs {b['idx']}"
    ao, bo = a["output"], b["output"]
    if ao == "<ERROR>" or bo == "<ERROR>":
        errors.append({
            "idx": a["idx"],
            "ironmlx_output": ao,
            "omlx_output": bo,
        })
        continue
    if ao != bo:
        diff_at = next(
            (i for i, (x, y) in enumerate(zip(ao, bo)) if x != y),
            min(len(ao), len(bo)),
        )
        mismatches.append({
            "idx": a["idx"],
            "prompt_preview": a["prompt"][:60] + "...",
            "diff_at": diff_at,
            "ironmlx_len": len(ao),
            "omlx_len": len(bo),
            "ironmlx_excerpt": ao[max(0, diff_at - 10):diff_at + 30],
            "omlx_excerpt": bo[max(0, diff_at - 10):diff_at + 30],
        })

print()
print("P5d T3 Cross-Prompt Greedy Alignment")
print("=" * 50)
print(f"Total prompts:  {len(ironmlx)}")
print(f"Errors (any):   {len(errors)}")
print(f"Identical:      {len(ironmlx) - len(mismatches) - len(errors)}")
print(f"Mismatched:     {len(mismatches)}")
print()

if errors:
    print(f"Prompts with <ERROR> responses (first 3):")
    for e in errors[:3]:
        print(f"  [{e['idx']}] ironmlx={e['ironmlx_output']!r}  omlx={e['omlx_output']!r}")
    print()

if mismatches:
    print(f"First 5 mismatches:")
    for m in mismatches[:5]:
        print(f"\n  [{m['idx']}] {m['prompt_preview']}")
        print(f"    diff at char {m['diff_at']} "
              f"(ironmlx_len={m['ironmlx_len']}, omlx_len={m['omlx_len']}):")
        print(f"      ironmlx: ...{m['ironmlx_excerpt']!r}")
        print(f"      omlx:    ...{m['omlx_excerpt']!r}")
    print()
    total_bad = len(mismatches) + len(errors)
    sys.exit(1)

if errors:
    print("FAIL: some prompts returned <ERROR>")
    sys.exit(1)

print(f"All {len(ironmlx)} prompts: ironmlx greedy output coincides with the "
      f"external reference recording (observational triangulation, NOT an "
      f"alignment gate — ironmlx is an independent implementation).")
sys.exit(0)
