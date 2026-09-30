#!/usr/bin/env python3
"""Audit saved B1 evidence without launching inference or filtering slow rows.

This checks evidence integrity, not factual answer quality or thermal stability.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

APPS = ("ironmlx", "omlx", "splash", "tensorfold")
PROMPTS = {f"{kind}-{i}" for kind in ("code", "knowledge") for i in range(1, 4)}


def audit(directory, label, sessions):
    issues = []
    apps = {}
    identities = defaultdict(set)
    inputs = defaultdict(set)
    for app in APPS:
        outputs = defaultdict(set)
        first_chunks = set()
        starts = []
        thermals = set()
        hot_processes = []
        measured = 0
        for sid in range(sessions):
            prefix = directory / f"{label}-{app}-s{sid}"
            data = json.loads(prefix.with_suffix(".json").read_text())
            meta = json.loads(prefix.with_suffix(".command.json").read_text())
            starts.append(data["started_utc"])
            if data["session"] != sid or meta["session"] != sid or meta["app"] != app:
                issues.append(f"{app}/{sid}: session identity mismatch")
            identities["prompts_sha256"].add(data["prompts_sha256"])
            for field in ("benchmark_client", "benchmark_runner", "protocol"):
                identities[field].add(meta[field]["sha256"])
            identities[f"{app}_entrypoint"].add(meta["entrypoint"]["sha256"])
            for phase in ("before", "after"):
                thermals.add(data[f"thermal_{phase}"])
                for line in data[f"processes_{phase}"].splitlines()[1:]:
                    parts = line.strip().split(None, 2)
                    if len(parts) == 3 and float(parts[1]) > 20:
                        hot_processes.append(
                            dict(session=sid, phase=phase, process=line.strip())
                        )
            rows = [r for r in data["runs"] if r["category"] != "warmup"]
            if len(rows) != 6 or {r["id"] for r in rows} != PROMPTS:
                issues.append(f"{app}/{sid}: incomplete prompt set")
            for row in rows:
                measured += 1
                if (
                    not row["valid"]
                    or row["finish_reason"] != "stop"
                    or row["reasoning"]
                ):
                    issues.append(
                        f"{app}/{sid}/{row['id']}: invalid or nonnatural output"
                    )
                digest = hashlib.sha256(row["output"].encode()).hexdigest()
                if digest != row["output_sha256"]:
                    issues.append(f"{app}/{sid}/{row['id']}: answer hash mismatch")
                outputs[row["id"]].add(digest)
                inputs[row["id"]].add(row["usage"]["prompt_tokens"])
                first_chunks.add(row["first_chunk_tokens"])
            if app == "splash":
                log = prefix.with_suffix(".server.log").read_text()
                cached = re.findall(r"· cached ([\d,]+) ·", log)
                if len(cached) != 8 or any(int(c.replace(",", "")) for c in cached):
                    issues.append(f"splash/{sid}: missing or nonzero cache accounting")
        apps[app] = dict(
            measured_requests=measured,
            session_start_utc=starts,
            output_variants={p: len(hashes) for p, hashes in sorted(outputs.items())},
            first_chunk_tokens=sorted(first_chunks),
            thermal_snapshots=sorted(thermals),
            processes_above_20_percent=hot_processes,
        )
    for name, hashes in identities.items():
        if len(hashes) != 1:
            issues.append(f"{name}: identity changed")
    return dict(
        integrity_pass=not issues,
        issues=issues,
        apps=apps,
        identities={k: sorted(v) for k, v in identities.items()},
        input_token_counts={k: sorted(v) for k, v in sorted(inputs.items())},
        note="No samples excluded. Process snapshots are not continuous telemetry. "
        "This audit does not certify numerical correctness, factual quality or no throttling.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve prior audit: choose a new output")
    if args.sessions < 1:
        parser.error("sessions must be positive")
    result = audit(args.directory, args.label, args.sessions)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(integrity_pass=result["integrity_pass"], issues=result["issues"])
        )
    )
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
