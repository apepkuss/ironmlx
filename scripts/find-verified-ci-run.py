#!/usr/bin/env python3
"""Find an earlier passing CI push run that tested exactly the same source tree.

A squash merge onto dev, or a fast-forward onto main, often produces a new
commit whose tree is identical to a branch tip that already passed the native
jobs. Re-running 80 minutes of identical work adds no evidence, so the
workflow skips the native jobs when this script finds such a run.

Only push runs count: a pull_request run tests a merge commit whose tree is not
the reported head commit's tree. Every required job must have succeeded in that
run; a run that itself skipped the native jobs never qualifies. Any API error
yields no match, so the full checks run.

Prints GitHub step outputs: verified_run_id and verified_run_url (empty when
there is no match).
"""

import argparse
import json
import subprocess
import sys


def gh_api(path):
    result = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def select_run(runs, tree, required_jobs, fetch_jobs, current_run_id=None):
    """Return the newest qualifying run, or None."""
    for run in runs:
        if run.get("id") == current_run_id:
            continue
        if run.get("event") != "push" or run.get("conclusion") != "success":
            continue
        if (run.get("head_commit") or {}).get("tree_id") != tree:
            continue
        conclusions = {job.get("name"): job.get("conclusion") for job in fetch_jobs(run["id"])}
        if all(conclusions.get(name) == "success" for name in required_jobs):
            return run
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow", required=True, help="workflow file name, e.g. ci.yml")
    parser.add_argument("--tree", required=True, help="tree SHA of the commit under test")
    parser.add_argument("--current-run-id", type=int)
    parser.add_argument("--required-job", action="append", required=True, dest="required_jobs")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    run = None
    try:
        listing = gh_api(
            f"repos/{args.repository}/actions/workflows/{args.workflow}/runs"
            f"?event=push&status=success&per_page={args.limit}"
        )

        def fetch_jobs(run_id):
            return gh_api(f"repos/{args.repository}/actions/runs/{run_id}/jobs?per_page=100")["jobs"]

        run = select_run(
            listing.get("workflow_runs", []),
            args.tree,
            args.required_jobs,
            fetch_jobs,
            args.current_run_id,
        )
    except (subprocess.CalledProcessError, KeyError, ValueError) as error:
        detail = getattr(error, "stderr", "") or error
        print(f"warning: cannot look up earlier CI runs; running full checks: {detail}", file=sys.stderr)

    if run is None:
        print(f"No earlier passing push run tested tree {args.tree}", file=sys.stderr)
        print("verified_run_id=")
        print("verified_run_url=")
    else:
        print(
            f"Tree {args.tree} already passed in run {run['id']} "
            f"({run.get('head_branch')} {run.get('head_sha')}): {run.get('html_url')}",
            file=sys.stderr,
        )
        print(f"verified_run_id={run['id']}")
        print(f"verified_run_url={run.get('html_url', '')}")


if __name__ == "__main__":
    main()
