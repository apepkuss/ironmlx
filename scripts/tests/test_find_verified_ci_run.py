"""Skip native CI only for a tree that already passed every native job."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "find-verified-ci-run.py"
spec = importlib.util.spec_from_file_location("find_verified_ci_run", SCRIPT)
finder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finder)

TREE = "a" * 40
REQUIRED = ["Native A", "Native B"]


def run(run_id, tree=TREE, event="push", conclusion="success"):
    return {"id": run_id, "event": event, "conclusion": conclusion,
            "head_commit": {"tree_id": tree}}


class SelectRunTests(unittest.TestCase):
    def select(self, runs, jobs, current=None):
        def fetch(run_id):
            return [{"name": n, "conclusion": c} for n, c in jobs[run_id].items()]
        return finder.select_run(runs, TREE, REQUIRED, fetch, current)

    def test_same_tree_with_all_native_jobs_passed_is_verified(self):
        jobs = {1: {"Native A": "success", "Native B": "success", "Docs": "skipped"}}
        self.assertEqual(self.select([run(1)], jobs)["id"], 1)

    def test_skipped_or_missing_native_job_does_not_qualify(self):
        jobs = {2: {"Native A": "success", "Native B": "skipped"},
                3: {"Native A": "success"},
                4: {"Native A": "success", "Native B": "success"}}
        self.assertEqual(self.select([run(2), run(3), run(4)], jobs)["id"], 4)
        self.assertIsNone(self.select([run(2), run(3)], jobs))

    def test_other_trees_pull_requests_and_current_run_are_ignored(self):
        ok = {"Native A": "success", "Native B": "success"}
        runs = [run(5, tree="b" * 40), run(6, event="pull_request"),
                run(7, conclusion="failure"), run(8)]
        self.assertIsNone(self.select(runs, {8: ok}, current=8))


if __name__ == "__main__":
    unittest.main()
