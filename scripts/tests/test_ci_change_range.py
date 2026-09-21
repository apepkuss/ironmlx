"""CI must not skip code gates when a rewritten push base is unavailable."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "resolve-ci-change-range.sh"
ZERO = "0" * 40
MISSING = "f" * 40


class ChangeRangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "CI Test")
        self.git("config", "user.email", "ci@example.invalid")
        self.base = self.commit("README.md", "base", "docs: initial content")
        self.git("update-ref", "refs/remotes/origin/main", self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def commit(self, path, text, subject):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        self.git("add", "--", path)
        self.git("commit", "-qm", subject)
        return self.git("rev-parse", "HEAD")

    def run_script(self, base, head, env=None):
        return subprocess.run([str(SCRIPT), base, head], cwd=self.repo,
                              capture_output=True, text=True, env=env)

    def resolve(self, base, head):
        result = self.run_script(base, head)
        self.assertEqual(result.returncode, 0, result.stderr)
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_normal_docs_change_stays_docs_only(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        result = self.resolve(self.base, head)
        self.assertEqual(result["base_sha"], self.base)
        self.assertEqual(result["docs_only"], "true")
        self.assertEqual(result["code_changed"], "false")

    def test_code_change_runs_full_quality(self):
        head = self.commit("src/lib.rs", "fn main() {}", "feat: add source")
        result = self.resolve(self.base, head)
        self.assertEqual(result["code_changed"], "true")
        self.assertEqual(result["docs_only"], "false")

    def test_missing_old_tip_forces_full_quality_even_for_docs(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        self.assertNotEqual(subprocess.run(["git", "cat-file", "-e", MISSING],
                                          cwd=self.repo, capture_output=True).returncode, 0)
        result = self.resolve(MISSING, head)
        self.assertEqual(result["base_sha"], self.base)
        self.assertEqual(result["head_sha"], head)
        self.assertEqual(result["code_changed"], "true")
        self.assertEqual(result["docs_only"], "false")

    def test_new_branch_uses_reachable_ancestor(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        result = self.resolve(ZERO, head)
        self.assertEqual(result["base_sha"], self.base)
        self.assertEqual(result["code_changed"], "true")

    def test_closest_mainline_ancestor_is_used(self):
        dev = self.commit("src/lib.rs", "code", "feat: add source")
        self.git("update-ref", "refs/remotes/origin/dev", dev)
        head = self.commit("docs/api.md", "API", "docs: add API")
        self.assertEqual(self.resolve(MISSING, head)["base_sha"], dev)

    def test_missing_mainline_fails_without_classification(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        self.git("update-ref", "-d", "refs/remotes/origin/main")
        result = self.run_script(MISSING, head)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_fallback_cannot_validate_an_empty_range(self):
        result = self.run_script(MISSING, self.base)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_bad_identifiers_and_missing_head_fail_closed(self):
        for base, head in [("HEAD", self.base), (self.base, "HEAD"), (self.base, MISSING)]:
            with self.subTest(base=base, head=head):
                result = self.run_script(base, head)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_diff_failure_is_not_swallowed(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        wrapper = self.repo / "bin"
        wrapper.mkdir()
        real_git = shutil.which("git")
        (wrapper / "git").write_text(
            '#!/bin/sh\nif [ "$1" = diff ]; then exit 42; fi\n'
            f'exec "{real_git}" "$@"\n'
        )
        (wrapper / "git").chmod(0o755)
        env = dict(os.environ, PATH=str(wrapper) + os.pathsep + os.environ["PATH"])
        result = self.run_script(self.base, head, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("cannot compare CI commits", result.stderr)


if __name__ == "__main__":
    unittest.main()
