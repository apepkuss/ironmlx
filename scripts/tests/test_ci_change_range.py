"""Route website/native changes independently without skipping shared gates."""
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

    def test_normal_docs_change_runs_documentation_and_website_quality(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        result = self.resolve(self.base, head)
        self.assertEqual(result["base_sha"], self.base)
        self.assertEqual(result["docs_changed"], "true")
        self.assertEqual(result["website_changed"], "true")
        self.assertEqual(result["code_changed"], "false")

    def test_website_changes_do_not_run_native_quality(self):
        for path in ["website/index.html", "website/zh/index.html", "website/styles.css",
                     "website/docs.js", "website/assets/navigation-icons.svg",
                     "scripts/build-website-docs.py", "scripts/build-website-release-metadata.py",
                     "scripts/tests/test_website_docs.py", ".github/workflows/website-preview.yml",
                     "website/README.md"]:
            with self.subTest(path=path):
                base = self.git("rev-parse", "HEAD")
                head = self.commit(path, "website change", "feat(website): update page")
                result = self.resolve(base, head)
                self.assertEqual(result["website_changed"], "true")
                self.assertEqual(result["code_changed"], "false")
                self.assertEqual(result["docs_changed"], str(path.endswith(".md")).lower())

    def test_documentation_assets_and_root_sources_run_website_quality(self):
        for path in ["docs/images/example.png", "docs/validation/clients.json",
                     "docs/zh-CN/user-guide.md", "CONTRIBUTING.md", "SECURITY.md",
                     "SUPPORT.md", "ironmlx-audio/README.md", "README.zh-CN.md"]:
            with self.subTest(path=path):
                base = self.git("rev-parse", "HEAD")
                head = self.commit(path, "updated documentation", "docs: update website inputs")
                result = self.resolve(base, head)
                self.assertEqual(result["docs_changed"], "true")
                self.assertEqual(result["website_changed"], "true")
                self.assertEqual(result["code_changed"], "false")

    def test_shared_website_inputs_run_website_and_native_quality(self):
        for path in ["ironmlx-app/Sources/IronMLXAppCore/Resources/supported-models.json",
                     "scripts/build-supported-models-docs.py", "VERSION",
                     ".github/release-notes-template.md", "scripts/render-release-notes.py",
                     "scripts/tests/test_render_release_notes.py"]:
            with self.subTest(path=path):
                base = self.git("rev-parse", "HEAD")
                head = self.commit(path, "shared change", "build: update shared inputs")
                result = self.resolve(base, head)
                self.assertEqual(result["website_changed"], "true")
                self.assertEqual(result["code_changed"], "true")

    def test_ci_and_repository_policy_changes_run_every_quality_job(self):
        for path in [".github/workflows/ci.yml", ".github/workflows/stable-release.yml",
                     "scripts/resolve-ci-change-range.sh", "scripts/tests/test_ci_change_range.py",
                     ".gitignore", ".gitattributes", ".gitmodules"]:
            with self.subTest(path=path):
                base = self.git("rev-parse", "HEAD")
                head = self.commit(path, "policy change", "ci: update quality policy")
                result = self.resolve(base, head)
                for flag in ["docs_changed", "website_changed", "code_changed"]:
                    self.assertEqual(result[flag], "true")

    def test_deleting_website_files_still_runs_website_quality(self):
        base = self.commit("website/retired.html", "old page", "feat(website): add page")
        self.git("rm", "website/retired.html")
        self.git("commit", "-qm", "fix(website): remove retired page")
        result = self.resolve(base, self.git("rev-parse", "HEAD"))
        self.assertEqual(result["website_changed"], "true")
        self.assertEqual(result["code_changed"], "false")

    def test_moves_across_website_and_native_inputs_check_both_sides(self):
        for old, new in [
            ("ironmlx-app/Sources/IronMLXAppCore/Resources/supported-models.json", "website/catalogue.json"),
            ("src/lib.rs", "website/example.rs"),
            ("website/example.js", "src/example.js"),
        ]:
            with self.subTest(old=old, new=new):
                base = self.commit(old, "same file contents", "test: add move fixture")
                (self.repo / new).parent.mkdir(parents=True, exist_ok=True)
                self.git("mv", old, new)
                self.git("commit", "-qm", "refactor: move shared input")
                result = self.resolve(base, self.git("rev-parse", "HEAD"))
                self.assertEqual(result["website_changed"], "true")
                self.assertEqual(result["code_changed"], "true")

    def test_mixed_website_and_native_changes_run_both_jobs(self):
        self.commit("website/index.html", "page", "feat(website): add page")
        head = self.commit("src/lib.rs", "source", "feat: add source")
        result = self.resolve(self.base, head)
        self.assertEqual(result["website_changed"], "true")
        self.assertEqual(result["code_changed"], "true")

    def test_native_change_runs_native_quality_only(self):
        head = self.commit("src/lib.rs", "fn main() {}", "feat: add source")
        result = self.resolve(self.base, head)
        self.assertEqual(result["code_changed"], "true")
        self.assertEqual(result["docs_changed"], "false")
        self.assertEqual(result["website_changed"], "false")

    def test_unknown_non_website_changes_retain_native_quality(self):
        head = self.commit("tools/custom-build-input.txt", "input", "build: add build input")
        result = self.resolve(self.base, head)
        self.assertEqual(result["code_changed"], "true")
        self.assertEqual(result["website_changed"], "false")

    def test_mixed_change_runs_documentation_and_code_quality(self):
        docs = self.commit("docs/api.md", "API", "docs: add API")
        head = self.commit("src/lib.rs", "fn main() {}", "feat: add source")
        result = self.resolve(self.base, head)
        self.assertNotEqual(docs, head)
        self.assertEqual(result["docs_changed"], "true")
        self.assertEqual(result["website_changed"], "true")
        self.assertEqual(result["code_changed"], "true")

    def test_missing_old_tip_forces_full_quality_even_for_docs(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        self.assertNotEqual(subprocess.run(["git", "cat-file", "-e", MISSING],
                                          cwd=self.repo, capture_output=True).returncode, 0)
        result = self.resolve(MISSING, head)
        self.assertEqual(result["base_sha"], self.base)
        self.assertEqual(result["head_sha"], head)
        self.assertEqual(result["docs_changed"], "true")
        self.assertEqual(result["website_changed"], "true")
        self.assertEqual(result["code_changed"], "true")

    def test_new_branch_uses_reachable_ancestor(self):
        head = self.commit("docs/api.md", "API", "docs: add API")
        result = self.resolve(ZERO, head)
        self.assertEqual(result["base_sha"], self.base)
        self.assertEqual(result["docs_changed"], "true")
        self.assertEqual(result["website_changed"], "true")
        self.assertEqual(result["code_changed"], "true")

    def test_unavailable_bases_force_all_jobs_for_website_changes(self):
        head = self.commit("website/index.html", "page", "feat(website): add page")
        for base in [ZERO, MISSING]:
            with self.subTest(base=base):
                result = self.resolve(base, head)
                for flag in ["docs_changed", "website_changed", "code_changed"]:
                    self.assertEqual(result[flag], "true")

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
