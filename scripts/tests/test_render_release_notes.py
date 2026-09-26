"""Release-note links and changelog boundaries."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "render-release-notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)


class ReleaseNotesTests(unittest.TestCase):
    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release@example.invalid")
        (self.root / ".github").mkdir()
        (self.root / ".github/release-notes-template.md").write_text(
            "{{RELEASE_NOTES_URL}}\n{{RELEASE_NOTES_ZH_URL}}\n{{CHANGELOG_URL}}\n"
        )
        for language in ("", "zh-CN/"):
            path = self.root / f"docs/{language}release-notes"
            path.mkdir(parents=True)
            (path / "0.1.0.md").write_text("old\n")
        self.git("add", ".")
        self.git("commit", "-qm", "initial release")
        self.git("tag", "v0.1.0")

    def prepare_candidate(self, tag="v0.2.0-rc.1"):
        for language in ("", "zh-CN/"):
            path = self.root / f"docs/{language}release-notes/0.2.0.md"
            path.write_text("new\n")
        self.git("add", ".")
        self.git("commit", "-qm", "prepare candidate")
        self.git("tag", tag)

    def test_candidate_links_versioned_notes_and_previous_stable_changelog(self):
        self.prepare_candidate()
        body = release_notes.render(self.root, "owner/repo", "v0.2.0-rc.1", True)
        self.assertIn("/blob/v0.2.0-rc.1/docs/release-notes/0.2.0.md", body)
        self.assertIn("/blob/v0.2.0-rc.1/docs/zh-CN/release-notes/0.2.0.md", body)
        self.assertIn("/compare/v0.1.0...v0.2.0-rc.1", body)

    def test_missing_versioned_notes_are_rejected(self):
        self.git("commit", "--allow-empty", "-qm", "missing notes")
        self.git("tag", "v0.2.0-rc.1")
        with self.assertRaisesRegex(FileNotFoundError, "0.2.0"):
            release_notes.render(self.root, "owner/repo", "v0.2.0-rc.1", True)

    def test_invalid_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            release_notes.version_tuple("v0.2-rc.1")


if __name__ == "__main__":
    unittest.main()
