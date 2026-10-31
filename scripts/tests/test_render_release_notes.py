"""Release-specific content, immutable links and changelog boundaries."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "render-release-notes.py"
TEMPLATE = SCRIPT.parent.parent / ".github/release-notes-template.md"
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
            TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8"
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
            path.write_text("\n".join(
                f"<!-- release-summary:{key} -->\n"
                f"- {language or 'en/'}{key} change: [guide](../user-guide.md#models).\n"
                "<!-- /release-summary -->"
                for key in ("rc.1", "rc.2", "rc.3", "rc.4", "stable")
            ), encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "--allow-empty", "-qm", "prepare candidate")
        self.git("tag", tag)

    def test_candidate_links_versioned_notes_and_previous_stable_changelog(self):
        self.prepare_candidate()
        body = release_notes.render(self.root, "owner/repo", "v0.2.0-rc.1", True)
        self.assertIn("/blob/v0.2.0-rc.1/docs/release-notes/0.2.0.md", body)
        self.assertIn("/blob/v0.2.0-rc.1/docs/zh-CN/release-notes/0.2.0.md", body)
        self.assertIn("/compare/v0.1.0...v0.2.0-rc.1", body)
        self.assertIn("### Changes since v0.1.0", body)
        self.assertIn("en/rc.1 change", body)
        self.assertIn("zh-CN/rc.1 change", body)
        self.assertNotIn("rc.2 change", body)
        self.assertNotIn("{{", body)

    def test_successive_candidates_use_only_their_own_changes_and_previous_rc(self):
        for n in range(1, 5):
            tag = f"v0.2.0-rc.{n}"
            self.prepare_candidate(tag)
            body = release_notes.render(self.root, "owner/repo", tag, True)
            previous = "v0.1.0" if n == 1 else f"v0.2.0-rc.{n - 1}"
            self.assertIn(f"/compare/{previous}...{tag}", body)
            self.assertIn(f"en/rc.{n} change", body)
            self.assertIn(f"zh-CN/rc.{n} change", body)
            for other in range(1, 5):
                if other != n:
                    self.assertNotIn(f"rc.{other} change", body)
            self.assertNotIn("stable change", body)

    def test_stable_uses_release_line_summary_and_previous_stable(self):
        self.prepare_candidate()
        self.git("tag", "v0.2.0")
        body = release_notes.render(self.root, "owner/repo", "v0.2.0", False)
        self.assertIn("/compare/v0.1.0...v0.2.0", body)
        self.assertIn("en/stable change", body)
        self.assertIn("zh-CN/stable change", body)
        self.assertNotIn("rc.1 change", body)

    def test_preview_before_tag_creation_uses_previous_candidate(self):
        self.prepare_candidate()
        body = release_notes.render(self.root, "owner/repo", "v0.2.0-rc.2", True)
        self.assertIn("/compare/v0.2.0-rc.1...v0.2.0-rc.2", body)

    def test_missing_preceding_candidate_is_rejected(self):
        self.prepare_candidate()
        with self.assertRaisesRegex(ValueError, "preceding candidate tag"):
            release_notes.render(self.root, "owner/repo", "v0.2.0-rc.3", True)

    def test_summary_links_resolve_against_each_languages_tagged_notes(self):
        self.prepare_candidate()
        path = self.root / "docs/release-notes/0.2.0.md"
        text = path.read_text().replace(
            "en/rc.1 change:",
            "en/rc.1 change: [external](https://example.com/page) [section](#details)",
        )
        path.write_text(text)
        body = release_notes.render(self.root, "owner/repo", "v0.2.0-rc.1", True)
        self.assertIn("(https://example.com/page)", body)
        self.assertIn("/blob/v0.2.0-rc.1/docs/release-notes/0.2.0.md#details)", body)
        self.assertIn("/blob/v0.2.0-rc.1/docs/user-guide.md#models)", body)
        self.assertIn("/blob/v0.2.0-rc.1/docs/zh-CN/user-guide.md#models)", body)

    def test_missing_translation_summary_is_rejected(self):
        self.prepare_candidate()
        path = self.root / "docs/zh-CN/release-notes/0.2.0.md"
        path.write_text("<!-- release-summary:stable -->\n- Stable only.\n<!-- /release-summary -->")
        with self.assertRaisesRegex(ValueError, "rc.1.*zh-CN"):
            release_notes.render(self.root, "owner/repo", "v0.2.0-rc.1", True)

    def test_empty_duplicate_and_unclosed_summaries_are_rejected(self):
        self.prepare_candidate()
        path = self.root / "docs/release-notes/0.2.0.md"
        start = "<!-- release-summary:rc.1 -->"
        end = "<!-- /release-summary -->"
        for notes in (f"{start}\n{end}", f"{start}\n- Missing end.",
                      f"{start}\n- First.\n{end}\n{start}\n- Duplicate.\n{end}",
                      f"{start}\n<!-- release-summary:rc.2 -->\n- Nested.\n{end}"):
            with self.subTest(notes=notes):
                path.write_text(notes)
                with self.assertRaises(ValueError):
                    release_notes.render(self.root, "owner/repo", "v0.2.0-rc.1", True)

    def test_tag_and_publication_mode_must_agree(self):
        self.prepare_candidate()
        for tag, candidate in (("v0.2.0", True), ("v0.2.0-rc.1", False)):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "--candidate"):
                release_notes.render(self.root, "owner/repo", tag, candidate)

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
