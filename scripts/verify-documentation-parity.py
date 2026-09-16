#!/usr/bin/env python3
"""Check the documented English/Simplified Chinese file map and structure."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
ENGLISH = ROOT / "docs"
CHINESE = ENGLISH / "zh-CN"
PAIRED_DOCS = (
    "automatic-updates.md", "building-from-source.md", "known-issues.md",
    "privacy.md", "release-notes/0.1.0.md", "storage-and-uninstall.md",
    "supported-models.md", "troubleshooting.md", "user-guide.md",
    "versioning-and-releases.md",
)


def structure(path: Path):
    headings = []
    lists = []
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^(#{1,6})\s+", line)
        if heading:
            headings.append(len(heading.group(1)))
        if re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line):
            lists.append(bool(re.match(r"^\s*\d+[.)]\s+", line)))
    return headings, lists


def main() -> int:
    errors = []
    root_en = ROOT / "README.md"
    root_zh = ROOT / "README.zh-CN.md"
    if structure(root_en) != structure(root_zh):
        errors.append("heading or list structure differs: README.md vs README.zh-CN.md")
    for relative in PAIRED_DOCS:
        english = ENGLISH / relative
        chinese = CHINESE / relative
        if not chinese.is_file():
            errors.append(f"missing Simplified Chinese counterpart: {english.relative_to(ROOT)}")
            continue
        en_headings, en_lists = structure(english)
        zh_headings, zh_lists = structure(chinese)
        if en_headings != zh_headings:
            errors.append(f"heading structure differs: {english.relative_to(ROOT)} vs {chinese.relative_to(ROOT)}")
        if len(en_lists) != len(zh_lists):
            errors.append(f"list item count differs: {english.relative_to(ROOT)} vs {chinese.relative_to(ROOT)}")
    if errors:
        print("Documentation parity check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("Documentation mappings and structure passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
