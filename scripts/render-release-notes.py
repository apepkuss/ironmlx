#!/usr/bin/env python3
"""Render the GitHub Release body from the RC/stable release template."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


def version_tuple(tag: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)(?:-rc\.[1-9]\d*)?", tag)
    if match is None:
        raise ValueError(f"invalid release tag: {tag}")
    return tuple(int(value) for value in match.groups())


def previous_stable_tag(root: Path, tag: str) -> str | None:
    current = version_tuple(tag)
    try:
        tags = subprocess.check_output(
            ["git", "-C", str(root), "tag", "--merged", tag, "--list", "v*"],
            text=True,
            stderr=subprocess.PIPE,
        ).splitlines()
    except subprocess.CalledProcessError:
        # Allow release-note previews before the immutable tag is created.
        tags = subprocess.check_output(
            ["git", "-C", str(root), "tag", "--merged", "HEAD", "--list", "v*"],
            text=True,
        ).splitlines()
    candidates = []
    for candidate in tags:
        match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", candidate)
        if match is None:
            continue
        parsed = tuple(int(value) for value in match.groups())
        if parsed < current:
            candidates.append((parsed, candidate))
    return max(candidates)[1] if candidates else None


def render(root: Path, repository: str, tag: str, candidate: bool) -> str:
    template = (root / ".github/release-notes-template.md").read_text()
    version = tag.removeprefix("v").split("-", 1)[0]
    for relative in (f"docs/release-notes/{version}.md",
                     f"docs/zh-CN/release-notes/{version}.md"):
        if not (root / relative).is_file():
            raise FileNotFoundError(f"missing release notes: {relative}")
    previous = previous_stable_tag(root, tag)
    base = f"https://github.com/{repository}"
    release_url = f"{base}/releases/download/{tag}"
    values = {
        "RELEASE_STATUS_EN": f"{tag} is a prerelease." if candidate else f"{tag} is a stable release.",
        "PURPOSE_EN": "testing and feedback." if candidate else "general use.",
        "RELEASE_STATUS_ZH": f"{tag} 是预发布版本。" if candidate else f"{tag} 是正式版本。",
        "PURPOSE_ZH": "测试与反馈" if candidate else "日常使用",
        "MIN_MACOS": "26.4",
        "DMG_URL": f"{release_url}/IronMLX.dmg",
        "ZIP_URL": f"{release_url}/IronMLX-{version}.zip",
        "SUPPORTED_MODELS_URL": f"{base}/blob/{tag}/docs/supported-models.md",
        "KNOWN_ISSUES_URL": f"{base}/blob/{tag}/docs/known-issues.md",
        "SUPPORTED_MODELS_ZH_URL": f"{base}/blob/{tag}/docs/zh-CN/supported-models.md",
        "KNOWN_ISSUES_ZH_URL": f"{base}/blob/{tag}/docs/zh-CN/known-issues.md",
        "RELEASE_NOTES_URL": f"{base}/blob/{tag}/docs/release-notes/{version}.md",
        "RELEASE_NOTES_ZH_URL": f"{base}/blob/{tag}/docs/zh-CN/release-notes/{version}.md",
        "ISSUES_URL": f"{base}/issues",
        "CHANGELOG_URL": f"{base}/compare/{previous or tag + '^'}...{tag}",
        "UPDATE_CHANNELS_EN": "RC and stable updates use separate channels." if candidate else "Stable updates use the stable channel.",
        "UPDATE_CHANNELS_ZH": "RC 与稳定版使用独立更新通道。" if candidate else "正式版更新使用稳定通道。",
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    body = render(root, args.repository, args.tag, args.candidate)
    if args.output:
        args.output.write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
