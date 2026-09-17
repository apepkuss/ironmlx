#!/usr/bin/env python3
"""Render the GitHub Release body from the RC/stable release template."""

from __future__ import annotations

import argparse
from pathlib import Path


def render(root: Path, repository: str, tag: str, candidate: bool) -> str:
    template = (root / ".github/release-notes-template.md").read_text()
    version = tag.removeprefix("v").split("-", 1)[0]
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
        "RELEASE_NOTES_URL": f"{base}/blob/{tag}/docs/release-notes/0.1.0.md",
        "RELEASE_NOTES_ZH_URL": f"{base}/blob/{tag}/docs/zh-CN/release-notes/0.1.0.md",
        "ISSUES_URL": f"{base}/issues",
        "CHANGELOG_URL": f"{base}/compare/{tag}^...{tag}",
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
