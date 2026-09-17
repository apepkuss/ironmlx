#!/usr/bin/env python3
"""Populate the website release badge and stable/RC download links."""

from __future__ import annotations

import html
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "apepkuss/ironmlx")
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=100"
TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")


def releases() -> list[dict]:
    request = urllib.request.Request(
        API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ironmlx-website-build"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def version_key(release: dict) -> tuple[int, int, int, int, int]:
    match = TAG_RE.match(release.get("tag_name", ""))
    if not match:
        return (-1, -1, -1, -1, -1)
    major, minor, patch, rc = (int(value or 0) for value in match.groups())
    # A stable release sorts after all RCs of the same version.
    return (major, minor, patch, 1 if rc == 0 and "-rc." not in release["tag_name"] else 0, rc)


def dmg_url(release: dict) -> str | None:
    assets = release.get("assets", [])
    # Prefer the unified filename, while accepting older releases that used
    # the version in the DMG filename.
    candidates = [a for a in assets if a.get("name") == "IronMLX.dmg"]
    candidates += [
        a for a in assets
        if a.get("name", "").startswith("IronMLX-") and a.get("name", "").endswith(".dmg")
    ]
    if candidates:
        return candidates[0].get("browser_download_url") or (
            f"https://github.com/{REPOSITORY}/releases/download/{release['tag_name']}/{candidates[0]['name']}"
        )
    return None


def main() -> None:
    try:
        candidates = [r for r in releases() if not r.get("draft") and TAG_RE.match(r.get("tag_name", ""))]
    except Exception as error:  # pragma: no cover - network is only available in deployment
        raise SystemExit(f"Unable to read GitHub releases: {error}")
    if not candidates:
        raise SystemExit("No public semantic-versioned GitHub release found")

    newest = max(candidates, key=version_key)
    stable = [r for r in candidates if not r.get("prerelease") and "-rc." not in r["tag_name"]]
    stable_release = max(stable, key=version_key) if stable else None
    stable_url = dmg_url(stable_release) if stable_release else None
    stable_button_en = (
        f'<a class="button primary" href="{html.escape(stable_url, quote=True)}">Download stable release</a>'
        if stable_url
        else '<span class="button primary disabled">Stable release coming soon</span>'
    )
    stable_button_zh = (
        f'<a class="button primary" href="{html.escape(stable_url, quote=True)}">下载正式版</a>'
        if stable_url
        else '<span class="button primary disabled">正式版即将发布</span>'
    )

    tag = newest["tag_name"]
    is_rc = bool(newest.get("prerelease")) or "-rc." in tag
    rc_url = dmg_url(newest) if is_rc else None
    rc_en = (
        f'<a href="{html.escape(rc_url, quote=True)}">Try {html.escape(tag)}</a>'
        if rc_url
        else ""
    )
    rc_zh = (
        f'<a href="{html.escape(rc_url, quote=True)}">尝鲜 {html.escape(tag)}</a>'
        if rc_url
        else ""
    )
    replacements = {
        "__RELEASE_BADGE_EN__": f"● {tag} · {'Release Candidate' if is_rc else 'Latest Release'}",
        "__RELEASE_BADGE_ZH__": f"● {tag} · {'候选版本' if is_rc else '正式版'}",
        "__STABLE_BUTTON_EN__": stable_button_en,
        "__STABLE_BUTTON_ZH__": stable_button_zh,
        "__RC_DOWNLOAD_EN__": rc_en,
        "__RC_DOWNLOAD_ZH__": rc_zh,
    }
    for relative in (Path("website/index.html"), Path("website/zh-Hans/index.html")):
        path = ROOT / relative
        content = path.read_text()
        for marker, value in replacements.items():
            content = content.replace(marker, value)
        path.write_text(content)
    print(f"Website release metadata: newest={tag}, stable={stable_release['tag_name'] if stable_release else 'none'}")


if __name__ == "__main__":
    main()
