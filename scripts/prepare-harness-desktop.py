#!/usr/bin/env python3
"""Fetch the pinned, unmodified official Desktop; verify before using or distributing it."""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/harness-desktop-source.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_app(app, source):
    with (app / 'Contents/Info.plist').open('rb') as stream:
        info = plistlib.load(stream)
    require(info['CFBundleIdentifier'] == source['bundle_id'], 'unexpected Harness bundle ID')
    require(info['CFBundleShortVersionString'] == source['version'], 'unexpected Harness version')
    subprocess.run(['codesign', '--verify', '--deep', '--strict', app], check=True)
    identity = subprocess.check_output(['codesign', '-dv', app], stderr=subprocess.STDOUT, text=True)
    require(f"TeamIdentifier={source['team_id']}" in identity.splitlines(), 'unexpected Harness signing team')
    executable = app / 'Contents/MacOS' / info['CFBundleExecutable']
    subprocess.run(['lipo', executable, '-verify_arch', source['architecture']], check=True)
    subprocess.run(['spctl', '--assess', '--type', 'execute', app], check=True)


def prepare(output, archive=None):
    source = json.loads(SOURCE.read_text())
    require(re.fullmatch(r'[a-f0-9]{64}', source['sha256']), 'invalid pinned checksum')
    require(source['url'].startswith('https://download.deepseek.com/dsh-desk/bin/mac-arm64/'), 'unexpected source origin')
    require(not output.exists(), 'Harness output already exists; refusing to reuse unverified files')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ironmlx-harness-', dir=output.parent) as temporary:
        staging = Path(temporary)
        if archive is None:
            archive = staging / 'harness.zip'
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
                            '--proto', '=https', '--proto-redir', '=https', source['url'], '-o', archive], check=True)
        require(digest(archive) == source['sha256'], 'Harness archive checksum mismatch')
        with zipfile.ZipFile(archive) as package:
            for entry in package.infolist():
                path = Path(entry.filename)
                require(not path.is_absolute() and '..' not in path.parts, 'unsafe Harness archive path')
                require(path.parts[0] in ('DeepSeek Harness.app', '__MACOSX'), 'unexpected Harness archive root')
                if stat.S_ISLNK(entry.external_attr >> 16):
                    target = package.read(entry).decode('utf-8')
                    require(not target.startswith('/') and
                            (staging / path.parent / target).resolve().is_relative_to(staging / path.parts[0]),
                            'escaping Harness symlink')
        subprocess.run(['ditto', '-x', '-k', archive, staging], check=True)
        app = staging / 'DeepSeek Harness.app'
        validate_app(app, source)
        # Preserve all upstream resources, notices, native code and signatures.
        shutil.move(app, output)
    print(f"Verified official Harness {source['version']}: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--archive', type=Path, help='reuse downloaded archive; the pinned hash remains mandatory')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        validate_app(args.output.resolve(), json.loads(SOURCE.read_text()))
    else:
        prepare(args.output.resolve(), args.archive.resolve() if args.archive else None)


if __name__ == '__main__':
    main()
