#!/usr/bin/env python3
"""Upload and verify a complete draft before making a stable release or RC public."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def run(*args):
    return subprocess.check_output([str(arg) for arg in args], text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def publish(repo, tag, commit, assets, candidate=False):
    require(re.fullmatch(r'[\w.-]+/[\w.-]+', repo), 'invalid repository')
    pattern = r'v[0-9]+\.[0-9]+\.[0-9]+' + (r'-rc\.[1-9][0-9]*' if candidate else '')
    require(re.fullmatch(pattern, tag), 'invalid release tag')
    require(len({p.name for p in assets}) == len(assets), 'duplicate asset names')
    expected = {p.name: sha(p) for p in assets}
    route = f'repos/{repo}'

    def check_tag():
        require(json.loads(run('gh', 'api', f'{route}/commits/{tag}'))['sha'] == commit,
                'remote release tag moved')

    def verify_downloads():
        release = json.loads(run('gh', 'api', f'{route}/releases/{release_id}'))
        require(release['tag_name'] == tag, 'release tag differs')
        require(release['prerelease'] == candidate, 'unexpected prerelease state')
        names = [asset['name'] for asset in release['assets']]
        require(len(names) == len(expected) and set(names) == set(expected), 'release asset set differs')
        with tempfile.TemporaryDirectory(prefix='ironmlx-release-download-') as tmp:
            run('gh', 'release', 'download', tag, '--repo', repo, '--dir', tmp)
            require({p.name: sha(p) for p in Path(tmp).iterdir()} == expected,
                    'downloaded release assets differ')
        return release

    check_tag()
    def find_release():
        # The by-tag REST endpoint does not resolve unpublished drafts.
        ids = run('gh', 'api', f'{route}/releases', '--paginate', '--jq',
                  f'.[] | select(.tag_name == "{tag}") | .id').splitlines()
        require(len(ids) <= 1, 'multiple releases for tag')
        require(all(value.isdigit() for value in ids), 'invalid release ID')
        return ids[0] if ids else None

    release_id = find_release()
    if release_id is None:
        created = json.loads(run('gh', 'api', f'{route}/releases', '--method', 'POST',
                                 '-f', f'tag_name={tag}', '-f', f'target_commitish={commit}',
                                 '-f', f'name=IronMLX {tag}', '-F', 'draft=true',
                                 '-F', f'prerelease={str(candidate).lower()}',
                                 '-F', 'generate_release_notes=true', '-f', 'make_latest=false'))
        release_id = created.get('id')
        require(type(release_id) is int and release_id > 0, 'invalid created release ID')
        require(created.get('tag_name') == tag and created.get('draft') is True
                and created.get('prerelease') == candidate, 'unexpected created release state')
        run('gh', 'release', 'upload', tag, '--repo', repo, *assets)
    else:
        existing = json.loads(run('gh', 'api', f'{route}/releases/{release_id}'))
        require(existing['draft'], 'release is already public')
        # Resume only an exact draft: never replace or add assets on retry.
    release = verify_downloads()
    require(release['draft'], 'release became public before verification')
    check_tag()
    run('gh', 'api', f"{route}/releases/{release['id']}", '--method', 'PATCH',
        '-F', 'draft=false', '-f', 'make_latest=false' if candidate else 'make_latest=true')
    require(not verify_downloads()['draft'], 'release is still a draft')
    print(f'Published and downloaded verified release: {tag}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--repository', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    run(root / 'scripts/release-legal-gate.sh')
    run('python3', root / 'scripts/verify-release-identity.py', *(['--candidate'] if args.candidate else []), args.tag, root / 'dist/IronMLX.app')
    run(root / 'scripts/verify-app-bundle.sh', root / 'dist/IronMLX.app')
    run('xcrun', 'stapler', 'validate', root / 'dist/IronMLX.app')
    run('spctl', '--assess', '--type', 'execute', root / 'dist/IronMLX.app')
    run('python3', root / 'scripts/release-archives.py', 'verify', root / 'dist/IronMLX.app',
        root / '.build/stable-release')
    for dmg in (root / '.build/stable-release').glob('*.dmg'):
        run('codesign', '--verify', '--strict', dmg)
        run('xcrun', 'stapler', 'validate', dmg)
        run('spctl', '--assess', '--type', 'open', '--context', 'context:primary-signature', dmg)
    # Detailed materials remain inside the installation packages.
    version = (root / 'VERSION').read_text().strip()
    assets = [root / '.build/stable-release' / f'IronMLX-{version}.{kind}'
              for kind in ('dmg', 'zip')]
    update = root / '.build/app-update'
    data = json.loads((update / 'update.json').read_text())
    require(data['tag'] == args.tag and data['channel'] == ('release-candidate' if args.candidate else 'stable'), 'update identity mismatch')
    require(data['archive'] == f'IronMLX-{args.tag}-update.zip' and data['feed'] == ('release-candidate.xml' if args.candidate else 'stable.xml'),
            'invalid update asset names')
    for kind in ('archive', 'feed'):
        require(sha(update / data[kind]) == data[kind + '_sha256'], 'update hash mismatch')
    # The signed feed and its metadata are published separately on updates.
    assets.append(update / data['archive'])
    manifest = root / '.build/RELEASE-SHA256SUMS'
    manifest.write_text(''.join(f'{sha(p)}  {p.name}\n' for p in assets))
    publish(args.repository, args.tag, run('git', '-C', root, 'rev-parse', 'HEAD'), assets + [manifest], candidate=args.candidate)


if __name__ == '__main__':
    main()
