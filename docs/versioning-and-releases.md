# Versioning and release process

[简体中文](zh-CN/versioning-and-releases.md)

## One product version

The repository-root `VERSION` file is canonical. Rust workspace packages, CLI,
`healthz`, App `CFBundleShortVersionString`, and release tags must agree.
`CFBundleVersion` is a monotonically increasing positive integer.

Do not edit versions file by file. Run:

```bash
scripts/bump-version.sh 0.2.0
```

The script updates `VERSION`, workspace package versions, internal explicit
dependencies, `Cargo.lock`, and the App plist. It increments the App build
number by default; pass an explicit number when needed:

```bash
scripts/bump-version.sh 0.2.0 7
```

Commit all generated changes and run:

```bash
scripts/verify-version-consistency.sh
```

CI also verifies that every workspace crate declares `publish = false`, so
IronMLX cannot accidentally publish to crates.io.

## Tags and release notes

Stable tags use `vX.Y.Z` and must match `VERSION`. DMG, App About, CLI
`--version`, `healthz.version`, the release tag, and release notes must use the
same product version. Release candidates use `vX.Y.Z-rc.N` tags.

## Current hard gate

`scripts/release-legal-gate.sh` checks the authorized distribution flag and required licenses, Notices, inventory and reproducible SBOM. Read `scripts/release-config.sh` for the current flag. Passing this gate does not publish anything; signing, notarization, tag identity and explicit publication remain separate checks. See [Release pipeline](stable-release-pipeline.md) for material updates.

## Stable release identity

Before packaging, run the identity gate with the exact existing release tag:

```bash
python3 scripts/verify-release-identity.py v0.1.0
python3 scripts/verify-release-identity.py v0.1.0 dist/IronMLX.app
```

The source check requires the tag under `refs/tags/` to resolve to HEAD, match
`VERSION`, and use a clean checkout, including non-ignored untracked files.
Lightweight and annotated tags are supported. The optional App check additionally
requires matching product version, build number, source commit, and a `clean`
source-tree marker. The stable packager always performs both checks; its third
argument is the release tag (default: `v` plus `VERSION`). Both automatic and
manual stable workflows pass the selected tag explicitly.

This gate validates identity metadata, not cryptographic build provenance or
signing/notarization. Those remain separate release gates. Local development
builds and their static Bundle checks continue to allow dirty source trees.

RC identity validation is explicitly separate from stable packaging:

```bash
python3 scripts/verify-release-identity.py --candidate v0.1.0-rc.1 dist/IronMLX.app
```

Candidate mode accepts only `vX.Y.Z-rc.N` with a positive, non-zero-prefixed N.
It compares the base `X.Y.Z` to the App version and `VERSION`, retaining all
clean-checkout, tag/HEAD, build-number and Bundle-source checks. Stable packaging
and publication never enable this mode and continue to reject RC tags.

## RC and stable release entry points

| Mode | Tag | Workflow |
| --- | --- | --- |
| RC | `vX.Y.Z-rc.N` | Release Candidate |
| Stable | `vX.Y.Z` | Stable Release |

Tag pushes and manual `publish=false` build and validate only. Explicit `publish=true` enters signing, notarization and publication.
Both require the repository update public key. Credentials, ordering, channels and recovery are documented in the [release pipeline](stable-release-pipeline.md).

## Archive content checks

Stable assets use one output directory containing `IronMLX-X.Y.Z.dmg`,
`IronMLX-X.Y.Z.zip`, `SHA256SUMS`, the individual legal materials and
`THIRD_PARTY_LICENSES/`. The ZIP has an `IronMLX-X.Y.Z/` root; the mounted DMG
has the same contents at its volume root. Both contain `IronMLX.app` and all
legal materials. The output directory must be absent or empty; existing
artifacts are never automatically deleted.

Archive mechanics can be validated without a release tag or Developer ID:

```bash
python3 scripts/release-archives.py assemble dist/IronMLX.app .build/archive-check
python3 scripts/release-archives.py verify dist/IronMLX.app .build/archive-check
```

This checks current materials/SBOM, product version and identifier, exact
checksum coverage, actual ZIP extraction and read-only DMG mounting, and every
App file, executable bit and symlink against the reference Bundle (including
version/source metadata and embedded legal files). It does not certify that the
reference Bundle is a current, clean, signed release. The stable packager retains
identity, clean-source, legal-authorization, static Bundle, signing and Gatekeeper
gates before invoking this archive engine. Content-only artifacts must not be
published as approved stable releases.

Credentials, signing, notarization, feeds and recovery are documented in the [release pipeline](stable-release-pipeline.md).
