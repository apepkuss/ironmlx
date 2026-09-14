# Versioning and release process

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
same product version. Development previews use the separate
`preview-YYYYMMDD-shortSHA` namespace.

## Current hard gate

`release-legal-gate.sh` runs during packaging and in the GitHub preview workflow.
`IRONMLX_PUBLIC_DISTRIBUTION_READY=false` currently makes public binary release
fail. After P0-8B, an authorized reviewer may enable it only when notices,
inventory, license texts, SBOM, and final legal review are complete.

The gate requires the project `LICENSE`, `NOTICE`, and deterministic
`SBOM.cdx.json` to be present in the release materials. It does not require or
imply a particular first-party open-source license;
that policy is a separate release decision. See [Third-party materials](third-party-materials.md)
for the locked inventory process.

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

## RC packaging and publication

The Release Candidate workflow requires an immutable vX.Y.Z-rc.N tag. Tag push
and publish=false validate only, without Apple credentials or publication.
The repository variable IRONMLX_UPDATE_PUBLIC_ED_KEY is required.

Manual publish=true uses the stable-release Environment credentials and retains
public-distribution authorization gates. It signs and notarizes the App, staples
its ticket, packages ZIP/DMG, then signs/notarizes/staples the final DMG.
The App remains IronMLX.app and uses release-candidate distribution/update channels.
Archive output uses .build/stable-release (shared with the stable packaging engine).

A draft is uploaded and downloaded for exact asset/hash verification before
promotion to a prerelease with make_latest=false. The RC feed is updated only
after public download verification. Existing releases/tags are never overwritten.
See [RC signing](zh-CN/rc-signing.md) for configuration and validation boundaries.

## Stable archive layout and independent content verification

Stable assets now use one output directory containing `IronMLX-X.Y.Z.dmg`,
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

Public Sparkle channel configuration and feed deployment are documented in
[Automatic updates](automatic-updates.md).

See [Stable release pipeline](stable-release-pipeline.md) for validation-only dispatch, signing credentials and draft publication.
