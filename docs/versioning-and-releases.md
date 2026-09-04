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
