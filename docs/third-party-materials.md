# Third-party dependency and license materials

## Scope

P0-8A creates a reproducible engineering inventory for the macOS arm64 Release
product actually embedded in `IronMLX.app`, rather than copying the complete
`Cargo.lock` graph. It includes the Rust Release binaries and their target
dependencies, external SwiftPM packages (Sparkle and ZIPFoundation), the pinned
MLX fork and native inputs used by its Release build, and third-party bundled
graphics and branding assets. It excludes system frameworks and model weights
downloaded separately by users; see [Model rights boundary](model-license-boundary.md).

These materials preserve notices and detect dependency drift. They are not legal
advice or public-distribution authorization.

## Inputs and generated files

- `Cargo.lock`, product manifests, and `about.toml`;
- `ironmlx-app/Package.swift`;
- the pinned MLX fork commit in `scripts/release-config.sh`;
- locked native dependency and bundled-asset manifests with SHA-256 values;
- `third-party-inventory.json`;
- `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/`.

The generator records the exact MLX fork commit, upstream repository and base
revision, and validates native archives, bundled files, and license hashes.

## Update and verification

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
CARGO_ABOUT="$(command -v cargo-about)" scripts/update-third-party-materials.sh
scripts/verify-third-party-materials.sh
```

After dependency changes, review the complete diff of all generated materials;
updating a hash alone is not an acceptable workaround. CI regenerates materials
from the actual App-build inputs and compares them byte-for-byte.

## App and archives

Release builds copy these materials to:

```text
IronMLX.app/Contents/Resources/Legal/
```

The App menu exposes **Third-Party Notices…**. Approved development previews
also include the same materials at the DMG/ZIP root; their verifier extracts the
ZIP, mounts the DMG, and compares every item.

## Remaining P0-8B gate

- Final legal review of license expressions, attribution, and closed-binary
  distribution obligations;
- CycloneDX `SBOM.cdx.json` generation and review;
- final legal review of the model-rights boundary statement;
- explicit authorization to set `IRONMLX_PUBLIC_DISTRIBUTION_READY=true`;
- Developer ID signing, notarization, stapling, and minimum-target acceptance.
