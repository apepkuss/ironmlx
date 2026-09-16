# RC and stable release pipeline

[简体中文](zh-CN/stable-release-pipeline.md)

For release maintainers. Version/tag identity is defined in [Versioning and releases](versioning-and-releases.md); this document owns signing, publication and recovery. User-facing update behavior is in [Automatic updates](automatic-updates.md).

## Build and validation modes

| Mode | Tag | Distribution / update channel | Public Release |
| --- | --- | --- | --- |
| RC | `vX.Y.Z-rc.N` | `release-candidate` | Prerelease, `make_latest=false` |
| Stable | `vX.Y.Z` | `stable` | Stable release |

Both workflows build an existing immutable tag from a clean full checkout. They check VERSION, build number, source commit and Bundle identity. MLX is pinned in `scripts/release-config.sh`; toolchain versions are defined by the workflows.
Tag pushes and manual `publish=false` build and verify archives without signing secrets, Release creation or feed publication. The repository variable `IRONMLX_UPDATE_PUBLIC_ED_KEY` is required even in validation mode. Archive checks do not establish notarization or Gatekeeper acceptance.

## Publication configuration

Explicit manual `publish=true` transfers the exact validated App ZIP, retaining permissions, framework symlinks and signatures. Distribution materials must pass `release-legal-gate.sh`; the workflow checks the authorization flag rather than changing it.
The publish job uses the `stable-release` GitHub Environment. Configure its deployment rules according to the repository's release policy. Required values are:

| Kind | Name | Value |
|---|---|---|
| Secret | `IRONMLX_DEVELOPER_ID_P12_BASE64` | Base64 PKCS#12 containing Developer ID Application certificate and private key |
| Secret | `IRONMLX_DEVELOPER_ID_P12_PASSWORD` | Nonempty PKCS#12 export password |
| Variable | `IRONMLX_SIGNING_IDENTITY` | Full `Developer ID Application: Name (TEAMID)` identity |
| Variable | `IRONMLX_APPLE_TEAM_ID` | Certificate team ID |
| Secret | `IRONMLX_NOTARY_KEY_ID` | App Store Connect team API key ID |
| Secret | `IRONMLX_NOTARY_ISSUER_ID` | Team API issuer UUID |
| Secret | `IRONMLX_NOTARY_PRIVATE_KEY` | API key `.p8` contents |
| Secret | `IRONMLX_UPDATE_PRIVATE_ED_KEY` | Existing Sparkle Ed25519 seed, matching the build's public key |

The public key variable stays at repository scope because build jobs do not use the production Environment. The Sparkle private key can also be supplied by the existing repository secret. Do not commit private credentials.

## Signing and publication order

1. Download the validated candidate and recheck the exact source/tag/Bundle identity. Finalize Bundle metadata before signing.
2. Import credentials into a temporary keychain. Sign Sparkle components from inside out, Rust helpers and the App with hardened runtime and a secure timestamp. Preserve Downloader sandbox/network entitlements; no JIT or library-validation exceptions are added.
3. Submit the App ZIP to Apple, require `Accepted`, then staple and validate the ticket and perform signature/Gatekeeper checks. A plist status declaration is not ticket evidence.
4. Package the installer ZIP and DMG from the stapled App. Sign, notarize and staple the DMG, refresh checksums, then extract/mount and verify the contents against the reference App and source materials.
5. Generate a separate App-only update ZIP, sign ZIP/XML with Sparkle and verify them. Record `update.json` and `RELEASE-SHA256SUMS` for uploaded assets. The installer ZIP is not the update ZIP; no delta updates are produced.
6. Create a draft, upload the exact asset set, download and hash-check it, and recheck the remote tag. Only then make it public with the appropriate RC/stable status.
7. Recheck public downloads and release identity before publishing the feed. RC uses only its own feed and is never promoted through the stable feed by this operation.

Both modes retain `IronMLX.app`; shared packaging uses `.build/stable-release`. App and final DMG notarization are separate submissions.

## Failure and retry

Build/signing/notarization failures stop before Release creation. Temporary keys, certificates and keychains are removed on normal failure/termination, with workflow `always()` cleanup. Apple receipts stay in `.build/notarization` and are not public assets.
Upload or draft verification failure leaves a draft without making it public. A normal rerun rejects an existing Release rather than replacing its assets or deleting it; inspect and resolve the failed draft explicitly.
If the Release is public but feed publication failed, reuse the exact published ZIP/XML/`update.json` with `publish-update-feed.py`. Do not rebuild or overwrite release assets. Identical manifests can be retried idempotently.

## Update channel and version rules

The App product version stays `X.Y.Z`; RC tags/feed display versions add `-rc.N`. Sparkle compares positive integer `CFBundleVersion`, which must increase for every update, including successive RCs. Use `scripts/bump-version.sh X.Y.Z N` with a higher build number.
Feeds live on the separate `updates` branch at `https://raw.githubusercontent.com/<owner>/<repo>/updates/stable.xml` and `release-candidate.xml` in the same directory. Source branches and release tags are not modified by feed publication.
RC entries carry the `release-candidate` channel marker; stable clients do not subscribe to it. Switching an installed RC to stable requires a deliberate installation/channel change.
Feed publishing rejects lower or conflicting build numbers and uses non-force updates; concurrent changes fail without overwriting another channel. Before first publication the feed may not exist; a missing or unavailable feed must not prevent use of the installed App.

## Persistent signing configuration

The update Ed25519 key is separate from Apple Developer ID. Keep one persistent
32-byte seed for published updates. The existing key utility can generate that
format; its development name does not make generated keys temporary:

```bash
swift scripts/generate-development-update-key.swift /absolute/private/path/update-key
```

The file must not already exist. The utility writes it with mode 0600 and prints
only its public key. Store that public key as GitHub repository variable
`IRONMLX_UPDATE_PUBLIC_ED_KEY`, and the seed file's contents as secret
`IRONMLX_UPDATE_PRIVATE_ED_KEY`. Retain the private seed outside the checkout;
changing the key after distribution requires a separate migration plan.

The publisher refuses to sign if the supplied seed does not match the public
key embedded in the App. No persistent key or GitHub setting is created by
running the source tests.

For a signed stable Bundle, provide before building/signing:

```bash
IRONMLX_UPDATE_CHANNEL=stable \
IRONMLX_UPDATE_FEED_URL=https://raw.githubusercontent.com/OWNER/REPO/updates/stable.xml \
IRONMLX_UPDATE_PUBLIC_ED_KEY=PUBLIC_KEY \
scripts/build-app-bundle.sh
```

RC CI sets the corresponding RC URL and channel. Validation-only RC and stable workflows also require the persistent public key. Publishing additionally requires the matching private-key secret.

## Third-party material generation and verification

### Scope

The material generator creates a reproducible engineering inventory for the macOS arm64 Release
product actually embedded in `IronMLX.app`, rather than copying the complete
`Cargo.lock` graph. It includes the Rust Release binaries and their target
dependencies, external SwiftPM packages (Sparkle and ZIPFoundation), the pinned
MLX fork and native inputs used by its Release build, and third-party bundled
graphics and branding assets. It excludes system frameworks and model weights
downloaded separately by users; see [Model rights boundary](model-license-boundary.md).

These materials preserve notices and detect dependency drift. They are not legal
advice or public-distribution authorization.

### Inputs and generated files

- `Cargo.lock`, product manifests, and `about.toml`;
- `ironmlx-app/Package.swift`;
- the pinned MLX fork commit in `scripts/release-config.sh`;
- locked native dependency and bundled-asset manifests with SHA-256 values;
- `third-party-inventory.json`;
- `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/`.
- `SBOM.cdx.json` (CycloneDX 1.6, generated deterministically from the same
  inventory).

The generator records the exact MLX fork commit, upstream repository and base
revision, and validates native archives, bundled files, and license hashes.

### Update and verification

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
CARGO_ABOUT="$(command -v cargo-about)" scripts/update-third-party-materials.sh
scripts/verify-third-party-materials.sh
```

After dependency changes, review the complete diff of all generated materials;
updating a hash alone is not an acceptable workaround. CI regenerates materials
from the actual App-build inputs and compares them byte-for-byte.

### App and archives

Release builds copy these materials to:

```text
IronMLX.app/Contents/Resources/Legal/
```

The App menu exposes **Third-Party Notices…**. Release Bundles include the
project `LICENSE`, `NOTICE`, `SBOM.cdx.json`, and the third-party materials in
`Contents/Resources/Legal/`. RC and stable archives also include these
materials in the ZIP root and in `Documentation/` inside the DMG. The DMG root
also contains the `Applications` shortcut. `scripts/release-archives.py`
extracts the ZIP, mounts the DMG, and verifies the archived materials against
the source tree.

### Model-weight exclusion checks

Every App, DMG, and ZIP intended for distribution must pass the release script's
model-distribution boundary check, which rejects common model-weight files. This
check does not replace the user's upstream license review or change the user's
responsibility for downloaded models.

## Validation boundaries

`test_app_updates.py` checks channel/build policy and real Sparkle signatures, including key mismatch and tampering. `validate-update-installation.py` uses an isolated App and localhost HTTPS; its temporary keychain trust is removed afterward. It does not load production models or establish full production upgrade acceptance.
Local Apple/GitHub failure-path tests verify ordering and cleanup, not remote acceptance. Record actual notarization, Gatekeeper, installation, model recovery and data-preservation evidence against each candidate separately; these instructions do not assert completion for a particular release.
