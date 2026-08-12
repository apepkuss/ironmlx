# IronMLX App Bundle 0.1.0

P0-1 produces one self-contained `IronMLX.app` for Apple Silicon `arm64` on
macOS 26.2 or newer. It does not support Intel or earlier macOS releases.

## Release build

The release builder requires a clean MLX checkout pinned to commit
`16dea39b545cd641310fdcfdfc6fc62bb141ddd7`. By default it uses the sibling
checkout at `../iron-rivals/mlx`; set `MLX_SRC` to another clean checkout of
that exact commit when needed.

```bash
scripts/build-app-bundle.sh
```

The command rebuilds MLX, `mlx.metallib`, the Rust helpers, and the Swift App
executable in an isolated build directory. It does not copy from an existing
`target/debug`, `target/release`, SwiftPM product directory, or external MLX
installation. The resulting bundle is `dist/IronMLX.app`.

The assembled layout is:

```text
IronMLX.app/Contents/
├── Info.plist
├── MacOS/IronMLX
├── Helpers/
│   ├── ironmlx
│   └── iron-bench
└── Resources/
    ├── mlx.metallib
    ├── dashboard2.html
    ├── AppIcon.icns
    ├── menubar-icon.png
    ├── menubar-icon@2x.png
    ├── logo.png
    ├── sidebar-logo@2x.png
    └── Legal/
        ├── THIRD_PARTY_NOTICES.md
        ├── third-party-inventory.json
        └── THIRD_PARTY_LICENSES/
```

The build is ad-hoc signed for local execution. Developer ID signing,
notarization, and DMG/PKG creation are outside P0-1.

The bundled helper shares the same per-user backend lock as direct
`ironmlx serve` CLI launches. Only one IronMLX backend may run for a macOS
user, regardless of App instance or port. When another backend owns the lock,
the App reports the conflict before creating its Dashboard or menu-bar UI,
does not enter automatic helper recovery, and exits after the alert is dismissed.
The App always starts the model-manager helper first, even when no local model
is configured; after the helper is healthy, the UI is shown and confirmed models
continue restoring asynchronously.

GitHub Actions development previews preserve this ad-hoc, non-notarized
boundary and label it explicitly in the App, archives, release title, and
release notes. Public preview packaging and publication are currently blocked
by `scripts/release-legal-gate.sh` until P0-8B reviews the generated third-party
materials, supplies an SBOM, and explicitly authorizes distribution. See
`docs/development-preview-release.md`.

## Static verification

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
```

This gate validates bundle contents, `arm64`, macOS 26.2 minimum versions,
system-only dynamic dependencies, the metallib AIR target, absence of symlinks,
absence of developer-home/Cargo fallback paths, exact bundled legal materials,
and the ad-hoc signature.

Real release acceptance additionally requires launching this exact artifact in
an environment without Rust, Xcode, an external MLX installation, or `MLX_*` /
`DYLD_LIBRARY_PATH`; then loading a real model and completing real inference.
The same artifact must be copied from the M5 Max builder to an M1 Pro running
macOS 26.2 or newer without rebuilding or replacing any bundled file.
