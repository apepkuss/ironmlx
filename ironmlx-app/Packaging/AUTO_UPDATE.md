# IronMLX automatic updates

Phase one integrates Sparkle 2.9.6 for development validation. It does not
authorize public distribution and does not configure the stable channel.

## Channel isolation

The tracked `Info.plist` contains no feed URL or update public key. A normal
build therefore keeps automatic updates disabled. The Bundle builder accepts
only these phase-one values:

```bash
IRONMLX_UPDATE_CHANNEL=development \
IRONMLX_UPDATE_FEED_URL=https://127.0.0.1:18443/appcast.xml \
IRONMLX_UPDATE_PUBLIC_ED_KEY='<development-public-key>' \
IRONMLX_APP_BUILD_NUMBER=2 \
scripts/build-app-bundle.sh
```

The development channel requires loopback HTTPS, a non-empty EdDSA public
key, signed feeds, and update verification before extraction. The builder and
the App both reject a `stable` channel in phase one. Production feed and key
injection are intentionally deferred until Developer ID signing and
notarization are available.

Never commit a private EdDSA key. The development validator creates an
ephemeral key under a temporary directory with mode `0600` and deletes it when
the validation exits.

## Development validation

Install the pinned inventory tool once in a repository-local directory:

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about \
  --root .build/release-tools
```

Then run:

```bash
scripts/validate-development-auto-update.sh
```

The validator builds ad-hoc build 1 and build 2, creates a signed appcast,
serves it over loopback HTTPS, and exercises Sparkle's automatic download and
install-on-quit flow. It also verifies that a modified archive, a modified
signed feed, and an offline feed cannot change the installed App.

macOS asks for permission to temporarily trust the generated loopback TLS
certificate. The script removes that exact certificate by fingerprint on exit.
No production key, feed, certificate, signing identity, or update artifact is
created by this workflow.

## Runtime behavior

The menu item delegates to `SPUStandardUpdaterController`. Automatic checks
and downloads are enabled only in development-configured builds. When Sparkle
requests termination for installation, the existing App termination path
cancels model downloads, waits for the backend to stop, and only then replies
that termination may continue.

`--ironmlx-development-update-test-marker <absolute-path>` is an automated
validation hook. It is honored only when the Bundle contains a valid
development update configuration; it starts a background check and records
the update-ready or error result for the validation script.
