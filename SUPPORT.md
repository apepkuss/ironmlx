# Support

[简体中文](docs/zh-CN/support.md)

## Supported platform

IronMLX 0.1.0 targets Apple Silicon (`arm64`) on macOS 26.2 or later. Intel
Macs, older macOS versions, and unmodified third-party model runtimes are
outside the supported platform.

## Getting help

For a reproducible product problem, open a [GitHub issue](https://github.com/apepkuss/ironmlx/issues)
with:

- the IronMLX version or immutable commit;
- macOS version and Apple Silicon model;
- the model repository and immutable revision, without uploading weights;
- the exact operation, safe error text, and a minimal reproduction;
- whether the problem occurs in a source build or development preview.

Do not post credentials, prompts, tool arguments, model weights, private URLs,
or unredacted logs. Security-sensitive reports belong in [SECURITY.md](SECURITY.md),
not in a public issue.

## Diagnostics and privacy

Use Dashboard **Export diagnostic information** when a maintainer asks for
runtime context. The export is created locally, does not upload data, and is
bounded and redacted. Review the ZIP before sharing it; never substitute raw
logs or the original configuration. See [diagnostic export](docs/diagnostic-bundle.md)
and [privacy boundary](docs/privacy.md).

## Scope and expectations

Support covers the documented App, local runtime, loopback API, authenticated
LAN mode, model-management flows, and release artifacts on the supported
platform. Experimental models, unsupported architectures, upstream service
outages, and local modifications may be investigated but are not guaranteed to
work. There is no guaranteed response or resolution SLA; reproducible security
issues are handled through [SECURITY.md](SECURITY.md).
