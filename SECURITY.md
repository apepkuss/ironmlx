# Security policy

[简体中文](docs/zh-CN/security.md)

## Reporting a vulnerability

Please report suspected vulnerabilities through a [GitHub private security
advisory](https://github.com/apepkuss/ironmlx/security/advisories/new). Do not
open a public issue or include exploit details in a pull request.

Include, when safe to do so:

- the affected IronMLX version or immutable commit;
- macOS version and Apple Silicon model;
- a minimal reproduction and impact description;
- relevant endpoint, configuration, or release-artifact details.

Never include model weights, prompts, tool arguments, API keys, HF tokens,
Keychain data, Authorization headers, private certificates, or unredacted logs.
Use the Dashboard's **Export diagnostic information** only after reviewing the
archive; it is local-only and intentionally excludes request bodies and
credentials. See [diagnostic export](docs/diagnostic-bundle.md).

## Scope

Reports may cover the IronMLX App, Rust/MLX runtime integration, HTTP API,
model-download integrity checks, release scripts, and bundled assets. Issues in
an upstream model repository, model weights, Hugging Face/ModelScope services,
macOS, or third-party dependencies should also be reported to the relevant
upstream project when IronMLX is not the cause.

## Supported versions and disclosure

The current `dev` tip and the latest published release candidate are the
security-support baseline while 0.1.0 is unreleased. Maintainers will
acknowledge and triage reports when practical, prioritize critical impact, and
coordinate a fix and disclosure timeline with the reporter. There is no
guaranteed response or remediation SLA.

## Security boundary

Read [Network and image security boundary](docs/security-boundary.md) for the
supported LAN authentication, image limits, and stable security error codes.
