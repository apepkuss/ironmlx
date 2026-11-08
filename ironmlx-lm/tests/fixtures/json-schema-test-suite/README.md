# JSON Schema conformance fixtures

These MIT-licensed inputs come from the official
[JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite),
pinned at `245a45abc90190c0bff04d97e6d2b541b4bb9f12`.

`draft2020-12/core` contains all 46 top-level Draft 2020-12 test files:
385 groups and 1,300 instances. `draft2020-12/remotes` contains the 79 reference
resources. `draft2020-12/optional` contains all 34 optional files, 79 groups and
1,074 instances. Inputs are unmodified; `draft2020-12/provenance.json` records each
SHA-256 digest. Reference resources are explicitly provided to the production
validator without network or filesystem retrieval by that validator.
The directory's Git whitespace attribute preserves upstream trailing spaces;
do not format these files or change their recorded digests.

Run the persistent gate with the project's MLX test environment configured:

```sh
cargo test -p ironmlx-lm --lib official_draft2020
```

The gate verifies fixture digests, compiles every schema, compares all instance
results, and checks every instance against a generic JSON token grammar with the
production semantic completion guard. Compile failures and unsupported schemas
fail the gate. Separate native-tool tests cover all six dialects, model-facing
envelopes, argument restoration, replay, reasoning, forks and rollback.

The optional gate checks all 1,074 instances with explicit format assertions,
including IDN email and hostnames. It also checks the default profile separately:
164 non-format instances keep their official results and the 910 format instances
are accepted as annotations, as their schemas contain only `$schema` and `format`.
The production default does not enable format assertions merely to pass tests.
Explicit format assertions are exposed by the low-level validator, not a new HTTP
request field. The two smaller files in this directory retain the earlier normalization
and numeric regressions; they are selected unchanged groups, not additional
claims of suite coverage. See `LICENSE` for the upstream license.
