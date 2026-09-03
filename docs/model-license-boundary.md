# Model rights boundary

IronMLX provides model search, download, snapshot integrity verification, and
local loading. It does not own, transfer, or relicense any model rights. The
IronMLX software license does not apply to model weights, configuration files,
tokenizers, or other model-card content.

## User responsibility

Users choose the repository and intended use. Before using a model, users must
read the upstream model page (for example, Hugging Face or ModelScope),
including its license, gated-access terms, Acceptable Use Policy, commercial
restrictions, and redistribution rules. Users must provide credentials they are
authorized to use and comply with upstream terms and applicable law. This is
not legal advice.

IronMLX does not bypass upstream access controls, accept licenses or gated
terms on a user's behalf, or guarantee that a model is suitable for a particular
commercial or redistribution scenario. Technical loadability only means that
the runtime recognizes the format or architecture; it is not an authorization.

## Download and distribution boundary

- The App can search and download from a user-selected Hugging Face or ModelScope
  repository using credentials supplied by the user.
- Downloads resolve upstream references to an immutable revision and verify file
  sizes and SHA-256. These are transport/cache integrity checks, not license or
  use-compliance checks.
- IronMLX does not bundle model weights in the App, DMG, ZIP, or source release,
  host a model mirror, or repackage and redistribute third-party weights.
- The supported-models matrix describes verified runtime capability only. Users
  must return to the upstream model page to confirm rights and conditions.

## Release acceptance

Every App, DMG, and ZIP intended for distribution must pass the release script's
model-distribution boundary check, which rejects common model-weight files. This
check does not replace the user's upstream license review or change the user's
responsibility for downloaded models.
