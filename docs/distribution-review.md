# v0.1.0 distribution materials review

Reviewed on 2026-09-09 against `e1d1f99ad3926b7c17bd62ac38db170e8d28da11`
plus this review's uncommitted corrections. Scope is the default macOS arm64
Release product, not every feature combination or third-party model.
The [full review and artifact hashes](zh-CN/distribution-review.md) record the
findings and final release steps.

The regenerated inventory covers 271 Rust versions, five native components,
two Swift packages and four registered graphics: 282 CycloneDX 1.6 components
and 132 license/notice files.
Reproduction and SBOM checks passed. Selected license branches include MIT,
Apache-2.0, BSD-3-Clause, ISC, Zlib, MIT-0, Unicode-3.0,
CDLA-Permissive-2.0 and MPL-2.0. No LGPL/GPL/AGPL branch is selected in this
product graph; the broader `about.toml` allowlist is not the effective selection.
Compound AND requirements remain represented with all selected texts.

Corrections made:

- Added an explicit MPL-2.0 source-availability statement for option-ext 0.2.0.
  The downloaded [source archive](https://static.crates.io/crates/option-ext/option-ext-0.2.0.crate)
  is 7,345 bytes and hashes to
  `04744f49eae99ab78e0d5c0b603ab218f515ea8cfe5a456d7629ad883a3b6e7d`, matching Cargo.lock.
  See [Mozilla's distribution guidance, Q8](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).
- Preserved the inline Hugging Face/ModelScope graphics as requested. Located
  matching LobeHub implementations under MIT and added source/modified-artwork
  records, the license and geometry/count checks. Official project-use references
  are [Hugging Face brand assets](https://huggingface.co/brand) and
  [ModelScope logos](https://modelscope.cn/models/modelscope/logos). Marks identify
  download sources without implying affiliation or endorsement.

Model weights remain separate user downloads; software licensing and technical
support do not grant model rights. Existing archive gates reject common model
file types, but final candidate inspection remains necessary.

Project-owned code remains covered by the project's Apache-2.0 license; this
review does not introduce a separate ownership approval requirement.
`IRONMLX_PUBLIC_DISTRIBUTION_READY` remains false by the owner's instruction and
will be enabled only after all final acceptance steps, immediately before public
release. This deferred switch does not block completion of the materials review.
Actual signing, notarization, installation and upgrade acceptance remain separate
later steps.
