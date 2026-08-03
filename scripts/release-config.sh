#!/usr/bin/env bash
# Shared immutable inputs for CI and development-preview builds.
# shellcheck disable=SC2034 # This file is sourced; consumers use these values.

readonly IRONMLX_MLX_REPOSITORY="https://github.com/apepkuss/mlx.git"
readonly IRONMLX_MLX_COMMIT="16dea39b545cd641310fdcfdfc6fc62bb141ddd7"
readonly IRONMLX_MLX_UPSTREAM_REPOSITORY="https://github.com/ml-explore/mlx.git"
readonly IRONMLX_MLX_UPSTREAM_REVISION="973e27f82ffe68dbd626cda31ba34997045d1eb7"
readonly IRONMLX_PREVIEW_WARNING_ZH="未使用 Developer ID 签名、未经 Apple 公证，仅供开发验证"
readonly IRONMLX_PREVIEW_WARNING_EN="Not signed with Developer ID, not notarized by Apple, for development validation only"

# Public binary distribution remains blocked until P0-8B supplies and reviews
# the required legal review and SBOM. P0-8A generates and packages engineering
# inventories, notices, and source license texts, but those outputs do not by
# themselves authorize distribution. Changing this flag is deliberately
# insufficient on its own: release-legal-gate.sh also checks every artifact.
readonly IRONMLX_PUBLIC_DISTRIBUTION_READY="false"
