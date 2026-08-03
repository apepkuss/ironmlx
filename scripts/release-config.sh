#!/usr/bin/env bash
# Shared immutable inputs for CI and development-preview builds.
# shellcheck disable=SC2034 # This file is sourced; consumers use these values.

readonly IRONMLX_MLX_REPOSITORY="https://github.com/apepkuss/mlx.git"
readonly IRONMLX_MLX_COMMIT="16dea39b545cd641310fdcfdfc6fc62bb141ddd7"
readonly IRONMLX_PREVIEW_WARNING_ZH="未使用 Developer ID 签名、未经 Apple 公证，仅供开发验证"
readonly IRONMLX_PREVIEW_WARNING_EN="Not signed with Developer ID, not notarized by Apple, for development validation only"
