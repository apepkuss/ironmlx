#!/usr/bin/env bash
# Shared immutable inputs for CI and release builds.
# shellcheck disable=SC2034 # This file is sourced; consumers use these values.

readonly IRONMLX_MLX_REPOSITORY="https://github.com/apepkuss/mlx.git"
readonly IRONMLX_MLX_COMMIT="8b39c1037825f3179dd5a04477565c1eedf4605e"
readonly IRONMLX_MLX_UPSTREAM_REPOSITORY="https://github.com/ml-explore/mlx.git"
readonly IRONMLX_MLX_UPSTREAM_REVISION="8a81722b1d71cac9b7dde47e56a438c4b529129b"

# Native build platform shared by the MLX and App Bundle builds.
readonly IRONMLX_MACOS_DEPLOYMENT_TARGET="26.4"
readonly IRONMLX_ARCHITECTURE="arm64"

# Distribution materials and scoped acceptance reviewed for the first RC.
# This enables packaging gates; public release still requires explicit dispatch.
readonly IRONMLX_PUBLIC_DISTRIBUTION_READY="true"
