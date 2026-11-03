#!/usr/bin/env bash
# Shared immutable inputs for CI and release builds.
# shellcheck disable=SC2034 # This file is sourced; consumers use these values.

readonly IRONMLX_MLX_REPOSITORY="https://github.com/apepkuss/mlx.git"
readonly IRONMLX_MLX_COMMIT="65383d68c984ce4ef49f4b600820f3b9064c5a82"
readonly IRONMLX_MLX_UPSTREAM_REPOSITORY="https://github.com/ml-explore/mlx.git"
readonly IRONMLX_MLX_UPSTREAM_REVISION="a8f66b5901791a38ff527f249e1f1a84f46c03b6"

# Distribution materials and scoped acceptance reviewed for the first RC.
# This enables packaging gates; public release still requires explicit dispatch.
readonly IRONMLX_PUBLIC_DISTRIBUTION_READY="true"
