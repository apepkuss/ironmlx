#!/usr/bin/env bash
# Validate commit subjects in a Git range against the Conventional Commits shape.

set -euo pipefail

base_sha="${1:-}"
head_sha="${2:-}"
readonly ZERO_SHA="0000000000000000000000000000000000000000"
readonly SUBJECT_PATTERN='^[a-z][a-z0-9-]*(\([a-zA-Z0-9._/-]+\))?!?: [^[:space:]].*$'

fail() {
  echo "error: $*" >&2
  exit 1
}

resolve_new_branch_base() {
  local head="$1"
  local candidate
  local merge_base
  local distance
  local best_base=""
  local best_distance=""

  for candidate in refs/remotes/origin/dev refs/remotes/origin/main; do
    git show-ref --verify --quiet "$candidate" || continue
    merge_base="$(git merge-base "$head" "$candidate" 2>/dev/null || true)"
    [ -n "$merge_base" ] || continue
    distance="$(git rev-list --count "$merge_base..$head")"
    if [ -z "$best_distance" ] || [ "$distance" -lt "$best_distance" ]; then
      best_base="$merge_base"
      best_distance="$distance"
    fi
  done

  [ -n "$best_base" ] || \
    fail "cannot resolve a new branch base from origin/dev or origin/main"
  printf '%s\n' "$best_base"
}

[[ "$base_sha" =~ ^[0-9a-f]{40}$ ]] || fail "invalid base SHA: $base_sha"
[[ "$head_sha" =~ ^[0-9a-f]{40}$ ]] || fail "invalid head SHA: $head_sha"
git cat-file -e "$head_sha^{commit}" 2>/dev/null || fail "head commit is unavailable: $head_sha"

if [ "$base_sha" = "$ZERO_SHA" ]; then
  base_sha="$(resolve_new_branch_base "$head_sha")"
else
  git cat-file -e "$base_sha^{commit}" 2>/dev/null || fail "base commit is unavailable: $base_sha"
fi

invalid=0
checked=0
while IFS= read -r commit_sha; do
  [ -n "$commit_sha" ] || continue
  checked=$((checked + 1))
  subject="$(git show -s --format=%s "$commit_sha")"
  if [[ ! "$subject" =~ $SUBJECT_PATTERN ]]; then
    printf 'error: non-conventional commit %s: %s\n' "${commit_sha:0:12}" "$subject" >&2
    invalid=1
  fi
done < <(git rev-list --reverse "$base_sha..$head_sha")

[ "$invalid" -eq 0 ] || fail "commit subjects must match <type>[optional scope][!]: <description>"
echo "Conventional Commits verification passed for $checked commit(s)"
