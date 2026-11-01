#!/usr/bin/env bash
# Emit a validated CI range and file classification as GitHub step outputs.
set -euo pipefail

base_sha="${1:-}"
head_sha="${2:-}"
readonly ZERO_SHA="0000000000000000000000000000000000000000"

fail() { echo "error: $*" >&2; exit 1; }
[[ "$base_sha" =~ ^[0-9a-f]{40}$ ]] || fail "invalid base SHA: $base_sha"
[[ "$head_sha" =~ ^[0-9a-f]{40}$ ]] || fail "invalid head SHA: $head_sha"
git cat-file -e "$head_sha^{commit}" 2>/dev/null || fail "head commit is unavailable: $head_sha"

force_full=false
if [[ "$base_sha" = "$ZERO_SHA" ]] || ! git cat-file -e "$base_sha^{commit}" 2>/dev/null; then
  # A force push can make the old tip unreachable from a fresh checkout. Use a
  # reachable mainline ancestor for commit validation, but require full quality
  # checks: its tree diff cannot establish what changed from the missing tip.
  best_base=""
  best_distance=""
  for candidate in refs/remotes/origin/main refs/remotes/origin/dev; do
    git show-ref --verify --quiet "$candidate" || continue
    ancestor="$(git merge-base "$head_sha" "$candidate" 2>/dev/null || true)"
    [[ -n "$ancestor" ]] || continue
    distance="$(git rev-list --count "$ancestor..$head_sha")"
    [[ "$distance" -gt 0 ]] || continue
    if [[ -z "$best_distance" || "$distance" -lt "$best_distance" ]]; then
      best_base="$ancestor"
      best_distance="$distance"
    fi
  done
  [[ -n "$best_base" ]] || fail "cannot resolve a nonempty range from main/dev; refusing to skip checks"
  echo "Base $base_sha is unavailable or new; using $best_base and requiring full quality checks" >&2
  base_sha="$best_base"
  force_full=true
fi

# Evaluate the diff before iteration: errors inside process substitution would
# otherwise be swallowed and misclassify an unavailable range as docs-only.
# Classify both sides of moves: renaming a shared/native input into website/
# must not hide its removal from the native build.
changed_files="$(git diff --name-only --no-renames "$base_sha" "$head_sha")" || fail "cannot compare CI commits"
docs_changed=false
website_changed=false
code_changed=false
if [[ "$force_full" = true ]]; then
  docs_changed=true
  website_changed=true
  code_changed=true
else
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ "$path" == README.md || "$path" == README.zh-CN.md || "$path" == docs/* || "$path" == *.md ]]; then
      docs_changed=true
    fi
    case "$path" in
      # This workflow only deploys the static website, not native artifacts.
      website/*|scripts/build-website-docs.py|scripts/build-website-release-metadata.py|scripts/tests/test_website_docs.py|.github/workflows/website-preview.yml)
        website_changed=true
        ;;
      # Shared CI/repository policy must exercise every quality job.
      .github/workflows/*|scripts/resolve-ci-change-range.sh|scripts/tests/test_ci_change_range.py|.gitignore|.gitattributes|.gitmodules)
        docs_changed=true
        website_changed=true
        code_changed=true
        ;;
      # The catalogue is consumed by the App and the docs generator. Release
      # summaries/templates are shared by the website and native publication.
      ironmlx-app/Sources/IronMLXAppCore/Resources/supported-models.json|scripts/build-supported-models-docs.py|VERSION|.github/release-notes-template.md|scripts/render-release-notes.py|scripts/tests/test_render_release_notes.py)
        website_changed=true
        code_changed=true
        ;;
      docs/*|*.md)
        website_changed=true
        ;;
      *)
        # Unknown non-website changes retain the existing native quality gate.
        code_changed=true
        ;;
    esac
  done <<< "$changed_files"
fi
printf 'base_sha=%s\nhead_sha=%s\ndocs_changed=%s\nwebsite_changed=%s\ncode_changed=%s\n' \
  "$base_sha" "$head_sha" "$docs_changed" "$website_changed" "$code_changed"
