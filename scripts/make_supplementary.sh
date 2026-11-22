#!/usr/bin/env bash
# Build the supplementary zip from a tagged commit, so its contents match the tag cited in the paper.
#
#   scripts/make_supplementary.sh v1.0-submission     # -> supplementary_v1.0-submission.zip
set -euo pipefail
TAG=${1:?usage: scripts/make_supplementary.sh <tag>}
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "tag $TAG not found" >&2; exit 1; }
OUT="supplementary_${TAG}.zip"
git archive --format=zip --prefix="crossq-muscle-fatigue-${TAG}/" -o "$OUT" "$TAG"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) from $(git rev-parse --short "$TAG^{commit}")"
