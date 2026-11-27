#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
input=$(realpath "${1:?usage: build-image.sh ARTIFACT_DIR IMAGE_TAG}")
image=${2:?set the output image tag}
: "${BASE_IMAGE:?set BASE_IMAGE to the authorized r148 registry reference or local tag}"
test "$(docker image inspect -f '{{.Id}}' "$BASE_IMAGE")" = sha256:9722b0304456438355302be279e95e1f2a5e0aa08401d16d19ba35a6a4821a3d
(cd "$input" && sha256sum -c "$here/runtime.sha256")
context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
cp "$here/Dockerfile" "$here/check-runtime.sh" "$input/runtime.tar.gz" "$input/extras.tar.gz" "$context/"
docker build --network=none --build-arg "BASE=$BASE_IMAGE" -t "$image" "$context"
