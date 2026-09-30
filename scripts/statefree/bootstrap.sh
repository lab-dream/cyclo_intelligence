#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
sub="$root/cyclo_brain/policy/lerobot/lerobot"
base=240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b
patch="$root/scripts/statefree/lerobot-image-only.patch"
if [[ ! -f "$sub/src/lerobot/policies/diffusion/modeling_diffusion.py" ]]; then
    git -C "$root" submodule update --init cyclo_brain/policy/lerobot/lerobot
fi
actual=$(git -C "$sub" rev-parse HEAD)
[[ "$actual" == "$base" ]] || { echo "Expected LeRobot $base, found $actual" >&2; exit 1; }
if git -C "$sub" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Image-only patch already applied."
else
    git -C "$sub" apply --check "$patch"
    git -C "$sub" apply "$patch"
fi
if ! git -C "$sub" diff --no-ext-diff -- src/lerobot/policies/diffusion | cmp -s "$patch" -; then
    echo "Diffusion source differs from the pinned base plus the managed patch" >&2
    exit 1
fi
sha256sum "$sub/src/lerobot/policies/diffusion/"{configuration,modeling}_diffusion.py
