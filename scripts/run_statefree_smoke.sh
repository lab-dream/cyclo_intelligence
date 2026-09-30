#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
bash "$root/scripts/statefree/bootstrap.sh"
export PYTHONPATH="$root/cyclo_brain/policy/lerobot/lerobot/src:$root/cyclo_brain/policy/lerobot:$root/cyclo_brain/policy/common/runtime:$root/scripts/statefree${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled MUJOCO_GL=${MUJOCO_GL:-egl}
python_bin=${STATEFREE_PYTHON:-$root/../.statefree-venv/bin/python}
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
exec "$python_bin" "$root/scripts/statefree/pipeline.py" "$@"
