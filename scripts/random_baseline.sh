#!/usr/bin/env bash
# Random-policy reference for the environment named in a config file.
#
#   scripts/random_baseline.sh <config.cfg> [n_episodes=100] [seed=0]
#   FATIGUE_FIXED=0.6 scripts/random_baseline.sh <drift config> 100 0    # drift tasks: fix the level
#
# Uses the same Python environment, ENV_ID and ENV_VARS as the training runs of that config.
# Saves results/random/<ENV_ID>[_fatigue<level>].npz and .txt, and prints the mean return.
set -euo pipefail
CONFIG=$(realpath "$1"); N=${2:-100}; SEED=${3:-0}
KIT_DIR=$(cd "$(dirname "$0")/.." && pwd)
FIXED=${FATIGUE_FIXED:-}
# shellcheck disable=SC1090
source "$CONFIG"
if [[ "$ENV_ID" == *Drift* && -z "$FIXED" ]]; then
  echo "Drift task: set FATIGUE_FIXED=<level> (a random policy has no training clock)." >&2; exit 1
fi
OUT_DIR="$KIT_DIR/results/random"; mkdir -p "$OUT_DIR"
TAG="${ENV_ID}${FIXED:+_fatigue$FIXED}"
# shellcheck disable=SC2086
env JAX_PLATFORMS=cpu PYTHONPATH="$KIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" ${ENV_VARS:-} \
    FATIGUE_FIXED="$FIXED" RUN_SEED="$SEED" \
    "$PYTHON" "$KIT_DIR/scripts/random_baseline.py" "$ENV_ID" "$N" "$SEED" "$OUT_DIR/$TAG.npz" \
    2>/dev/null | tee "$OUT_DIR/$TAG.txt"
{
  echo "config: ${CONFIG#"$KIT_DIR"/}"
  echo "fatigue_fixed: ${FIXED:-n/a}"
  echo "python: $("$PYTHON" --version 2>&1)"
  echo "packages: $("$PYTHON" -c 'import mujoco, gymnasium; print("mujoco", mujoco.__version__, "gymnasium", gymnasium.__version__)' 2>/dev/null)"
  echo "date_utc: $(date -u +%FT%TZ)"
} >> "$OUT_DIR/$TAG.txt"
