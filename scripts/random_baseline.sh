#!/usr/bin/env bash
# Random-policy reference for the environment named in a config file.
#
#   scripts/random_baseline.sh <config.cfg> [n_episodes=100] [seed=0]
#
# Uses the same Python environment and ENV_ID as the training runs of that config.
# Saves results/random/<ENV_ID>.npz and a meta file, and prints the mean return.
set -euo pipefail
CONFIG=$(realpath "$1"); N=${2:-100}; SEED=${3:-0}
KIT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1090
source "$CONFIG"
OUT_DIR="$KIT_DIR/results/random"; mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${ENV_ID}.npz"
JAX_PLATFORMS=cpu "$PYTHON" "$KIT_DIR/scripts/random_baseline.py" "$ENV_ID" "$N" "$SEED" "$OUT" \
  2>/dev/null | tee "$OUT_DIR/${ENV_ID}.txt"
{
  echo "config: ${CONFIG#"$KIT_DIR"/}"
  echo "python: $("$PYTHON" --version 2>&1)"
  echo "packages: $("$PYTHON" -c 'import mujoco, gymnasium; print("mujoco", mujoco.__version__, "gymnasium", gymnasium.__version__)' 2>/dev/null)"
  echo "date_utc: $(date -u +%FT%TZ)"
} >> "$OUT_DIR/${ENV_ID}.txt"
