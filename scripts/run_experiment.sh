#!/usr/bin/env bash
# Run ONE training run from a config file and record everything needed to reproduce it.
#
#   scripts/run_experiment.sh <config.cfg> <seed> [gpu_index]
#
# Creates results/raw/<run_id>/ containing:
#   config.cfg        copy of the config used
#   meta.txt          run id, env vars, exact command, code commit, Python, GPU, start/end time, exit code
#   code.diff         any changes to the code directory (empty = unmodified)
#   packages.txt      exact package versions
#   train.log         everything the training script printed
#   evaluations.npz   evaluation returns over training (the raw result)
#   eval_logs/, logs/ the training script's own output folders
# and appends one line to results/run_log.md.
set -uo pipefail

CONFIG=$(realpath "$1"); SEED=$2; GPU=${3:-0}
KIT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1090
source "$CONFIG"

RESULTS_ROOT=${RESULTS_ROOT:-$KIT_DIR/results/raw}
RUN_ID="${NAME}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$RESULTS_ROOT/$RUN_ID"
mkdir -p "$OUT"
cp "$CONFIG" "$OUT/config.cfg"

# shellcheck disable=SC2206
CMD=( "$PYTHON" "$CODE_DIR/train.py" -env "$ENV_ID" -seed "$SEED"
      -total_timesteps "$TOTAL_STEPS" -log_freq "$LOG_FREQ" -wandb_mode disabled $TRAIN_ARGS )

{
  echo "run_id: $RUN_ID"
  echo "config: ${CONFIG#"$KIT_DIR"/}"
  echo "seed: $SEED"
  echo "env_vars: RUN_SEED=$SEED ${ENV_VARS:-}"
  echo "command: ${CMD[*]}"
  echo "code_dir: $CODE_DIR"
  echo "code_commit: $(git -C "$CODE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "code_files_changed: $(git -C "$CODE_DIR" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
  echo "kit_commit: $(git -C "$KIT_DIR" rev-parse --verify -q HEAD 2>/dev/null || echo uncommitted)"
  echo "kit_files_changed: $(git -C "$KIT_DIR" status --porcelain --untracked-files=no -- configs scripts 2>/dev/null | wc -l)"
  echo "python: $("$PYTHON" --version 2>&1)"
  echo "jax_platform: ${JAX_PLATFORMS:-cuda}"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" 2>/dev/null || echo none)"
  echo "host: $(hostname)"
  echo "start_utc: $(date -u +%FT%TZ)"
} > "$OUT/meta.txt"
git -C "$CODE_DIR" diff > "$OUT/code.diff" 2>/dev/null
{ uv pip freeze --python "$PYTHON" 2>/dev/null || "$PYTHON" -m pip freeze; } > "$OUT/packages.txt"

echo "[start] $RUN_ID (gpu $GPU)"
start=$(date +%s)
# ENV_VARS (optional, from the config) are extra settings read by our own code, e.g. the
# fatigue schedule. RUN_SEED lets src/fatigue_drift.py seed MyoSuite. src/ is put on PYTHONPATH.
# shellcheck disable=SC2086
( cd "$OUT" && env CUDA_VISIBLE_DEVICES="$GPU" MPLBACKEND=Agg JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}" \
    RUN_SEED="$SEED" PYTHONPATH="$KIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" ${ENV_VARS:-} \
    "${CMD[@]}" > train.log 2>&1 )
rc=$?
minutes=$(( ($(date +%s) - start + 30) / 60 ))

npz=$(find "$OUT/eval_logs" -path "*/eval/evaluations.npz" 2>/dev/null | head -1)
[ -n "$npz" ] && cp "$npz" "$OUT/evaluations.npz"
{
  echo "end_utc: $(date -u +%FT%TZ)"
  echo "wall_minutes: $minutes"
  echo "exit_code: $rc"
  echo "evaluations_npz: $([ -n "$npz" ] && echo yes || echo MISSING)"
} >> "$OUT/meta.txt"

LOG="$(dirname "$RESULTS_ROOT")/run_log.md"
[ -f "$LOG" ] || printf '| run_id | config | seed | exit | minutes | notes |\n|---|---|---|---|---|---|\n' > "$LOG"
echo "| $RUN_ID | ${CONFIG#"$KIT_DIR"/} | $SEED | $rc | $minutes | |" >> "$LOG"

echo "[done]  $RUN_ID: exit $rc, $minutes min, results $([ -n "$npz" ] && echo saved || echo MISSING)"
exit $rc
