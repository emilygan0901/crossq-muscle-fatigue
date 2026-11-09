#!/usr/bin/env bash
# Run several config:seed jobs, spreading them over the available GPUs.
#
#   scripts/run_batch.sh configs/repro/walker2d_sac.cfg:1 configs/repro/walker2d_crossq.cfg:1 ...
#
# PER_GPU (default 1) sets how many runs share one GPU at the same time.
set -uo pipefail
KIT_DIR=$(cd "$(dirname "$0")/.." && pwd)

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] || NGPU=1
SLOTS=$(( NGPU * ${PER_GPU:-1} ))
echo "GPUs: $NGPU, parallel runs: $SLOTS, jobs: $#"

# Each slot is pinned to one GPU; a new job goes to the first free slot, so a GPU is never
# given a second run while another GPU is idle (jobs can have different lengths).
declare -a PIDS
for ((k=0; k<SLOTS; k++)); do PIDS[$k]=0; done
for job in "$@"; do
  cfg=${job%:*}; seed=${job##*:}
  while :; do
    free=-1
    for ((k=0; k<SLOTS; k++)); do
      if [ "${PIDS[$k]}" = 0 ] || ! kill -0 "${PIDS[$k]}" 2>/dev/null; then free=$k; break; fi
    done
    [ "$free" -ge 0 ] && break
    wait -n
  done
  gpu=$(( free % NGPU ))
  echo "start $cfg seed $seed on GPU $gpu"
  "$KIT_DIR/scripts/run_experiment.sh" "$cfg" "$seed" "$gpu" &
  PIDS[$free]=$!
  sleep 2   # distinct start times, so run ids never collide
done
wait
echo "All jobs finished. Summary in results/run_log.md"
