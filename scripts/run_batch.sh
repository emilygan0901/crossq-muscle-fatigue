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

i=0
for job in "$@"; do
  cfg=${job%:*}; seed=${job##*:}
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n; done
  "$KIT_DIR/scripts/run_experiment.sh" "$cfg" "$seed" $(( i % NGPU )) &
  i=$(( i + 1 ))
  sleep 2   # distinct start times, so run ids never collide
done
wait
echo "All jobs finished. Summary in results/run_log.md"
