#!/usr/bin/env bash
# Isolate the mechanism of the sporadic CUDA_ERROR_ILLEGAL_ADDRESS by crash-rate.
# 4 concurrent short training runs = a reliable fast trigger (GPU async-alloc
# pressure). Each batch runs the SAME workload under a different env-var lever;
# the lever that drives crashes to zero identifies the mechanism.
set -u
cd /home/lunav0/.SecretProjects/deepracer-genesis
mkdir -p logs/lever runs/lever
STEPS="${STEPS:-150000}"
PY=.venv/bin/python

run_batch() {   # $1=label ; rest = VAR=VAL env assignments for the workload
  label="$1"; shift
  for s in 0 1 2 3; do
    env "$@" "$PY" scripts/seed_sweep_progress.py --seed "$s" --steps "$STEPS" \
      --root "runs/lever/$label" > "logs/lever/${label}_seed${s}.log" 2>&1 &
  done
  wait
  n=0; iters=""
  for s in 0 1 2 3; do
    log="logs/lever/${label}_seed${s}.log"
    if grep -qE "CUDA_ERROR_ILLEGAL_ADDRESS|AcceleratorError|illegal memory access" "$log"; then
      n=$((n+1))
      last=$(grep -oE "iter [0-9]+/" "$log" | tail -1 | tr -dc '0-9')
      call=$(grep -oE "while calling [a-z_]+" "$log" | tail -1)
      iters="$iters [seed$s@iter${last:-?} ${call#while calling }]"
    fi
  done
  echo "RESULT lever=$label  crashed=$n/4  ${iters}"
}

echo "=== STEPS=$STEPS, 4 concurrent procs per batch ==="
run_batch baseline
run_batch launch_blocking       CUDA_LAUNCH_BLOCKING=1
run_batch no_torch_caching      PYTORCH_NO_CUDA_MEMORY_CACHING=1
run_batch expandable_segments   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== DONE ==="
