#!/usr/bin/env bash
# Thorough validation of the rsl_rl_sac (SAC) path, mirroring the PPO campaign.
# Auto-waits for the PPO campaign to finish first (single GPU: no contention /
# confound). Uses rsl_rl_sac via PYTHONPATH (does not touch site-packages).
#   BREADTH: 4 batches x 6 concurrent SAC procs (200k steps) = 24 procs.
#   DEPTH:   2 concurrent SAC procs at 1M steps (late-crash check).
set -u
cd /home/lunav0/.SecretProjects/deepracer-genesis
PY=.venv/bin/python
export PYTHONPATH=/tmp/rsl_rl_sac
CRASH='CUDA_ERROR_ILLEGAL_ADDRESS|illegal memory access'
count() { grep -lE "$CRASH" "$@" 2>/dev/null | wc -l; }
mkdir -p logs/thorough_sac

echo "waiting for the PPO campaign (THOROUGH_DONE) before starting SAC..."
until grep -q THOROUGH_DONE logs/thorough/summary.log 2>/dev/null; do sleep 30; done
echo "PPO campaign finished; starting SAC thorough validation"

echo "===== SAC BREADTH (4 x 6-concurrent x 200k steps) ====="
for b in 1 2 3 4; do
  for s in 0 1 2 3 4 5; do
    $PY scripts/sac_validate.py --seed $s --iters 520 --randomize \
      > logs/thorough_sac/sac_b${b}_$s.log 2>&1 &
  done
  wait
  echo "RESULT sac_breadth_batch$b: crashed=$(count logs/thorough_sac/sac_b${b}_*.log)/6  (total $(count logs/thorough_sac/sac_b*.log))"
done

echo "===== SAC DEPTH (2 concurrent x 1M steps) ====="
for s in 0 1; do
  $PY scripts/sac_validate.py --seed $s --iters 2604 --randomize \
    > logs/thorough_sac/sac_long_$s.log 2>&1 &
done
wait
echo "RESULT sac_depth_1Mx2: crashed=$(count logs/thorough_sac/sac_long_*.log)/2"

echo "SAC total: crashed=$(count logs/thorough_sac/sac_*.log)/$(ls logs/thorough_sac/sac_*.log 2>/dev/null | wc -l)"
echo "SAC_THOROUGH_DONE"
