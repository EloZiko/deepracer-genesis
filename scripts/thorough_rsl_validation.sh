#!/usr/bin/env bash
# Thorough validation that the rsl_rl VecEnv path is crash-immune before deleting
# the TorchRL front-end. Three stages, sequential (no cross-stage GPU contention):
#   1. POSITIVE CONTROL: the TorchRL front-end must still crash this session
#      (else a clean rsl_rl result proves nothing).
#   2. BREADTH: 6 batches x 8 concurrent rsl-rl PPO procs (150k steps) = 48 procs.
#   3. DEPTH:  4 concurrent rsl-rl PPO procs at 1M steps (catch late crashes).
set -u
cd /home/lunav0/.SecretProjects/deepracer-genesis
mkdir -p logs/thorough runs/thorough
PY=.venv/bin/python
CRASH='CUDA_ERROR_ILLEGAL_ADDRESS|illegal memory access'
count() { grep -lE "$CRASH" "$@" 2>/dev/null | wc -l; }

echo "===== STAGE 1: POSITIVE CONTROL (TorchRL front-end, 6 concurrent, 200k) ====="
for s in 0 1 2 3 4 5; do
  $PY scripts/seed_sweep_progress.py --seed $s --steps 200000 --root runs/thorough/ctrl \
    > logs/thorough/ctrl_$s.log 2>&1 &
done
wait
echo "RESULT control_torchrl: crashed=$(count logs/thorough/ctrl_*.log)/6"

echo "===== STAGE 2: rsl-rl PPO BREADTH (6 x 8-concurrent x 150k steps) ====="
for b in 1 2 3 4 5 6; do
  for s in $(seq 0 7); do
    $PY -m deepracer_genesis.train -B 16 --max_iterations 390 --randomize --seed $s \
      --exp_name thb${b}_$s > logs/thorough/ppo_b${b}_$s.log 2>&1 &
  done
  wait
  echo "RESULT ppo_breadth_batch$b: crashed=$(count logs/thorough/ppo_b${b}_*.log)/8  (running total $(count logs/thorough/ppo_b*.log))"
done

echo "===== STAGE 3: rsl-rl PPO DEPTH (4 concurrent x 1M steps) ====="
for s in 0 1 2 3; do
  $PY -m deepracer_genesis.train -B 16 --max_iterations 2604 --randomize --seed $s \
    --exp_name thlong_$s > logs/thorough/ppo_long_$s.log 2>&1 &
done
wait
echo "RESULT ppo_depth_1Mx4: crashed=$(count logs/thorough/ppo_long_*.log)/4"

echo "===== SUMMARY ====="
echo "TorchRL control : crashed=$(count logs/thorough/ctrl_*.log)/6"
echo "rsl-rl PPO total: crashed=$(count logs/thorough/ppo_*.log)/$(ls logs/thorough/ppo_*.log 2>/dev/null | wc -l)"
echo "THOROUGH_DONE"
