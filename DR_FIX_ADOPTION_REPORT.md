# DR CUDA-Crash Fix — Adoption & Empirical Findings

**Date:** 2026-07-27 · **Branch:** `debug/dr-cuda-illegal-access` (uncommitted) ·
**Author:** adoption pass following `../DR_CUDA_CRASH_REPORT.md`
**Stack tested:** genesis-world 1.2.1→**1.2.3**, quadrants 1.0.2→**1.1.1**,
torch 2.12.1+cu130, RTX 4060 Ti (SM 8.9), CUDA 13.3.

This report records what I verified while adopting the prior investigation
(`DR_CUDA_CRASH_REPORT.md`), including **one result that diverges from that
report's recommendation** and should change how DR is configured on this stack.

---

## TL;DR

1. **The prior report's root cause is correct and adopted:** the sporadic
   `CUDA_ERROR_ILLEGAL_ADDRESS` (and the NaN'd-weights "spinning cars") came from
   **quadrants 1.0.2 GPU-allocator memory-safety bugs**, fixed upstream in
   quadrants 1.1.1 / genesis 1.2.3. We upgraded. Strictly better; no downside.
2. **My earlier "Genesis↔torch cross-stream race" theory was WRONG** and is
   retracted here for the record. The prior report disproved it at the
   disassembly level (everything runs on stream 0; `torch.cuda.synchronize()` is
   device-wide and still crashed). The `qd.sync()`/step fences I had added were
   band-aids with no mechanism — they are now removed.
3. **New empirical finding (diverges from the prior report):** on 1.2.3,
   **per-episode physics DR still crashes** (~2 of 3 GPU runs), whereas
   **build-time (static per-env) DR runs clean** (full 1.8M frames, 0 crashes).
   The prior report's A/B validated *build-time* DR on 1.2.3; it *asserted*
   per-episode was safe again but did not directly test it. It isn't, here.
   → **We keep build-time (static) DR as the default.**
4. A separate, real repo bug (**progress-reward exploit** via `localize`
   waypoint teleport) is also fixed (clamp on `d_progress`); complementary to the
   prior report's mass-shift-clamp fix for the same "spinning" symptom.

---

## 1. Root cause (adopted from the prior report — confirmed correct)

quadrants 1.0.2 shipped three maintainer-confirmed allocator memory-safety bugs
(UnifiedAllocator chunk-tail overrun; CachingAllocator use-after-erase; DLPack
i32 byte-offset truncation), all fixed in quadrants 1.1.1. Corrupt a pointer →
illegal address; corrupt a persistent physics field → permanent NaN. GPU-only
(no such allocator on CPU), DR-correlated (DR at reset is the densest
alloc/free churn), crash frame in quadrants' `mem_free_async_impl`. The prior
report's disassembly evidence and A/B (1.0.2 died at 1.07M/1.23M/1.43M frames;
1.2.3 passed 2.3M+) are convincing and match everything observed here.

**Fix = upgrade.** `genesis-world>=1.2.3` (pins quadrants 1.1.1). Done in
`pyproject.toml` + `uv.lock`; runtime-verified `genesis 1.2.3 / quadrants 1.1.1`.

---

## 2. NEW FINDING: per-episode DR still crashes on 1.2.3; build-time DR is stable

The prior report recommends reverting physics DR to per-episode ("officially
supported, safe again on 1.2.3"). Its A/B, however, ran **build-time** DR
(commit d017a6b had moved physics DR to build-time-only). I A/B'd the two
directly on 1.2.3. Same config each time —
`FeatureEnvironment(num_envs=16) >> DomainRandomizationPhysics() >> VectorPolicy`,
GPU, TorchRL front-end, no fences — runtime-verified genesis 1.2.3 + quadrants
1.1.1:

| DR timing | run | budget | result |
|---|---|---|---|
| **build-time (static)** | 1 | 1.8M | **completed, 0 crashes** (rew_progress 434 @ iter 3991) |
| per-episode | 1 | 1.8M | **crashed @ iter 21** (~8k frames), `set_dofs_kv` → illegal address |
| per-episode | 2 | 600k | clean (completed) |
| per-episode | 3 | 1.8M | **crashed @ iter 571** (~219k frames), same signature |

**Per-episode: 2 of 3 crashed on 1.2.3.** The upgrade *reduces* the crash rate
(1.0.2 crashed reliably; 1.2.3 sometimes survives) but does not eliminate it for
per-episode DR. Build-time DR cleared the full danger zone (all three 1.0.2
crash points) with zero crashes.

**Caveat (honest):** n=3 for per-episode is a small sample and the crash is
sporadic, so "2/3" is not a precise rate. But build-time DR is *both* the
validated-stable config *and* strictly cheaper (writes bodies once, not every
reset), so defaulting to it costs nothing while the per-episode behavior on 1.2.3
remains uncertain.

**Decision:** default to **build-time (static per-env) DR** — each env keeps its
own randomized friction/mass/COM/gains/armature + camera mount for the whole run;
only spawn/direction (state) randomizes per episode. World-color DR stays per
reset (it is torch-only, no genesis setter). `set_dofs_armature` is a separate
per-run call anyway (it hides a full mass-matrix recompute per call).

---

## 3. Separate repo bug: progress-reward exploit (also fixed)

Independent of the CUDA crash, the "cars spin, reward goes permanently negative"
collapse had a second, reward-side contributor I found and fixed:

- `localize` picks the nearest waypoint by distance. The `reinvent_base` track
  (L≈17.7 m, 118 waypoints) passes within **2.0 m of itself** at a pinch
  (index-far waypoints, spatially close). A car there teleports its localized
  arclength → up to **~2.7 m of "progress" in one step** vs the **max_speed·dt ≈
  0.12 m** a car can physically move — below the 0.5·L lap-wrap threshold, so it
  slips through as huge spurious progress reward → PPO over-reacts to the outlier
  advantage → policy collapse (all envs, shared policy).
- **Fix:** clamp `d_progress` to the physical per-step bound. Extracted to a pure,
  tested `rules.lap_progress(prev, new, L, dir_sign, max_step)` (4 unit tests:
  normal step, finish-line wrap, pinch-jump clamp, reversed direction).

This is complementary to the prior report's **mass-shift clamp** (negative link
masses on light links → runaway velocities → NaN) and its **grad-norm guard**
(finite loss + inf gradient still bricked the weights). All three, plus the mdp
non-finite/huge-state force-reset guard, are kept as defense-in-depth.

---

## 4. Adoption performed (files on the branch)

**Upgrade**
- `pyproject.toml` — `genesis-world>=1.2.0` → `>=1.2.3`; added `pytest>=8` to the
  dev dependency-group (it was installed ad-hoc and `uv sync` pruned it).
- `uv.lock` — re-locked (`genesis-world 1.2.3`, `quadrants 1.1.1`, `gs-madrona 0.0.8`).

**Removed (band-aids with no mechanism on 1.2.3)**
- `randomization/physics.py` — the per-write `torch.cuda.synchronize()` +
  `qd.sync()` fences and the `_qd_sync` helper.
- `envs/base_env.py` — the per-control-step `torch.cuda.synchronize()` + `qd.sync()`
  fence and its `_qd_sync` helper.
- `experiment/spec.py` — the `view="gui"` + gpu + physics-DR `SpecError` guard
  (this also fixes the `test_examples` failure that guard caused).

**Kept / changed**
- `randomization/physics.py` — **mass-shift clamp** kept (per-link span ≤ 0.9×rest
  mass). Split out `randomize_armature` (per-run).
- `envs/base_env.py` — physics DR applied **once at build** (static per-env), not
  per reset (see §2); `reset_idx` no longer calls physics setters.
- `algorithms/ppo.py` / `lagrangian.py` — **gradient-norm guard** kept (skip the
  optimizer step on a non-finite pre-clip norm).
- `envs/mdp.py` — **non-finite/huge-state force-reset guard** kept.
- `envs/rules.py` + `envs/base_env.py` — **`lap_progress` clamp** (§3).
- `tests/test_backend_view.py` — the guard test now asserts gui+gpu+DR *builds*
  on both backends (guard removed).

**Result:** 118 tests pass on genesis 1.2.3.

---

## 5. Recommendation & open decisions

**Recommended config (current default):** genesis 1.2.3 + **build-time (static
per-env) physics DR** + the four hardening guards. Verified: 1.8M-frame GPU DR
run, 0 crashes, learning normally.

**Open decisions for the maintainer:**
1. **DR timing.** Keep build-time (evidence above), or expose a
   `physics_dr_mode: "static" | "per_episode"` opt-in so per-episode is available
   despite its sporadic crash on this stack? (Static learns from fixed per-env
   bodies vs resampled; both are valid DR. Static relies on enough parallel envs
   for diversity.)
2. **Longer per-episode validation.** If per-episode DR is important, run it many
   more times / to more frames on 1.2.3 to quantify the real crash rate before
   trusting it — n=3 here is small.
3. **Commit** the branch, and decide whether to fold this file + the per-episode
   finding into `DR_CUDA_CRASH_REPORT.md`.

---

## 6. Reproduction

```bash
# upgraded env
uv sync --all-extras --all-groups
python -c "import importlib.metadata as m; print(m.version('genesis-world'), m.version('quadrants'))"  # 1.2.3 1.1.1

# build-time DR (default) — should complete clean
python - <<'PY'
from deepracer_genesis.experiment import FeatureEnvironment, VectorPolicy, DomainRandomizationPhysics, run
spec = (FeatureEnvironment(num_envs=16) >> DomainRandomizationPhysics() >> VectorPolicy(keys=("state",))
        ).build(total_env_steps=1_800_000, eval_every_steps=0)
print(run(spec, root="runs/static").metrics["completion_rate"])
PY
# per-episode DR: to reproduce the residual crash, move randomize_physics(self, env_ids)
# back into base_env.reset_idx and run the same config a few times.
```
