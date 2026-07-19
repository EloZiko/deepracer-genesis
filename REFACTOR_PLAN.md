# Refactor plan: decompose the env, drop the cache, pass code as parameters

Goal: make the codebase easier to follow by (A) **deleting the run-cache and
automatic ablation**, (B) **decomposing the 681-line `DeepRacerEnv`** into
focused modules, (C) replacing the four string **registries with direct
parameter passing**, and (D) giving the config a real **type surface**. No
change to the GPU-hot path.

Decisions locked with the user:

- **No cache.** Ablations must retrain from scratch; no automatic ablation grid.
  Delete the content-hash identity machinery entirely.
- Decompose `DeepRacerEnv` into `scene.py`, `entities.py`, `renderers.py`,
  `track.py` (geometry only), `rules.py` (predicates), `mdp.py` (R/T + action +
  obs), and a `BaseDeepRacerEnv` + `VectorDeepRacerEnv` + `VisionDeepRacerEnv`.
- Renderers are a **strategy** with two roles: observation-renderer (batched,
  GPU) and view-renderer (spectator/top-down via the **CPU rasterizer**),
  orthogonal to each other.
- `rules.py` = world-fact predicates; `mdp.py` = reward + termination + action
  map + obs assembly (consumes `rules.py`).
- Config typing: runtime/GPU dicts → grouped `TypedDict`; build-time config →
  frozen `dataclass`; pluggable behavior → `Protocol` / type alias.
- Pass reward fn / feature set / algorithm / experiment **as parameters**, not
  via registries.

---

## How the two step()s relate (context for the decomposition)

`TorchRLDeepRacerEnv` does **not** run alongside `DeepRacerEnv`; it **wraps** it.
Only one front-end is live per run:

- **rsl-rl** (`train.py`): `OnPolicyRunner` calls `DeepRacerEnv.step()` directly
  (the wrapper is never imported).
- **TorchRL** (`experiment/trainer.py`): the collector drives
  `TorchRLDeepRacerEnv._step()`, which on `torchrl_env.py:65` calls
  `self.sim.step()` — `self.sim` **is** the `DeepRacerEnv`.

The only coupling is the **pre-reset snapshot**: `DeepRacerEnv.step` auto-resets
done envs inside itself, so before resetting it stashes `self.step_info`
(`deepracer_env.py:406-413`); the wrapper reads `sim.step_info`
(`torchrl_env.py:66-69`) to split `terminated` (crash/offtrack, bootstrap
killed) vs `truncated` (timeout, bootstrap kept), backed by
`_torchrl_native_autoreset = True`. Keep this contract intact through the
decomposition — `mdp.py` should own producing `step_info`.

---

## Part A — delete the run-cache and automatic ablation

Do this FIRST: it removes the only reason the registries needed string
identities, so Part C becomes trivial.

Remove / simplify:

- `ExperimentSpec.id()` and the sha1 machinery (`spec.py:127-136`).
- Hash-based `run_dir` (`spec.py:138-141`) → `f"{root}/{group}/{variant}-{seed}"`
  (optionally a monotone suffix if you want to keep old runs; no timestamp in the
  identity — it's just a folder name now).
- The cache-hit short-circuit in `Trainer.fit` (`trainer.py:86-90`) and the
  `force=` plumbing through `Trainer.fit`, `run()`, `Experiment.run`,
  `__main__.py`.
- Automatic ablation grid in `experiment/ablation.py` (keep a *manual* helper if
  you still want to launch a list of variants, but each is a fresh run).
- `EvalRecord.spec_id` field (`trainer.py:191`), the `--force` CLI flag.
- The footgun note in `rewards.py:26-29` (moot once names aren't hashed).

Keep: `EvalRecord`, `spec.to_dict()`/`spec.json` dump (still useful as a run
record and for mlflow params), tensorboard/mlflow logging.

Tests to update: `tests/test_ablation.py`, `tests/test_report.py`,
`tests/test_experiment_spec.py` (drop `id()`/cache assertions).

---

## Part B — decompose `DeepRacerEnv`

Target module layout under `envs/`:

```
scene.py       # assemble the gs.Scene: ground plane, lights, background, build()
entities.py    # Car (URDF + DOF control + kp/kv) and TrackEntity (mesh morphs)
renderers.py   # ObsRenderer strategy (Null/Madrona/Nyx) + ViewRenderer (rasterizer/CPU)
track.py       # ONLY geometry: waypoint load, mesh paths, Track / MultiTrack
rules.py       # predicates: is_off_track, is_flipped, lap/progress, spawn_pose
mdp.py         # map_action, assemble_obs, compute_reward, assemble_done, step_info
base_env.py    # BaseDeepRacerEnv: step / reset_idx / _post_physics orchestration
vector_env.py  # VectorDeepRacerEnv(BaseDeepRacerEnv)  — state obs only
vision_env.py  # VisionDeepRacerEnv(BaseDeepRacerEnv)  — + camera obs
torchrl_env.py # unchanged adapter (delegates to whichever env instance)
```

### B.1 `entities.py` — a `Car` object (biggest single win)

Encapsulate every `self.car.*` Genesis call + the hand-wired control behind one
class. Pulls ~120 lines out of the env (kp/kv setup, wheel radius, DOF
bookkeeping, per-step control, reset).

```python
class Car:
    def __init__(self, scene, urdf_path, cfg): ...     # add_entity(URDF morph)
    def configure_control(self, cfg): ...              # set_dofs_kp/kv + force_range  (deepracer_env.py:264-276)
    def apply(self, steer, speed, wheel_radius): ...   # control_dofs_position/velocity (389-395)
    def reset_pose(self, qpos, env_ids): ...           # set_qpos + zero dofs           (624-632)
    def state(self): ...                               # pos/quat/vel/ang               (436-449)
    @property
    def wheel_radius(self): ...                        # measured from STL              (278-281)
```
`TrackEntity` similarly wraps the track morph(s) (`deepracer_env.py:155-160`).
Pattern: plain encapsulation.

### B.2 `scene.py` — scene assembly

A `build_scene(cfg, entities, renderers) -> gs.Scene` (or a small `SceneBuilder`)
that owns: `gs.Scene(...)` options, the green ground plane
(`deepracer_env.py:127-131`), lights (`221-225` / Nyx sun `188-207`), background
color, and the final `scene.build(n_envs=...)`. It wires the car + track
entities and the renderers' scene-side setup, respecting Genesis's
create→add→build order. Pattern: builder.

### B.3 `renderers.py` — two strategy axes

```python
class ObsRenderer(Protocol):                 # policy-facing, batched
    def setup(self, scene, car, cfg): ...
    def obs(self) -> torch.Tensor | None: ...    # (N,3,H,W) in [0,1], or None

class NullRenderer(ObsRenderer): ...         # vector env: obs() -> None
class MadronaRenderer(ObsRenderer): ...      # BatchRenderer + world-color remap + rg_swap
class NyxRenderer(ObsRenderer): ...          # Nyx path-tracer sensors

class ViewRenderer(Protocol):                # human-facing, CPU rasterizer, mode-agnostic
    def setup(self, scene, track, cfg): ...
    def spectator(self) -> np.ndarray: ...       # all cars, one image (render_spectator)
    def topdown(self) -> torch.Tensor: ...       # per-env bird's-eye (render_topdown)

class RasterizerView(ViewRenderer): ...      # gs.renderers.Rasterizer(); works even in vector mode
class NullView(ViewRenderer): ...            # no debug views
```

This absorbs the ~150 lines of Madrona/Nyx/rasterizer branching now spread
across `_build_scene` (`92-93`, `162-247`) and `_post_physics` (`502-527`).
Crucially the **CPU rasterizer view is independent of the obs renderer**, so a
`VectorDeepRacerEnv` can still emit spectator videos. The env holds
`self.obs_renderer` + `self.view_renderer` and calls them; no `if nyx/if vision`
scattered through step.

### B.4 `track.py` (geometry only) + `rules.py` (predicates)

- `track.py` keeps `Track` / `MultiTrack` — waypoint loading, mesh/obj paths,
  `localize`, `lookahead`, `curvature_ahead`, `spawn_pose` geometry. Nothing
  about rewards or termination.
- `rules.py` gets the DeepRacer world-facts that today live inline in
  `_post_physics` / `_check_termination`:
  ```python
  def is_off_track(lateral, half_width, margin): ...      # deepracer_env.py:571
  def is_flipped(up_z): ...                                # 572
  def lap_progress(prev_m, new_m, total_len, dir_sign): ...# 461-470
  def spawn_pose(track, env_ids, cfg): ...                 # wraps track spawn + direction flip
  ```
  Pure functions over kinematics + track-frame; batched torch; no env state.

### B.5 `mdp.py` — the (S, A, R, T) interface

Owns the RL-facing transforms, consuming `rules.py`:
```python
def map_action(actions, cfg) -> (steer, speed): ...        # deepracer_env.py:388-392
def assemble_obs(env) -> TensorDict: ...                   # delegates to features.py + camera group
def compute_reward(env, reward_fn, scales) -> Tensor: ...  # 556-567 (reward_fn passed in, Part C)
def assemble_done(env, cfg) -> (done, step_info): ...      # 570-594 + the pre-reset snapshot
```
`compute_reward` / `assemble_done` call the `rules.py` predicates. `assemble_done`
produces `step_info` so the TorchRL adapter's contract is preserved.

### B.6 The env classes — template method + injected strategies

```python
class BaseDeepRacerEnv:
    """Owns step(), reset_idx(), _post_physics(); calls mdp/rules; holds a Car,
    a TrackEntity, an ObsRenderer and a ViewRenderer."""
    def get_observations(self): return TensorDict({"state": ..., **self._camera_group()})
    def _camera_group(self): return {}                     # hook

class VectorDeepRacerEnv(BaseDeepRacerEnv):
    obs_renderer = NullRenderer()                          # no camera key

class VisionDeepRacerEnv(BaseDeepRacerEnv):
    def _camera_group(self): return {"camera": self.obs_renderer.obs()}  # + camera_res/spec
```
`BaseDeepRacerEnv.step` keeps the exact sequence — `car.apply` → `scene.step ×
decimation` → `_post_physics` → `mdp.compute_reward` → `mdp.assemble_done` →
snapshot `step_info` → reset done envs — and the CUDA stream fences
(`deepracer_env.py:429-430`, `657-658`). The rsl-rl VecEnv tuple return and the
`extras["time_outs"]` stay identical so both front-ends keep working.

---

## Part C — registries → parameters (trivial once the cache is gone)

With no content-hash, nothing needs a hashable string. Pass the code directly.

- **Reward**: `RewardFn = Callable[[Env], dict[str, Tensor]]` (alias in
  `rewards.py`). Delete `REWARDS`/`register_reward`/`resolve_reward`. Env +
  `mdp.compute_reward` take the callable; `EnvSpec.reward: RewardFn = deepracer`.
- **Feature set**: keep the `FeatureSet` base; delete `FEATURE_SETS`/
  `register_feature_set`/`make_feature_set`. Pass the class;
  `EnvSpec.feature_set: type[FeatureSet] = ClassicFeatures`.
- **Algorithm**: `Algorithm` is already a `Protocol`. Delete `ALGORITHMS`/
  `register_algorithm`/`make_algorithm`. `AlgorithmSpec.cls: type[Algorithm] =
  PPO`; `trainer.py:98-100` does `spec.algorithm.cls().setup(builder)`. Replace
  `algo.kind == "ppo_lagrangian"` checks (`spec.py:245-254`, `trainer.py:115`)
  with a class capability flag, e.g. `PPOLagrangian.requires_cost = True`.
- **Experiments**: replace `__init_subclass__` auto-registration
  (`registry.py:95-100`) with an explicit `EXPERIMENTS = {...}` map that
  `experiments/__init__.py` builds and `run()`/`__main__` import. Keep the
  `Experiment` authoring base class; drop the implicit side-effect.

`spec.to_dict()` (for mlflow/records) serializes callables as
`fn.__qualname__` — display only, no hashing.

---

## Part D — type the config

- **Runtime env dict → grouped `TypedDict`** (`configs/schema.py`): `SimConfig`,
  `ActionMapConfig`, `ActuationConfig`, `SpawnTermConfig`, `ObsConfig`,
  `VisionConfig`, `RandConfig`, composed into `EnvConfig`. Stays a dict → zero
  perf change; nesting means `self.cfg["actuation"]["steer_kp"]` (mechanical
  churn across the env + `domain_rand.py` + `builder.sim_cfg`).
- **Build-time config → frozen `dataclass`**: the `ExperimentSpec` family already
  is; author `get_train_cfg`'s rsl-rl structure as dataclasses and `asdict()` at
  the single `OnPolicyRunner(...)` boundary (external lib still gets its dict).
- **Pluggable behavior → `Protocol`/alias**: `RewardFn`, `FeatureSet`,
  `Algorithm`, plus `ObsRenderer`/`ViewRenderer` from Part B.

---

## Part F — correctness & cleanup findings (fix during the decomposition)

Verified issues found while mapping the code; the decomposition is the natural
moment to fix them.

1. **The `FeatureSet` abstraction is disconnected from the env (correctness).**
   `deepracer_env.py` never references `feature_set`/`make_feature_set` — it
   hardcodes the classic vector inline (`state_buf`, `num_state_obs = 8 + 2 *
   lookahead_k`, `deepracer_env.py:320,481-497`) and never sets
   `self.feature_set`. But `builder.py:84` passes `cfg["feature_set"]`, the spec
   exposes `feature_set="classic"|"perception"`, and `rollout.py:199` reads
   `sim.feature_set`. Consequences: (a) `feature_set="perception"` is **silently
   ignored** — the whole sim2real perception vector in `features.py` is dead
   through the main env; (b) `rollout.py:199` is a **latent `AttributeError`**.
   Fix: `mdp.assemble_obs` instantiates the selected `FeatureSet` (passed as a
   class, Part C) and calls `.compute()`; the env exposes `self.feature_set`;
   `num_state_obs = feature_set.dim`. Delete the inline `state_buf` construction.
   Add a test that `perception` actually changes the observation width.

2. **`Track`'s single-track query methods are dead (cleanup).**
   `Track.localize/lookahead/spawn_pose` (`track.py:84-120`) are never called —
   `MultiTrack` reimplements them over padded batched tensors and only uses
   `Track` as a per-track attribute container. Fix: strip `Track` to a geometry
   container (or fold into `MultiTrack`); the query logic lives once in
   `MultiTrack`. Lands with the `track.py`-geometry-only split (Part B.4).

3. **Magic numbers vs. named constants (cleanup).** `features.py` defines
   `YAW_RATE_NORM=5.0`, `MAX_SPEED=4.0`, `MAX_STEER_RAD`, etc., but the env uses
   bare literals for the same quantities (`/ 5.0` yaw norm at
   `deepracer_env.py:485`, flip threshold `up_z < 0.3` at `:572`, hard-off
   `+ 0.4` at `:580`). Centralize physical/normalization constants (a
   `constants.py` or the typed `EnvConfig`) and reference them everywhere.

4. **Domain-randomization code living in the env (cohesion).** `_resample_world
   _color` + the RGB↔YIQ remap matrices (`deepracer_env.py:32-38,530-553`) are a
   DR/obs-transform concern sitting in the env. Move to `randomization/` (or make
   it an `ObsRenderer` post-step from Part B.3), so the env holds sim state, not
   augmentation math.

5. **No tests for the domain logic (risk).** `tests/` covers the experiment
   layer but not the subtle rules — lap-wrap across the finish line, `dir_sign`
   reversed driving, off-track/flip thresholds, progress wrap. Extracting
   `rules.py` (Part B.4) is the moment to add unit tests for these; they're pure
   batched-torch functions, easy to test on tiny synthetic tracks.

## Part E — type annotations + Google-style docstrings + a checker (cross-cutting)

The single biggest day-to-day readability win. The codebase leans on untyped
params, so relationships are invisible until you read the body — e.g.
`TorchRLDeepRacerEnv.__init__(self, sim, ...)` gave no hint that `sim` is a
`DeepRacerEnv` (fixed: `torchrl_env.py` now annotates `sim: DeepRacerEnv` under a
`TYPE_CHECKING` guard). Apply the same treatment everywhere.

Priorities (high-value, low-churn first):

1. **Boundary objects** — annotate every `sim`, `env`, `builder`, `spec`, `cfg`,
   `tensordict` parameter and the attributes that hold them (`self.sim`,
   `self.b`, `self.track`, `self.obs_renderer`, ...). These are the ones that
   make jump-to-definition work.
2. **Public method signatures** across `envs/`, `experiment/`, `algorithms/` —
   params + return types. `from __future__ import annotations` is already common
   here, so forward refs are free.
3. **The config surface** — lands with Part D (`EnvConfig` etc. replace the bare
   `dict` annotations).
4. **Pluggable contracts** — `RewardFn`, `FeatureSet`, `Algorithm`,
   `ObsRenderer`/`ViewRenderer` (Part B/C) become the annotation types.

Use `TYPE_CHECKING` for heavy/cyclic imports (genesis, torch-heavy modules,
intra-package env refs) so annotations stay zero-cost.

**Google-style docstrings** on the same pass. The `experiment/`, `algorithms/`,
and `spec` layers already model the house style (`Args:` / `Returns:` /
`Raises:`); the env/track/features/rewards core is thin on it. Bring every public
function/method to that standard — a one-line summary plus `Args`/`Returns`
(`Raises`/`Note` where relevant). Types go in annotations (Part E), not repeated
in the docstring prose. Optionally enforce presence with `ruff`'s pydocstyle
rules (`D` / `google` convention) alongside pyright.

**Add a checker** so it doesn't rot: `pyright` in basic mode (fast, great
inference, no config beyond `pyrightconfig.json` pinning `envs/`, `experiment/`,
`algorithms/`). Wire it into CI as non-blocking first, then tighten. This is what
prevents the next untyped `sim` from creeping back in.

## Part G — migrate downstream consumers (experiments + notebooks)

Every refactor changes the **authoring API** (`experiment/stages.py`:
`Algo(...)`, `RewardShaping(...)`, the `Experiment` base) and the config
factories — so every experiment file and notebook that uses them must be
migrated *in the same PR that changes the API*, or they silently break. The
DSL stage signatures are the leverage point: change `stages.py` once, fix all
call sites.

Verified affected surface:

**`experiments/*.py`**

| file | uses | breaks under |
|---|---|---|
| `camera.py` | `@experiment` ×4, `ablation_group=` | C.4, A |
| `feature.py` | `@experiment` ×3, `ablation_group=` | C.4, A |
| `safe.py` | `@experiment`, `class SafeTransfer(Experiment)`, `ablation_group=` | C.4, A |
| `template.py` | `class MyExperiment(Experiment)`, `ablation_group=` | C.4, A |
| `hpo_optuna.py` | `run(..., force=True)` | A |
| `__init__.py` | "importing fires the registrations" docstring/behavior | C.4 |

Migration: `@experiment`/subclass → entries in the explicit `EXPERIMENTS = {...}`
map (C.4); drop `force=` (A); `ablation_group`/`variant` stay as plain report
labels (remove only if report grouping is dropped — see judgment call below).

**`notebooks/*.ipynb`**

| notebook | uses | breaks under |
|---|---|---|
| `custom_algorithm_reinforce.ipynb` | `@register_algorithm` ×3, `Algo(kind=...)` ×3, `force=`, `ablation` | C.3, A |
| `hpo_cnn.ipynb` | `ablation`, `force=` | A |
| `deepracer_genesis_colab.ipynb` | `ablation` | A |
| `deepracer_genesis_colab_output.ipynb` | `ablation` ×2 (committed *executed* copy) | A + regenerate |
| `track_designer.ipynb` | `get_env_cfg` ×2 | D |

Migration: rewrite the REINFORCE notebook to define a plain `Algorithm` class and
pass it via `Algo(cls=REINFORCE)` (C.3); strip `force=`/`ablation` demos (A);
update `get_env_cfg` usage to the typed/grouped `EnvConfig` (D); **regenerate**
`deepracer_genesis_colab_output.ipynb` by re-executing (it's a committed output
artifact — stale cells would misdocument the new API).

**Judgment call:** with automatic ablation gone, decide whether `report.py`'s
per-group delta tables (and thus the `ablation_group`/`variant` tags) stay as a
*manual* comparison aid, or are removed too. If kept, experiments keep the tags
unchanged; if removed, strip them from all `build(...)` calls above.

**Testing the notebooks:** they don't run under pytest. Add a smoke check that
`nbconvert --execute` runs each with a tiny `num_envs`/`total_env_steps` (or at
minimum that every code cell imports/parses), so API drift is caught.

## Suggested order (each ships independently, tests green between)

0. **Part E (boundary pass) + checker** — annotate the `sim`/`env`/`builder`/
   `spec` boundary objects and stand up `pyright` (non-blocking). Immediate
   readability, and it guards every later step. Then keep typing each module as
   you touch it in A–D.
1. **Part A** — delete cache + auto-ablation. Smallest, unblocks C.
2. **Part C** — registries → parameters (now that identity is a non-issue).
3. **Part D (flat first)** — add `EnvConfig`/`TrainConfig` typing as annotations,
   keys still flat (no call-site churn yet).
4. **Part B** — decompose the env, in this sub-order:
   1. `entities.py` (`Car`, `TrackEntity`),
   2. `rules.py` + `mdp.py` (pull the predicates + R/T out),
   3. `renderers.py` (obs + view strategies),
   4. `scene.py`,
   5. `base_env.py` + `vector_env.py` + `vision_env.py`.
5. **Part D (grouping)** — nest the TypedDicts + do the mechanical call-site
   churn, last, once modules are stable.

**Part G is not a final step — it rides with each part.** Whenever a step changes
the authoring API (`stages.py` signatures, `Experiment` base, `get_env_cfg`),
migrate the affected `experiments/*.py` and notebooks *in that same PR* (see the
Part G tables for which break under which part). A refactor PR that leaves a
notebook calling the old API is incomplete.

## Risk / test checklist

- **Both front-ends keep working**: after Part B, run `python -m
  deepracer_genesis.train -B 64 --max_iterations 2` (rsl-rl path) AND a tiny
  TorchRL `run(..., total_env_steps=small)` (wrapper path). The wrapper's
  `step_info` contract (`terminated`/`truncated` split) must be unchanged.
- **CPU view renderer** works in a `VectorDeepRacerEnv` (spectator video with no
  camera obs).
- **rsl-rl boundary**: `get_train_cfg` still yields the exact dict
  `OnPolicyRunner` expects.
- **No global state**: grep confirms `REWARDS`/`FEATURE_SETS`/`ALGORITHMS`/
  `REGISTRY` and `spec.id()` are gone.
- **Perf**: `step()`/`_post_physics` still read a dict and call batched torch; no
  per-step dataclass/attribute overhead introduced.
- Existing suites: `test_experiment_spec`, `test_ablation`, `test_report`,
  `test_ppo_lagrangian`, `test_transforms`.
- **Downstream consumers (Part G)**: every `experiments/*.py` still
  imports/builds a spec; every notebook parses (ideally `nbconvert --execute`
  with tiny sizes). Regenerate `deepracer_genesis_colab_output.ipynb`. Grep
  confirms no `@register_*` / `Algo(kind=` / `force=` / auto-`ablation` remain.
