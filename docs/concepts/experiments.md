# Experiments & the `>>` DSL

Experiments are **config-as-code**: you compose an environment, optional domain
randomization, a policy, and (optionally) an algorithm into an immutable
`ExperimentSpec` using the `>>` operator, then `build()` or `run()` it.

> Mental model in one sentence: each `>>` stage is a pure function
> `ExperimentSpec -> ExperimentSpec`; the chain must start with an environment and
> contain exactly one policy, and `build()` folds the stages into a validated,
> frozen spec.

---

## The pipeline

`>>` chains `Stage` objects into a `Pipeline` (`experiment/stages.py`). A single
stage becomes a one-element pipeline (`Stage.__rshift__`), and pipelines concatenate
(`Pipeline.__rshift__`). `Pipeline.build(**overrides)`:

1. validates structure (`_check_structure`),
2. folds each stage's `apply()` into a fresh `ExperimentSpec`,
3. infers the algorithm if none was set (`_infer_algorithm`),
4. calls `spec.validate()` and returns the frozen spec.

Structure rules: the **first stage must be an environment**; **exactly one policy**
stage; at most one each of encoder / action-DR / algorithm / camera-DR / physics-DR;
zero or more reward-shaping and DR stages.

## Stage catalog

| Role | Stage | Key params |
|------|-------|-----------|
| Env | `FeatureEnvironment` | `feature_set`, `feature_params`, `lookahead_k`, `tracks`, `num_envs`, `random_start`, `random_direction` |
| Env | `CameraEnvironment` | `render` (`"madrona"`/`"nyx"`), `resolution`, `fov`, `feature_set`, `tracks`, `num_envs` |
| Env (safe-RL) | `SafeRLFeatureEnvironment` / `SafeRLCameraEnvironment` | above + `cost`, `budget` (emits a cost stream → infers PPO-Lagrangian) |
| Reward | `RewardShaping` | `fn` (custom `RewardFn` or None), `scales` (override dict) |
| DR | `DomainRandomizationTrackAppearance` | `strength` |
| DR | `DomainRandomizationCamera` | `brightness`, `contrast`, `saturation`, `hue`, `blur`, `cutout`, `noise`, `camera_jitter` |
| DR | `DomainRandomizationPhysics` | `friction`, `mass`, `com`, `gains`, `armature` |
| DR | `DomainRandomizationActions` | `steer_noise`, `speed_noise`, `delay_steps` |
| Encoder | `FrozenCNNToFeatureVector` | `checkpoint`, `output_dim`, `layer`, `out_key` |
| Policy | `VectorPolicy` | `keys`, `mlp`, `actions` |
| Policy | `AsymmetricVectorPolicy` / `AsymmetricCameraPolicy` | `actor_keys`, `critic_keys`, `mlp`/`cnn`, `actions` |
| Algorithm | `PPO`, `PPOLagrangian`, `Algo(cls=...)` | see [Custom algorithms](../guides/custom-algorithms.md) |

Policies expose **`actor_keys` / `critic_keys`** — the asymmetric-critic hook: the
critic can read richer observation keys than the actor (e.g. `critic_keys=("camera",
"state")` with `actor_keys=("camera",)`).

## Two example chains

Feature-vector baseline:

```python
spec = (
    FeatureEnvironment(num_envs=1024, lookahead_k=10)
    >> VectorPolicy(keys=("state",))
).build(seed=0)
```

Camera + full DR + transfer encoder + safe-RL:

```python
spec = (
    SafeRLCameraEnvironment(render="madrona", cost="offtrack_or_overspeed", budget=25.0)
    >> DomainRandomizationCamera(brightness=(0.7, 1.3))
    >> FrozenCNNToFeatureVector(checkpoint="runs/.../best.pt", output_dim=256)
    >> VectorPolicy(keys=("encoded", "state"))
    >> DomainRandomizationActions(steer_noise=0.02)
).build(seed=0)
```

The safe-RL env emits a cost stream, so `_infer_algorithm` selects PPO-Lagrangian
automatically (no explicit algorithm stage needed).

## Building vs running

- `build(target, **overrides)` — validate and return the `ExperimentSpec` (also
  accepts a registered experiment name, an `Experiment` subclass, a `Pipeline`, or a
  spec).
- `run(target, *, root="runs", **overrides)` — build, then train via
  `Trainer(Builder(spec))`. Returns the eval record.

```python
from deepracer_genesis.experiment import run
run(spec, root="runs")
```

Authoring longer-lived experiments as a class (with `total_env_steps`,
`eval_every_steps`, `num_envs`, and a `pipeline()` method) is covered in the
[tutorial](../tutorial.md).
