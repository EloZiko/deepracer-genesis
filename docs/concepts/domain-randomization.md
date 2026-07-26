# Domain randomization

Domain randomization (DR) perturbs the simulator so a policy trained in Genesis
transfers to the real DeepRacer. This repo keeps **all DR *definitions* in one
folder** — `deepracer_genesis/randomization/` — even though each knob is *applied*
at a different layer.

> Mental model in one sentence: `randomization/` is the single home for *what can
> be randomized* (the `CATALOG`) and *how each knob samples* (a `Space`), while
> *where* it acts stays at its native layer — physics before stepping, visual in
> the renderer, actuation/image as TorchRL transforms.

---

## The catalog — one table of every knob

`randomization/catalog.py` is documentation-as-data: importing it has no runtime
effect. Each `Knob` (`catalog.py:24`) records its name, a suggested `Space`, the
**layer** it acts at, the `cfg` key its value lands in, and the **signal(s)** it
perturbs (the Part K vocabulary — see [Feature vectors](features.md)):

```python
Knob("friction", FloatRange(0.6, 1.4), "physics", "rand.friction_range",
     ("v_forward", "lateral"), "per-link friction ratio")
```

| Layer | Applied where | Example knobs |
|-------|---------------|---------------|
| `physics` | at `reset`, before stepping (`randomization/physics.py`) | friction, mass_shift, com_shift, steer_kp_scale, wheel_kv_scale, armature |
| `visual` | in the renderer (`randomization/visual.py`) | world_color (YIQ remap), camera_pitch_jitter, camera_pos_jitter, pixel_noise |
| `actuation` | TorchRL inverse-action transform (`randomization/actuation.py`) | steer_noise, speed_noise, delay_steps |
| `image` | TorchRL obs transform (`randomization/actuation.py`, `ImageAug`) | brightness, contrast, saturation, hue, blur, cutout, noise |

`CATALOG` (`catalog.py:45`) is the full list; `by_layer(layer)` filters it and
`BY_NAME` indexes it. DR, HPO, and the build-time learnability check all read this
one table.

## Search-space types (shared with HPO)

Ranges are declared once as `Space` objects in `randomization/spaces.py` and reused
by both DR (resampled per episode on GPU) and [HPO](../guides/hpo.md) (searched as
a scalar per trial):

- `FloatRange(lo, hi, log=False)` — continuous; `suggest()` for HPO, `sample(n, device)` for DR.
- `IntRange(lo, hi)` — integer, both paths.
- `SymRange(m)` — samples `[-m, m]`; DR-native (`suggest()` raises — no scalar to freeze).
- `Choice(values)` — HPO-only categorical; `sample()` raises (no batched-GPU categorical DR).

See the plan's Part H (`REFACTOR_PLAN.md`) for why one *type* is shared but the
declaration *sites* stay separate.

## What each layer perturbs

- **Physics** (`randomize_physics`, applied at reset): per-link friction, base
  mass, COM offset, steering `kp`/`kv` scale, wheel `kv` scale, and `armature`
  (reflected rotor inertia added to the joint mass matrix).
- **Visual** (in the [renderer](renderers.md)): a per-episode **world-color YIQ
  remap** (hue/saturation/value shift in chroma space), camera-mount pitch/position
  jitter (Madrona only), and additive pixel noise.
- **Actuation** (`ActionNoiseDelay`, a TorchRL inverse transform): k-step command
  latency then per-channel Gaussian noise on `[steer, speed]`.
- **Image** (`ImageAug`, a TorchRL obs transform): brightness/contrast/saturation/
  hue/blur/cutout/noise resampled per step per sub-env.

## What is NOT randomized

- **Lighting** is fixed at build time.
- The **offline texture bake** in `randomization/appearance.py` (sha1-cached track/
  field variants) is a rasterizer tool and is **not wired into camera training**;
  train-time appearance DR is the world-color remap above.

## Authoring DR in an experiment

DR is added with `>>` stages (see [Experiments](experiments.md)):

```python
CameraEnvironment(render="madrona", num_envs=128)
    >> DomainRandomizationTrackAppearance(strength=0.6)      # world-color
    >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                 camera_jitter=True)          # image + mount
    >> DomainRandomizationPhysics()                           # friction/mass/...
    >> AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))
    >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05, delay_steps=1)
```

Each stage accepts either a raw range/scalar or a `Space` (a suggested default
range lives in the catalog).
