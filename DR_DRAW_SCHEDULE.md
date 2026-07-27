# Domain Randomization — what's drawn when, and at what scope

Current wiring (build-time physics DR). "Per-env" = each env draws its own value;
"shared" = identical across all envs.

## 1. Randomized at BUILD TIME — drawn once per env, then FIXED for the whole run

Applied in `DeepRacerEnv.__init__` (`randomize_physics` + `randomize_armature` +
renderer `randomize_mount`), only when `rand.randomize` is on. Each env draws its
**own** value once; that value is **the same for that env across all its
episodes** — batch diversity comes from having N envs with N different bodies,
not from resampling.

| knob | cfg key | default | scope | effect |
|---|---|---|---|---|
| friction | `rand.friction_range` | (0.6, 1.4) ×base | per-env, per-link | friction ratio |
| mass shift | `rand.mass_shift_kg` | ±0.2 kg (clamped ≤0.9× each link's rest mass) | per-env, per-link | additive link mass |
| COM shift | `rand.com_shift_m` | ±0.01 m (3-axis) | per-env, per-link | center-of-mass offset |
| steer gains | `rand.steer_kp_scale` | ×(0.8, 1.2) | per-env | scales steering PD **kp AND kv** together |
| wheel damping | `rand.wheel_kv_scale` | ×(0.8, 1.2) | per-env | wheel drive `kv` |
| armature | `rand.armature_range` | (0.0, 0.01) | per-env | motor armature (separate call — hides a mass-matrix recompute) |
| camera pitch | `rand.camera_pitch_jitter_deg` | ±2.0° | per-env | camera-mount pitch (camera envs only) |
| camera position | `rand.camera_pos_jitter_m` | ±0.005 m | per-env | camera-mount offset (camera envs only) |

## 2. Re-drawn PER EPISODE (every `reset_idx`, per-env)

Not build-time — these change each episode for the envs that reset:

| what | cfg key | default | scope |
|---|---|---|---|
| spawn waypoint | `spawn.random_start` | on | per-env |
| spawn lateral noise | `spawn.spawn_lateral_noise` | 0.15 m | per-env |
| spawn yaw noise | `spawn.spawn_yaw_noise` | 0.3 rad | per-env |
| driving direction (CW/CCW) | `spawn.random_direction` | off | per-env |
| world-color remap (YIQ) | `vision.appearance.world_color` | 0.0 (off) | per-env (vision only) |

## 3. Re-drawn PER STEP (every control step / observation, per-env)

| what | cfg key | default | scope |
|---|---|---|---|
| state obs noise | `obs.obs_noise` | 0.0 (off) | per-env |
| camera pixel noise | `vision.pixel_noise` | 0.0 (off) | per-env (vision) |
| image aug (brightness/contrast/saturation/hue/blur) | `obs_dr.image_aug.*` | off | per-env (vision, torchrl `ImageAug`) |
| action noise + delay | `action_dr.steer_noise` / `speed_noise` / `delay_steps` | 0 | per-env (torchrl `ActionNoiseDelay`) |

## 4. Drawn the SAME for every env (shared / not per-env randomized)

Everything **not** in §1–§3 is identical across all envs — the nominal body and
gains before DR scaling, wheel radius (measured from the STL), track geometry,
`dt`/`decimation`, action limits, reward scales. And any DR knob left at its
default-off value (`obs_noise`, `pixel_noise`, image aug, action noise,
`world_color`, `random_direction`) means every env is identical on that axis.

## Summary

- **Fixed per env for the run (build-time):** friction, mass, COM, steer kp/kv,
  wheel kv, armature, camera mount. → this is "what's randomizable at build time,"
  and each env keeps the same draw across all its episodes.
- **Re-drawn each episode:** spawn pose/direction, world-color.
- **Re-drawn each step:** obs/pixel/image/action noise (all off by default).
- **Identical across envs:** the base body/gains/track/timing, plus any off knob.
