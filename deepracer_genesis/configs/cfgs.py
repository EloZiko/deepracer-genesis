"""Default env + rsl-rl-lib 5.x train configs for DeepRacer-Genesis."""

from ..physics.limits import MAX_SPEED, MAX_STEERING_DEG, MIN_SPEED
from .schema import EnvConfig


def get_env_cfg(vision=False, track="reinvent_base", randomize=False,
                topdown=False) -> EnvConfig:
    """Build the default nested env config, one typed section per subsystem."""
    return {
        "sim": {
            "dt": 0.01,
            "decimation": 2,          # control at 50 Hz
            "episode_length_s": 30.0,
            "track": track,
        },
        # action mapping (original DeepRacer Box([-30, 0.1], [30, 4.0]))
        "action": {
            "max_steering_deg": MAX_STEERING_DEG,
            "min_speed": MIN_SPEED,
            "max_speed": MAX_SPEED,
            "action_table": None,     # discrete (steer, speed) pairs, or None
        },
        "car": {
            "steer_kp": 25.0,
            "steer_kv": 5.0,          # heavy damping: low values cause front-wheel shimmy
            "wheel_kv": 5.0,
            "wheel_max_torque": 3.0,
            # "ackermann" (per-wheel inner/outer) | "parallel" (legacy, both equal)
            "steering_model": "ackermann",
        },
        "spawn": {
            "random_start": True,
            "random_direction": False,   # coin-flip CW/CCW driving direction per episode
            "spawn_lateral_noise": 0.15,
            "spawn_yaw_noise": 0.3,
            "spawn_height": 0.03,
        },
        "termination": {
            "off_track_margin": 0.10,   # m beyond road edge before terminating
            "wheel_margin": 0.08,       # ~half car width, for all_wheels_on_track
            "crash_penalty": -10.0,
            "overspeed_limit": 3.5,     # m/s, for the offtrack_or_overspeed cost
        },
        "obs": {
            "lookahead_k": 10,
            "lookahead_stride": 3,
            "lookahead_scale": 3.0,
            "obs_noise": 0.0,
            "feature_set": "classic",
            "feature_params": {},
        },
        "vision": {
            "vision": vision,
            "camera_res": (160, 120),   # DeepRacer-native observation resolution
            "camera_fov": 90,
            "camera_pitch_deg": 10.0,
            "policy_res": None,         # downscale target for the policy (None = camera_res)
            "topdown_camera": topdown,  # per-env batch camera (validation checks)
            "spectator": False,         # high-res rasterizer cam, all cars in one view
            "spectator_res": (1280, 960),
            "madrona_rg_swap": False,   # see env: only alpha-cutout textures are swapped
            "vision_renderer": "batch",  # "batch" (Madrona) | "nyx" (path tracer)
            "nyx_mode": "Forward",      # "Forward" | "FastPathTracer" | "RefPathTracer"
            "nyx_spp": 4,
            "nyx_light_intensity": 3.0,
            "light_intensity": 6.0,
            "pixel_noise": 0.0,
            "background_color": (0.55, 0.72, 0.9),
            "field_color": (0.30, 0.48, 0.32),
            "appearance": {},           # {"world_color": strength} for the color remap
        },
        "reward": {
            "reward": None,             # a reward callable, or None -> deepracer default
            "reward_scales": {
                "progress": 10.0,
                "speed": 0.5,
                "centered": 0.5,
                "heading": 0.5,
                "steering": 0.3,
                "action_rate": 0.05,
                "off_track": 2.0,
            },
            "reward_scale_overrides": {},
            "emit_cost": False,
            "cost_fn": None,
        },
        # per the Genesis DR guide: friction/mass/COM per link, kp/kv/armature
        # per dof, all per-env (needs batch_dofs_info/batch_links_info)
        "rand": {
            "randomize": randomize,
            "friction_range": (0.6, 1.4),
            "mass_shift_kg": 0.2,
            "com_shift_m": 0.01,
            "steer_kp_scale": (0.8, 1.2),
            "wheel_kv_scale": (0.8, 1.2),
            "armature_range": (0.0, 0.01),
            "camera_pitch_jitter_deg": 2.0,
            "camera_pos_jitter_m": 0.005,
        },
    }


def get_train_cfg(vision=False):
    if vision:
        obs_groups = {"actor": ["camera"], "critic": ["state", "camera"]}
        actor = {
            "class_name": "CNNModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
            "cnn_cfg": {
                "output_channels": [16, 32, 64],
                "kernel_size": [8, 4, 3],
                "stride": [4, 2, 1],
                "activation": "relu",
                "flatten": True,
            },
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        }
        critic = {
            "class_name": "CNNModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
            "cnn_cfg": {
                "output_channels": [16, 32, 64],
                "kernel_size": [8, 4, 3],
                "stride": [4, 2, 1],
                "activation": "relu",
                "flatten": True,
            },
        }
        share_cnn = True
    else:
        obs_groups = {"actor": ["state"], "critic": ["state"]}
        actor = {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        }
        critic = {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
        }
        share_cnn = False

    cfg = {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "obs_groups": obs_groups,
        "logger": "tensorboard",
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 3.0e-4,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "actor": actor,
        "critic": critic,
    }
    if share_cnn:
        cfg["algorithm"]["share_cnn_encoders"] = True
    return cfg
