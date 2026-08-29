"""Image rendue de toute la piste vue du dessus.

    python CNN/voir_dessus.py Monaco
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import genesis as gs
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs.base_env import DeepRacerEnv

RACINE = Path(__file__).resolve().parent.parent
piste = sys.argv[1].replace("_v2", "").replace("_dr", "")

gs.init(backend=gs.cpu, logging_level="warning")

cfg = get_env_cfg(vision=True, track=piste, backend="cpu")
cfg["vision"]["vision_renderer"] = "rasterizer"   # pas de CUDA sur Mac
cfg["vision"]["spectator"] = True
cfg["vision"]["spectator_res"] = (1280, 960)

env = DeepRacerEnv(num_envs=8, env_cfg=cfg)
import torch
env.reset_idx(torch.arange(env.num_envs, device=env.device))
env._post_physics(torch.arange(env.num_envs, device=env.device))

image = env.render_spectator()
sortie = RACINE / "data" / f"dessus_{piste}.png"
Image.fromarray(np.asarray(image, dtype=np.uint8)).save(sortie)
print("ouvre :", sortie)
