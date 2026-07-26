# Google Colab (CLI)

Colab is good for **feature-vector (state) training** — it has a CUDA GPU but no
NVIDIA graphics userland, so the Madrona/Nyx **camera renderers can't run there**.
Camera policies need a local NVIDIA machine. The reference notebook is
`notebooks/deepracer_genesis_colab.ipynb`.

> Mental model in one sentence: install from git with `uv`, define an `Experiment`
> class in a cell, `run()` it on `/content/runs`, then render a software-Mesa video
> and save the run to Drive.

---

## 1. Runtime + install

A T4 runtime is enough. First uninstall TensorFlow (its LLVM clashes), then install:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip uninstall -q -y tensorflow tensorflow-cpu tf-keras
!pip install -q uv
!uv pip install --system -q --no-cache \
    "deepracer-genesis @ git+https://github.com/Luna-v0/deepracer-genesis.git@main"
```

## 2. Define and run

```python
from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy, run

class ColabRacer(Experiment):
    total_env_steps = 5_000_000        # ~8 min on a T4
    eval_every_steps = 1_000_000
    num_envs = 512                     # T4-sized
    def pipeline(self):
        return FeatureEnvironment(lookahead_k=10, num_envs=self.num_envs) >> VectorPolicy(keys=("state",))

record = run(ColabRacer, root="/content/runs")
```

## 3. Watch a rollout

Videos render through the software (Mesa) rasterizer — the spectator view works
without the camera renderer:

```python
from deepracer_genesis.experiment.visualize import rollout_video
from IPython.display import Video
mp4 = rollout_video(ColabRacer, root="/content/runs", steps=300, num_envs=8,
                    spectator_res=(640, 480))
Video(mp4, embed=True, width=720)
```

## 4. Save the run

```python
from google.colab import drive; import os, glob, shutil
drive.mount("/content/drive")
run_dir = sorted(glob.glob("/content/runs/*/*"))[-1]
shutil.copytree(run_dir, "/content/drive/MyDrive/deepracer_runs/" + os.path.basename(run_dir),
                dirs_exist_ok=True)
```

For local, camera-capable training see [Local install & run](local.md).
