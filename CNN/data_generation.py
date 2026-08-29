import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import CameraEnvironment
from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.datasets.rollout import collect_rollout_dataset

RACINE = Path(__file__).resolve().parent.parent
piste = sys.argv[1]

monde = CameraEnvironment(
    backend="cpu",
    resolution=(160, 120),
    num_envs=64,
    tracks=(piste,),
    feature_set=PerceptionFeatures,
    random_start=True,
)

collect_rollout_dataset(monde, out=str(RACINE / "data" / piste),
                        steps=512, seed=0)
