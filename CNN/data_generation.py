import sys
import warnings
from pathlib import Path
from deepracer_genesis.experiment.stages import DomainRandomizationTrackAppearance
from deepracer_genesis.experiment import CameraEnvironment
from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.datasets.rollout import collect_rollout_dataset


warnings.filterwarnings("ignore")



RACINE = Path(__file__).resolve().parent.parent
piste = sys.argv[1]


monde = (CameraEnvironment(
    backend="cpu",
    resolution=(160, 120),
    num_envs=8,
    tracks=(piste,),
    feature_set=PerceptionFeatures,
    random_start=True,
) >> DomainRandomizationTrackAppearance(strength=0.6))

collect_rollout_dataset(monde, out=str(RACINE / "data" / f"{piste}_v2"), steps=1024, seed=0)

