import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

K = 4
SEUIL_VOITURE = 2.0          # m : en deca, une autre voiture bouche la vue
DEMI_FOV = np.radians(45)    # la camera a un champ de 90 deg
COURBURE_MAX = 1.0           # au-dela, le trace de la piste est aberrant
RACINE = Path(__file__).resolve().parent.parent   # la racine du repo

PISTES = tuple(f"{p}_v2" for p in (
    "reinvent_base", "Oval_track", "Bowtie_track", "Monaco", "Spain_track",
    "New_York_Track", "Austin", "Singapore", "Vegas_track", "China_track",
    "Mexico_track", "Tokyo_Training_track", "Canada_Training", "AWS_track",
    "reInvent2019_track", "2022_reinvent_champ", "2022_april_open",
    "2022_april_pro", "2022_august_open", "2022_august_pro",
    "2022_july_open", "2022_july_pro", "2022_june_open", "2022_june_pro",
    "2022_march_open", "2022_march_pro", "2022_may_open", "2022_may_pro",
    "2022_october_open", "2022_october_pro", "2022_september_open",
    "2022_september_pro", "2022_summit_speedway",
    "2022_summit_speedway_mini", "Albert", "AmericasGeneratedInclStart",
    "Aragon", "Belille", "FS_June2020", "H_track", "July_2020", "LGSWide",
    "arctic_open", "arctic_pro", "caecer_gp", "caecer_loop", "dubai_open",
    "dubai_pro", "hamption_open", "hamption_pro", "jyllandsringen_open",
    "jyllandsringen_pro", "morgan_open", "morgan_pro", "penbay_open",
    "penbay_pro", "red_star_open", "red_star_pro", "thunder_hill_open",
    "thunder_hill_pro",
))


def images_propres(df, seuil=SEUIL_VOITURE):
    """True pour les lignes ou aucune autre voiture n'est dans le champ.

    Les environnements partagent le monde sur le chemin CPU : chaque camera
    voit les voitures des autres. On repere celles qui genent a partir des
    positions enregistrees dans `pose`.
    """
    pose = np.array(df["pose"].tolist(), dtype=np.float32)   # x, y, yaw, progress
    t = df["t"].to_numpy()
    propre = np.ones(len(df), dtype=bool)

    for instant in np.unique(t):
        idx = np.flatnonzero(t == instant)
        xy, yaw = pose[idx, :2], pose[idx, 2]
        d = xy[None, :, :] - xy[:, None, :]                  # (m, m, 2)
        dist = np.linalg.norm(d, axis=2)
        angle = np.arctan2(d[:, :, 1], d[:, :, 0]) - yaw[:, None]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        gene = (dist > 1e-3) & (dist < seuil) & (np.abs(angle) < DEMI_FOV)
        propre[idx] = ~gene.any(axis=1)

    return propre


def courbures_valides(cibles, seuil=COURBURE_MAX):
    """True pour les lignes dont les deux courbures sont plausibles.

    Certaines pistes ont des waypoints presque confondus : le cercle ajuste sur
    trois points quasi superposes donne un rayon de quelques centimetres.
    """
    return np.abs(cibles[:, -2:]).max(axis=1) <= seuil


class RolloutDataset(Dataset):
    def __init__(self, pistes=PISTES, k=K):
        self.k = k
        self.tables = []          # un DataFrame par fichier
        self.index = []           # (numero de table, ligne de depart)

        for piste in pistes:
            dossier = RACINE / "data" / piste
            meta = json.loads((dossier / "meta.json").read_text())
            self.lo, self.hi = meta["cnn_target_slice"]

            for fichier in sorted(dossier.glob("rollout_*.parquet")):
                df = pd.read_parquet(fichier)
                env, ep = df["env"].to_numpy(), df["episode"].to_numpy()
                cibles = np.stack(df["state"].to_numpy())[:, self.lo:self.hi]
                propre = images_propres(df)
                courbure = courbures_valides(cibles)
                numero = len(self.tables)
                self.index += [(numero, i) for i in range(len(df) - k + 1)
                               if env[i] == env[i + k - 1] and ep[i] == ep[i + k - 1]
                               and propre[i:i + k].all() and courbure[i + k - 1]]
                self.tables.append(df)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        numero, i = self.index[n]
        df = self.tables[numero]
        x = torch.cat([self._frame(df, i + j) for j in range(self.k)], dim=0)
        state = df["state"].iloc[i + self.k - 1]
        y = torch.tensor(state[self.lo:self.hi], dtype=torch.float32)
        return x, y

    def _frame(self, df, row):
        img = Image.open(io.BytesIO(df["image"].iloc[row]))
        a = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(a).permute(2, 0, 1)     # (3, H, W)
