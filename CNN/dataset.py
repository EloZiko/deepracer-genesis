"""Jeu de donnees des rollouts, servi depuis un cache a acces direct.

Les parquets gardent les images en memoire une fois lus, et macOS duplique le
dataset dans chaque worker : la RAM grimpe avec la taille du jeu de donnees.
On les recopie donc une fois dans un fichier plat, lu ensuite en memmap — le
cache disque du systeme fait le travail et les workers partagent les memes
pages au lieu d'en avoir chacun une copie.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

K = 4
COURBURE_MAX = 1.0           # au-dela, le trace de la piste est aberrant
RACINE = Path(__file__).resolve().parent.parent   # la racine du repo
CACHE = RACINE / "data" / "cache"

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


def courbures_valides(cibles, seuil=COURBURE_MAX):
    """True pour les lignes dont les deux courbures sont plausibles.

    Certaines pistes ont des waypoints presque confondus : le cercle ajuste sur
    trois points quasi superposes donne un rayon de quelques centimetres.
    """
    return np.abs(cibles[:, -2:]).max(axis=1) <= seuil


def _sources(piste):
    return sorted((RACINE / "data" / piste).glob("rollout_*.parquet"))


def _empreinte(pistes):
    """Identifie les parquets sources : nom, taille et date de modification."""
    return sorted([f.name, f.stat().st_size, int(f.stat().st_mtime)]
                  for p in pistes for f in _sources(p))


def construire_cache(pistes=PISTES):
    """Recopie les images bout a bout dans un fichier plat + un index.

    N'ecrit rien si le cache correspond deja aux parquets presents.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = CACHE / "sources.json"
    empreinte = _empreinte(pistes)
    if stamp.exists() and json.loads(stamp.read_text()) == empreinte:
        return

    offsets, tailles, cibles, env, episode, piste_id = [], [], [], [], [], []
    position = 0
    with open(CACHE / "images.bin", "wb") as blob:
        for pid, piste in enumerate(pistes):
            dossier = RACINE / "data" / piste
            lo, hi = json.loads((dossier / "meta.json").read_text())["cnn_target_slice"]
            for fichier in _sources(piste):
                df = pd.read_parquet(fichier)
                for img in df["image"]:
                    blob.write(img)
                    offsets.append(position)
                    tailles.append(len(img))
                    position += len(img)
                cibles.append(np.stack(df["state"].to_numpy())[:, lo:hi])
                env.append(df["env"].to_numpy())
                episode.append(df["episode"].to_numpy())
                piste_id.append(np.full(len(df), pid))
            print(f"  cache {piste}", flush=True)

    np.savez(CACHE / "index.npz",
             offsets=np.array(offsets, np.int64),
             tailles=np.array(tailles, np.int32),
             cibles=np.concatenate(cibles).astype(np.float32),
             env=np.concatenate(env).astype(np.int32),
             episode=np.concatenate(episode).astype(np.int32),
             piste_id=np.concatenate(piste_id).astype(np.int16),
             pistes=np.array(pistes))
    stamp.write_text(json.dumps(empreinte))


class RolloutDataset(Dataset):
    """Empilements de k images consecutives d'une meme voiture, et leur cible."""

    def __init__(self, pistes=PISTES, k=K):
        construire_cache()
        d = np.load(CACHE / "index.npz")
        toutes = list(d["pistes"])
        garde = np.isin(d["piste_id"], [toutes.index(p) for p in pistes])

        self.k = k
        self.offsets, self.tailles = d["offsets"], d["tailles"]
        self.cibles = d["cibles"]
        self._blob = None                  # ouvert par worker, jamais serialise

        env, ep, pid = d["env"], d["episode"], d["piste_id"]
        courbure = courbures_valides(self.cibles)
        # un empilement est valide s'il reste dans la meme voiture, le meme
        # episode et le meme fichier, et si sa cible n'est pas aberrante
        i = np.flatnonzero(garde)[: -(k - 1) or None]
        j = i + k - 1
        ok = ((env[i] == env[j]) & (ep[i] == ep[j]) & (pid[i] == pid[j])
              & garde[j] & courbure[j])
        self.index = i[ok]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        i = self.index[n]
        x = torch.cat([self._frame(i + j) for j in range(self.k)], dim=0)
        y = torch.from_numpy(self.cibles[i + self.k - 1].copy())
        return x, y

    def _frame(self, ligne):
        if self._blob is None:
            self._blob = np.memmap(CACHE / "images.bin", dtype=np.uint8, mode="r")
        o, t = self.offsets[ligne], self.tailles[ligne]
        img = Image.open(io.BytesIO(self._blob[o:o + t].tobytes()))
        a = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(a).permute(2, 0, 1)     # (3, H, W)
