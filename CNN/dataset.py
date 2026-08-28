import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

K = 4  


class RolloutDataset(Dataset):
    def __init__(self, root="data/mini", k=K):
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        self.lo, self.hi = meta["cnn_target_slice"]
        self.k = k

        self.df = pd.read_parquet(root / "rollout_0000.parquet")
        env = self.df["env"].to_numpy()
        ep = self.df["episode"].to_numpy()
        self.index = [i for i in range(len(self.df) - k + 1)
                      if env[i] == env[i + k - 1] and ep[i] == ep[i + k - 1]]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        i = self.index[n]
        x = torch.cat([self._frame(i + j) for j in range(self.k)], dim=0)
        state = self.df["state"].iloc[i + self.k - 1]
        y = torch.tensor(state[self.lo:self.hi], dtype=torch.float32)
        return x, y

    def _frame(self, row):
        img = Image.open(io.BytesIO(self.df["image"].iloc[row]))
        a = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(a).permute(2, 0, 1)     # (3, H, W)


if __name__ == "__main__":
    ds = RolloutDataset()
    x, y = ds[0]
    print("exemples :", len(ds))
    print("entree   :", tuple(x.shape), x.dtype, f"[{x.min():.3f}, {x.max():.3f}]")
    print("cible    :", tuple(y.shape), y.dtype)
    print("valeurs  :", [round(v, 3) for v in y.tolist()])
