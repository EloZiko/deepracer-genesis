"""Train the perception CNN on the collected rollouts.

    python -m perception.train_cnn

Saves the best checkpoint by validation loss to ``perception/perception.pt``.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from perception.dataset import DATASET_TRACKS, REPO_ROOT, RolloutDataset
from perception.model import PerceptionCNN

# 10 tracks picked so their distribution matches that of the whole set
VAL_TRACKS = ("2022_july_pro_v2", "2022_march_open_v2", "2022_may_pro_v2",
              "2022_reinvent_champ_v2", "2022_summit_speedway_v2",
              "AmericasGeneratedInclStart_v2", "Belille_v2", "dubai_open_v2",
              "morgan_open_v2", "penbay_pro_v2")
TRAIN_TRACKS = tuple(t for t in DATASET_TRACKS if t not in VAL_TRACKS)

EPOCHS = 8          # 8x more data than before, so fewer epochs are enough
BATCH = 64
LR = 1e-4
VAL_MAX = 60_000    # validation sample: past this we are measuring noise


def evaluate(net, loader, loss_fn, device):
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(net(x), y).item() * len(x)
            n += len(x)
    net.train()
    return total / n


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_ds = RolloutDataset(tracks=TRAIN_TRACKS)
    val_ds = RolloutDataset(tracks=VAL_TRACKS)
    if len(val_ds) > VAL_MAX:      # fixed draw, so runs stay comparable
        g = torch.Generator().manual_seed(0)
        val_ds = Subset(val_ds, torch.randperm(len(val_ds), generator=g)[:VAL_MAX].tolist())
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=8, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=4, persistent_workers=True)

    net = PerceptionCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    print(f"train {len(train_ds):,} | val {len(val_ds):,} | {device}")
    best = float("inf")
    for epoch in range(EPOCHS):
        total = 0.0
        for n, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            if n % 50 == 0:
                done = n * BATCH / len(train_ds) * 100
                print(f"\r  epoch {epoch:3}  {done:5.1f}%", end="", flush=True)

        train_loss = total / len(train_ds)
        print(f"\r  epoch {epoch:3}  validating...", end="", flush=True)
        val_loss = evaluate(net, val_dl, loss_fn, device)
        print(f"\r  epoch {epoch:3}  train {train_loss:.5f}  val {val_loss:.5f}"
              f"  lr {sched.get_last_lr()[0]:.2e}")
        sched.step()

        if val_loss < best:
            best = val_loss
            torch.save(net.state_dict(), REPO_ROOT / "perception" / "perception.pt")
            print("    -> saved")

    print(f"\nbest val: {best:.5f}")


if __name__ == "__main__":
    main()
