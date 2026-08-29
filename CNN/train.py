import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PISTES, RACINE, RolloutDataset
from model import PerceptionCNN

VAL = ("reInvent2019_track_v2", "2022_reinvent_champ_v2",
       "Vegas_track_v2", "Mexico_track_v2")
TRAIN = tuple(p for p in PISTES if p not in VAL)

EPOCHS = 12
BATCH = 64
LR = 1e-4


def evaluer(net, dl, loss_fn, device):
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            total += loss_fn(net(x), y).item() * len(x)
            n += len(x)
    net.train()
    return total / n


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_ds = RolloutDataset(pistes=TRAIN)
    val_ds = RolloutDataset(pistes=VAL)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=2, persistent_workers=True)

    net = PerceptionCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    print(f"train {len(train_ds):,} | val {len(val_ds):,} | {device}")
    meilleure = float("inf")
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
                fait = n * BATCH / len(train_ds) * 100
                print(f"\r  epoque {epoch:3}  {fait:5.1f}%", end="", flush=True)

        train_loss = total / len(train_ds)
        val_loss = evaluer(net, val_dl, loss_fn, device)
        print(f"\r  epoque {epoch:3}  train {train_loss:.5f}  val {val_loss:.5f}")

        if val_loss < meilleure:
            meilleure = val_loss
            torch.save(net.state_dict(), RACINE / "CNN" / "perception.pt")
            print("    -> sauvegarde")

    print(f"\nmeilleure val : {meilleure:.5f}")


if __name__ == "__main__":
    main()
