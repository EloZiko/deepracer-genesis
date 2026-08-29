import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RACINE, RolloutDataset
from model import PerceptionCNN

TRAIN = ("reinvent_base", "Oval_track", "Bowtie_track",
         "Spain_track", "New_York_Track")
VAL = ("Monaco",)

EPOCHS = 2          # commence petit pour mesurer
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
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    net = PerceptionCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"train {len(train_ds):,} | val {len(val_ds):,} | {device}")
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)

        train_loss = total / len(train_ds)
        val_loss = evaluer(net, val_dl, loss_fn, device)
        print(f"  epoque {epoch:3}  train {train_loss:.5f}  val {val_loss:.5f}")

    torch.save(net.state_dict(), RACINE / "CNN" / "perception.pt")


if __name__ == "__main__":
    main()
