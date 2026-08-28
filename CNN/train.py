import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RACINE, RolloutDataset
from model import PerceptionCNN

EPOCHS = 60
BATCH = 8
LR = 1e-4


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ds = RolloutDataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    net = PerceptionCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"{len(ds)} exemples, {device}")
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)

            pred = net(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * len(x)

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  epoque {epoch:3}  loss {total / len(ds):.5f}")

    sortie = RACINE / "CNN" / "perception.pt"
    torch.save(net.state_dict(), sortie)
    print("modele sauve :", sortie)


if __name__ == "__main__":
    main()
