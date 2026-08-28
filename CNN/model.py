import torch
import torch.nn as nn


class PerceptionCNN(nn.Module):
    def __init__(self, in_channels=12, n_targets=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 13 * 18, 256), nn.ReLU(),
            nn.Linear(256, n_targets),
        )

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    net = PerceptionCNN()
    x = torch.zeros(8, 12, 120, 160)

    print("forme a chaque etage :")
    print(f"  entree        {tuple(x.shape)}")
    for couche in net.features:
        x = couche(x)
        if isinstance(couche, torch.nn.Conv2d):
            print(f"  apres conv    {tuple(x.shape)}")
    y = net.head(x)
    print(f"  sortie        {tuple(y.shape)}")

    n = sum(p.numel() for p in net.parameters())
    print(f"\nparametres a apprendre : {n:,}")
