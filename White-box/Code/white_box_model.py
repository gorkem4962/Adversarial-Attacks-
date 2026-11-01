"""White-box MNIST classifier for adversarial experiments.

Attacks are intentionally not implemented here — only the model API.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MnistCNN(nn.Module):
    """Small CNN: 1x28x28 -> 10 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 28, 28) in [0, 1]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits


class WhiteBoxModel:
    """Train / load / query a MNIST CNN for white-box attacks.

    Typical attack usage::

        model = WhiteBoxModel.load("checkpoints/mnist_cnn.pt")
        x = x.clone().detach().requires_grad_(True)  # (1,1,28,28) in [0,1]
        logits = model.logits(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad = x.grad  # use this for FGSM / etc.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net = MnistCNN().to(self.device)
        self.net.eval()

    # ------------------------------------------------------------------
    # Forward helpers (differentiable w.r.t. input if x.requires_grad)
    # ------------------------------------------------------------------

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits Z(x). Shape (N, 10)."""
        return self.net(x.to(self.device))

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities F(x)."""
        return F.softmax(self.logits(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted class indices C(x). Shape (N,)."""
        return self.logits(x).argmax(dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "arch": "MnistCNN",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> WhiteBoxModel:
        model = cls(device=device)
        ckpt = torch.load(path, map_location=model.device, weights_only=True)
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.eval()
        return model

    # ------------------------------------------------------------------
    # Data + training
    # ------------------------------------------------------------------

    @staticmethod
    def mnist_loaders(
        data_dir: str | Path = "data",
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        data_dir = Path(data_dir)
        tfm = transforms.ToTensor()  # -> float in [0, 1]
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, test_loader

    def train(
        self,
        epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 128,
        data_dir: str | Path = "data",
        verbose: bool = True,
    ) -> list[float]:
        """Train on MNIST. Returns list of epoch train losses (mean)."""
        train_loader, _ = self.mnist_loaders(data_dir=data_dir, batch_size=batch_size)
        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        history: list[float] = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.net(x), y)
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
            mean_loss = total_loss / max(n, 1)
            history.append(mean_loss)
            if verbose:
                acc = self.accuracy(data_dir=data_dir, batch_size=batch_size)
                print(f"epoch {epoch}/{epochs}  loss={mean_loss:.4f}  test_acc={acc:.4f}")

        self.net.eval()
        return history

    @torch.no_grad()
    def accuracy(
        self,
        data_dir: str | Path = "data",
        batch_size: int = 256,
        split: str = "test",
    ) -> float:
        """Classification accuracy on MNIST train or test split."""
        train_loader, test_loader = self.mnist_loaders(
            data_dir=data_dir, batch_size=batch_size
        )
        loader = train_loader if split == "train" else test_loader
        self.net.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            correct += (self.predict(x) == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)


def _demo() -> None:
    """Train once and save a checkpoint for attack experiments."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    ckpt = root / "checkpoints" / "mnist_cnn.pt"

    model = WhiteBoxModel()
    print(f"device: {model.device}")
    model.train(epochs=3, data_dir=data_dir)
    model.save(ckpt)
    print(f"saved: {ckpt}")
    print(f"final test accuracy: {model.accuracy(data_dir=data_dir):.4f}")


if __name__ == "__main__":
    _demo()
