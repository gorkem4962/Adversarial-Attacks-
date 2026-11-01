"""Minimal white-box starter — implement your attacks here.

Run after training:
    python Code/white_box_model.py
    python Code/attack_starter.py
"""

from pathlib import Path
import sys

import torch
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from white_box_model import WhiteBoxModel

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "mnist_cnn.pt"
DATA = ROOT / "data"


def load_one_sample(index: int = 0):
    ds = datasets.MNIST(DATA, train=False, download=False, transform=transforms.ToTensor())
    x, y = ds[index]
    # x: (1, 28, 28) -> (1, 1, 28, 28)
    return x.unsqueeze(0), torch.tensor([y])

def lbfgs(model, x, y):
    """
    L-BFGS attack implementation 
    """

    pass


def main() -> None:
    model = WhiteBoxModel.load(CKPT)
    x, y = load_one_sample(0)
    x = x.to(model.device)
    y = y.to(model.device)

    print(f"true label: {y.item()}")
    print(f"pred clean: {model.predict(x).item()}")

    # --- your FGSM / PGD / C&W goes below ---
    # Example gradient hook (untargeted FGSM sketch):
    #
    # x_adv = x.clone().detach().requires_grad_(True)
    # loss = F.cross_entropy(model.logits(x_adv), y)
    # loss.backward()
    # grad = x_adv.grad.detach()
    # ...
    #


if __name__ == "__main__":
    main()
