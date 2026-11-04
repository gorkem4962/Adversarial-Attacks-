"""Run the L-BFGS white-box attack and visualize the result.

Usage (from repo root ``2-White-box`` / ``White-box``):

    python Code/run_lbfgs.py
    python Code/run_lbfgs.py --index 10 --target 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attacks import lbfgs_line_search
from data_utils import load_one_sample
from visualize import show_attack
from white_box_model import WhiteBoxModel

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "mnist_cnn.pt"
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "lbfgs_example.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Szegedy L-BFGS-B attack on MNIST CNN")
    p.add_argument("--index", type=int, default=0, help="MNIST test index")
    p.add_argument("--target", type=int, default=None, help="target class 0-9")
    p.add_argument("--ckpt", type=Path, default=CKPT)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--no-show", action="store_true", help="only save figure, no GUI")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = WhiteBoxModel.load(args.ckpt)
    x, y = load_one_sample(args.data, index=args.index)
    x = x.to(model.device)
    y = y.to(model.device)

    print(f"true label: {y.item()}")
    print(f"pred clean: {model.predict(x).item()}")

    x_adv, info = lbfgs_line_search(model, x, y, t=args.target)
    print(
        f"L-BFGS: success={info['success']}  "
        f"true={info['true']} → target={info['target']}  "
        f"pred={info['pred']}  L2={info['l2']:.4f}  c={info['c']}"
    )
    show_attack(x, x_adv, info, out_path=args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
