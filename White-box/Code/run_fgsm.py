"""Run FGSM (untargeted or targeted) and visualize.

Usage:

    python Code/run_fgsm.py
    python Code/run_fgsm.py --targeted --target 1
    python Code/run_fgsm.py --epsilon 0.2 --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attacks import fgsm, fgsm_line_search
from data_utils import load_one_sample
from visualize import show_attack
from white_box_model import WhiteBoxModel

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "mnist_cnn.pt"
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "fgsm_example.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Goodfellow FGSM on MNIST CNN")
    p.add_argument("--index", type=int, default=0, help="MNIST test index")
    p.add_argument("--targeted", action="store_true", help="targeted FGSM (needs --target)")
    p.add_argument("--target", type=int, default=None, help="target class 0-9")
    p.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="fixed ε; if omitted, line-search over several ε",
    )
    p.add_argument("--ckpt", type=Path, default=CKPT)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.targeted and args.target is None:
        raise SystemExit("targeted FGSM requires --target")

    model = WhiteBoxModel.load(args.ckpt)
    x, y = load_one_sample(args.data, index=args.index)
    x = x.to(model.device)
    y = y.to(model.device)

    print(f"true label: {y.item()}")
    print(f"pred clean: {model.predict(x).item()}")

    if args.epsilon is not None:
        x_adv, info = fgsm(
            model, x, y, t=args.target, epsilon=args.epsilon, targeted=args.targeted
        )
    else:
        x_adv, info = fgsm_line_search(
            model, x, y, t=args.target, targeted=args.targeted
        )

    mode = "targeted" if info["targeted"] else "untargeted"
    print(
        f"FGSM ({mode}): success={info['success']}  "
        f"true={info['true']} → pred={info['pred']}  "
        f"target={info['target']}  eps={info['epsilon']:.3f}  "
        f"L∞={info['linf']:.4f}  L2={info['l2']:.4f}"
    )
    show_attack(x, x_adv, info, out_path=args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
