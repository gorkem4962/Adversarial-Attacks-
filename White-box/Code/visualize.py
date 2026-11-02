"""Plotting helpers for adversarial examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def show_attack(
    x: torch.Tensor,
    x_adv: torch.Tensor,
    info: dict,
    out_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    """Show clean / adversarial / |diff| side by side."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(x.squeeze().detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axs[0].set_title(f"clean → {info.get('true', '?')}")
    axs[1].imshow(x_adv.squeeze().detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axs[1].set_title(f"adv → {info.get('pred', '?')} (t={info.get('target', '?')})")
    diff = (x_adv - x).squeeze().detach().cpu().abs()
    axs[2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()), 1e-6))
    axs[2].set_title(f"|diff|  L2={info.get('l2', float('nan')):.3f}")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()

    saved: Path | None = None
    if out_path is not None:
        saved = Path(out_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saved, dpi=150)
        print(f"saved: {saved}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved
