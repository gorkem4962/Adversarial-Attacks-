"""Goodfellow FGSM (Fast Gradient Sign Method).

Untargeted (default):
    x' = x + ε · sign(∇_x CE(F(x), y))     # increase loss on true label

Targeted:
    x' = x - ε · sign(∇_x CE(F(x), t))     # decrease loss on target t

ε bounds the L∞ perturbation; pixels are kept in [0, 1].
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from white_box_model import WhiteBoxModel


def fgsm(
    model: WhiteBoxModel,
    x: torch.Tensor,
    y: torch.Tensor,
    t: int | None = None,
    epsilon: float = 0.1,
    targeted: bool = False,
) -> tuple[torch.Tensor, dict]:
    """One-step FGSM attack.

    Args:
        model: white-box classifier
        x: clean image ``(N, 1, 28, 28)`` in ``[0, 1]``
        y: true labels ``(N,)``
        t: target class (required if ``targeted=True``; ignored otherwise)
        epsilon: L∞ step size / budget
        targeted: if True, move toward class ``t``; else away from ``y``
    """
    for p in model.net.parameters():
        p.requires_grad_(False)
    model.net.eval()

    device = model.device
    x = x.detach().to(device)
    y = y.detach().to(device).long().view(-1)

    if targeted:
        if t is None:
            raise ValueError("targeted FGSM needs a target class t")
        label = torch.full_like(y, int(t))
    else:
        label = y
        t = None

    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model.logits(x_adv), label)
    model.net.zero_grad(set_to_none=True)
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    # + increases loss (untargeted); - decreases loss toward t (targeted)
    direction = -1.0 if targeted else 1.0
    x_adv = (x_adv.detach() + direction * epsilon * grad_sign).clamp(0.0, 1.0)

    with torch.no_grad():
        pred = model.predict(x_adv)
        linf = float((x_adv - x).abs().amax().item())
        l2 = float(torch.norm((x_adv - x).view(x_adv.size(0), -1), p=2, dim=1).mean().item())
        if targeted:
            success = bool((pred == label).all().item())
        else:
            success = bool((pred != y).all().item())

    meta = {
        "attack": "fgsm",
        "targeted": targeted,
        "success": success,
        "target": int(t) if targeted else None,
        "pred": int(pred[0].item()) if pred.numel() == 1 else pred.detach().cpu().tolist(),
        "true": int(y[0].item()) if y.numel() == 1 else y.detach().cpu().tolist(),
        "epsilon": float(epsilon),
        "linf": linf,
        "l2": l2,
        "loss": float(loss.detach().cpu().item()),
    }
    return x_adv.detach(), meta


def fgsm_line_search(
    model: WhiteBoxModel,
    x: torch.Tensor,
    y: torch.Tensor,
    t: int | None = None,
    targeted: bool = False,
    epsilon_list: list[float] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Try several ε; keep successful attack with smallest ε (then smallest L2).

    Default ε up to 0.1 (10% of the [0, 1] range).
    """
    if epsilon_list is None:
        epsilon_list = [0.01, 0.03, 0.05, 0.07, 0.1]

    if targeted and t is None:
        t = int((y.view(-1)[0].item() + 1) % 10)

    best_adv, best_meta = None, None
    x_adv, meta = x, {}
    for eps in epsilon_list:
        x_adv, meta = fgsm(model, x, y, t=t, epsilon=eps, targeted=targeted)
        if meta["success"]:
            if best_meta is None or eps < best_meta["epsilon"] or (
                eps == best_meta["epsilon"] and meta["l2"] < best_meta["l2"]
            ):
                best_adv, best_meta = x_adv, meta

    if best_adv is None:
        return x_adv, meta
    return best_adv, best_meta
