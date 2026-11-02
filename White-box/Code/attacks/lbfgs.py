"""Szegedy-style box-constrained L-BFGS-B attack.

    min_{x'}  c * ||x' - x||_2^2 + CE(F(x'), t)
    s.t.      x' in [0, 1]^n

Bounds are enforced by scipy's L-BFGS-B (no manual clamp).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from white_box_model import WhiteBoxModel


def lbfgs(
    model: WhiteBoxModel,
    x: torch.Tensor,
    y: torch.Tensor,
    t: int | None = None,
    c: float = 1.0,
    maxiter: int = 150,
) -> tuple[torch.Tensor, dict]:
    """Targeted L-BFGS-B attack for one fixed tradeoff ``c``.

    Args:
        model: white-box classifier
        x: clean image ``(1, 1, 28, 28)`` in ``[0, 1]``
        y: true label (logging / default target)
        t: target class; default ``(y + 1) % 10``
        c: weight in front of the L2 term
        maxiter: L-BFGS-B iterations

    Returns:
        ``(x_adv, meta)`` with success / L2 / pred / ...
    """
    for p in model.net.parameters():
        p.requires_grad_(False)
    model.net.eval()

    device = model.device
    x = x.detach().to(device)
    shape = tuple(x.shape)
    n = int(np.prod(shape))

    if t is None:
        t = int((y.item() + 1) % 10)
    t_tensor = torch.tensor([t], device=device, dtype=torch.long)

    x_flat = x.detach().cpu().numpy().astype(np.float64).ravel()
    bounds = [(0.0, 1.0)] * n

    def pack(x_adv_np: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(x_adv_np.astype(np.float32))
            .to(device)
            .view(shape)
            .requires_grad_(True)
        )

    def fun_and_grad(x_adv_np: np.ndarray) -> tuple[float, np.ndarray]:
        x_adv = pack(x_adv_np)
        dist = ((x_adv - x) ** 2).sum()
        cls_loss = F.cross_entropy(model.logits(x_adv), t_tensor)
        loss = c * dist + cls_loss
        loss.backward()
        grad = x_adv.grad.detach().cpu().numpy().astype(np.float64).ravel()
        return float(loss.detach().cpu().item()), grad

    x_opt, f_opt, info = fmin_l_bfgs_b(
        fun_and_grad,
        x_flat.copy(),
        bounds=bounds,
        maxiter=maxiter,
    )

    x_adv = torch.from_numpy(x_opt.astype(np.float32)).to(device).view(shape)
    with torch.no_grad():
        pred = int(model.predict(x_adv).item())
        l2 = float(torch.norm((x_adv - x).view(-1), p=2).item())

    meta = {
        "attack": "lbfgs",
        "success": pred == t,
        "target": t,
        "pred": pred,
        "true": int(y.item()),
        "l2": l2,
        "loss": float(f_opt),
        "c": c,
        "warnflag": int(info.get("warnflag", -1)),
        "nit": int(info.get("nit", -1)),
    }
    return x_adv.detach(), meta


def lbfgs_line_search(
    model: WhiteBoxModel,
    x: torch.Tensor,
    y: torch.Tensor,
    t: int | None = None,
    c_list: list[float] | None = None,
    maxiter: int = 150,
) -> tuple[torch.Tensor, dict]:
    """Sweep ``c``; keep successful attack with smallest L2."""
    if c_list is None:
        c_list = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    best_adv, best_meta = None, None
    x_adv, meta = x, {}
    for c in c_list:
        x_adv, meta = lbfgs(model, x, y, t=t, c=c, maxiter=maxiter)
        if meta["success"] and (best_meta is None or meta["l2"] < best_meta["l2"]):
            best_adv, best_meta = x_adv, meta

    if best_adv is None:
        return x_adv, meta
    return best_adv, best_meta
