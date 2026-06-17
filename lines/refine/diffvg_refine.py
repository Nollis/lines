"""Differentiable-render refinement (the DiffVG-style snapping pass, Unit 6).

Takes the model's predicted primitives and optimizes their *geometry* (not
type) by gradient descent so the rendered set matches the input ink. Unlike the
training :class:`SoftRenderer`, this renders every primitive -- including arcs --
as a differentiable polyline, so arc geometry is refined correctly.

The optimization is per-image and joint over all kept primitives: every
primitive is rendered to a soft distance field, the fields are combined with a
soft-OR, and the composite is matched to the target raster by MSE. The soft-OR
lets each primitive claim its own ink without hand-assigning pixels.

No external dependency (no `diffvg` build) -- the renderer is plain PyTorch.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from lines.models.encoding import (
    TYPE_ARC, TYPE_CIRCLE, TYPE_LINE, encode_primitive, decode_primitive,
)
from lines.primitives import PrimitiveSet

# bulge magnitude is clamped away from 0 during refinement so the arc->center
# conversion stays numerically stable (an arc never collapses to a line here).
_MIN_BULGE = 0.02


def _build_grid(size: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(0.5 / size, 1.0 - 0.5 / size, size),
        torch.linspace(0.5 / size, 1.0 - 0.5 / size, size),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1)  # (H, W, 2), normalized coords


def _arc_polyline(params: torch.Tensor, n: int) -> torch.Tensor:
    """Differentiable arc -> (n, 2) points, mirroring primitives.Arc.to_center_params."""
    p1 = params[0:2]
    p2 = params[2:4]
    bulge = params[4]
    sign = torch.sign(bulge.detach())
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    bulge = sign * torch.clamp(bulge.abs(), min=_MIN_BULGE)

    theta = 4.0 * torch.atan(bulge)
    half = theta / 2.0
    chord_vec = p2 - p1
    chord = torch.linalg.vector_norm(chord_vec).clamp_min(1e-6)
    r_signed = chord / (2.0 * torch.sin(half))
    mid = (p1 + p2) / 2.0
    u = chord_vec / chord
    normal = torch.stack([-u[1], u[0]])
    apothem = r_signed * torch.cos(half)
    center = mid + normal * apothem
    start = torch.atan2(p1[1] - center[1], p1[0] - center[0])
    ts = torch.linspace(0.0, 1.0, n, device=params.device)
    angles = start + ts * theta
    pts = torch.stack([center[0] + r_signed.abs() * torch.cos(angles),
                       center[1] + r_signed.abs() * torch.sin(angles)], dim=-1)
    return pts


def _line_polyline(params: torch.Tensor, n: int) -> torch.Tensor:
    p1 = params[0:2]
    p2 = params[2:4]
    ts = torch.linspace(0.0, 1.0, n, device=params.device).unsqueeze(-1)
    return p1.unsqueeze(0) * (1.0 - ts) + p2.unsqueeze(0) * ts


def _circle_polyline(params: torch.Tensor, n: int) -> torch.Tensor:
    center = params[0:2]
    radius = params[2]
    angles = torch.linspace(0.0, 2.0 * math.pi, n, device=params.device)
    return torch.stack([center[0] + radius * torch.cos(angles),
                        center[1] + radius * torch.sin(angles)], dim=-1)


def _polyline(type_id: int, params: torch.Tensor, n: int) -> torch.Tensor:
    if type_id == TYPE_LINE:
        return _line_polyline(params, n)
    if type_id == TYPE_CIRCLE:
        return _circle_polyline(params, n)
    if type_id == TYPE_ARC:
        return _arc_polyline(params, n)
    raise ValueError(f"cannot refine primitive type id {type_id}")


def _soft_polyline_image(points: torch.Tensor, grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Soft ink image (H, W) = exp(-d^2 / 2 sigma^2), d = distance to the polyline."""
    a = points[:-1]                       # (S, 2)
    b = points[1:]                        # (S, 2)
    v = b - a                             # (S, 2)
    l2 = (v * v).sum(-1).clamp_min(1e-9)  # (S,)
    g = grid.unsqueeze(2)                 # (H, W, 1, 2)
    w = g - a                             # (H, W, S, 2)
    t = (w * v).sum(-1) / l2              # (H, W, S)
    t = t.clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * v     # (H, W, S, 2)
    d2 = ((g - closest) ** 2).sum(-1)     # (H, W, S)
    d2min = d2.min(dim=-1).values         # (H, W)
    return torch.exp(-d2min / (2.0 * sigma * sigma))


def refine_primitives(
    primitives: PrimitiveSet,
    image: np.ndarray,
    canvas,
    steps: int = 50,
    lr: float = 8e-3,
    sigma_px: float = 1.5,
    sigma_start_px: float = 5.0,
    render_size: int | None = None,
    n_points: int = 48,
) -> PrimitiveSet:
    """Refine primitive geometry to match the input ink via gradient descent.

    ``image`` is ``HxW`` uint8 with 0 = ink, 255 = background (dataset format).
    Types are held fixed; only geometric parameters are optimized.

    Two robustness measures prevent the classic failure where a mostly-empty
    target makes an *empty* render the trivial MSE minimizer:

    * **sigma annealing** -- the soft-stroke width shrinks from ``sigma_start_px``
      to ``sigma_px`` over the steps, giving a broad gradient capture basin
      early (so a primitive feels a target stroke several pixels away) that
      sharpens for precise final placement.
    * **foreground-weighted loss** -- ink pixels are up-weighted by the
      background/foreground ratio so collapsing to blank is heavily penalized.
    """
    if not primitives.primitives:
        return PrimitiveSet([])

    size = render_size or canvas.width
    grid = _build_grid(size)

    # target foreground in [0, 1] at the render resolution (1 = ink)
    target = torch.from_numpy(1.0 - image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    if target.shape[-1] != size:
        target = F.interpolate(target, size=(size, size), mode="area")
    target = target[0, 0]

    # balanced pixel weights: up-weight ink so empty is not the minimizer
    fg_frac = target.mean().clamp(1e-4, 1.0 - 1e-4)
    weight = torch.where(target > 0.5, 0.5 / fg_frac, 0.5 / (1.0 - fg_frac))

    # encode each primitive to a normalized 5-vector leaf tensor
    type_ids, leaves = [], []
    for prim in primitives.primitives:
        t, p = encode_primitive(prim, canvas.width, canvas.height)
        type_ids.append(int(t))
        leaves.append(torch.tensor(p, dtype=torch.float32, requires_grad=True))

    opt = torch.optim.Adam(leaves, lr=lr)
    for step in range(steps):
        frac = step / max(1, steps - 1)
        sigma_px_t = sigma_start_px + (sigma_px - sigma_start_px) * frac
        sigma = sigma_px_t / size
        composite = torch.zeros_like(target)
        for tid, params in zip(type_ids, leaves):
            pts = _polyline(tid, params, n_points)
            composite = composite + _soft_polyline_image(pts, grid, sigma)
        composite = composite.clamp(0.0, 1.0)
        loss = (weight * (composite - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    refined = []
    for tid, params in zip(type_ids, leaves):
        prim = decode_primitive(tid, params.detach().numpy(), canvas.width, canvas.height)
        if prim is not None and prim.is_valid():
            refined.append(prim)
        else:
            # keep the original if refinement produced something degenerate
            refined.append(primitives.primitives[len(refined)])
    return PrimitiveSet(refined)


class RefiningPredictor:
    """Wrap any ``image -> PrimitiveSet`` predictor with a diffvg refinement pass.

    Lets the differentiable-render snapping be evaluated as an isolated ablation
    on top of a base predictor (e.g. the model with no other refinement).
    """

    def __init__(self, base_predictor, canvas, **refine_kwargs):
        self.base = base_predictor
        self.canvas = canvas
        self.refine_kwargs = refine_kwargs

    def __call__(self, image: np.ndarray) -> PrimitiveSet:
        pset = self.base(image)
        return refine_primitives(pset, image, self.canvas, **self.refine_kwargs)
