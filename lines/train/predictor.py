"""Adapt a trained :class:`SetPredictor` to the predictor protocol the eval
harness expects: ``image (numpy HxW uint8) -> PrimitiveSet``.

A "none" probability threshold filters out empty queries; the rest are decoded
into primitives in the requested canvas coordinates.
"""

from __future__ import annotations

import numpy as np
import torch

from lines.datagen.sampler2d import Canvas
from lines.models.encoding import N_TYPES, TYPE_NONE, decode_set
from lines.primitives import PrimitiveSet, Line, Circle, Arc
from lines.datagen.render import flatten_primitive
from lines.baselines.classical import _fit_circle, _terminal_pair, _arc_from_points


class ModelPredictor:
    def __init__(self, model, canvas: Canvas, none_prob_threshold: float = 0.85,
                 non_none_prob_threshold: float = 0.15, refine_distance: float | None = 6.0,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.canvas = canvas
        self.threshold = none_prob_threshold
        self.min_prob = non_none_prob_threshold
        self.refine_distance = refine_distance
        self.device = device

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> PrimitiveSet:
        x = torch.from_numpy(image.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)             # (1, 1, H, W)
        logits, params = self.model(x)                              # (1, N, K), (1, N, P)
        probs = logits.softmax(dim=-1)[0]                           # (N, K)
        
        # Keep slots where the probability of NONE is below self.threshold,
        # AND at least one non-NONE class is somewhat confident (above self.min_prob)
        none_p = probs[:, TYPE_NONE]
        max_non_none_p = probs[:, :TYPE_NONE].max(dim=-1).values
        keep = (none_p < self.threshold) & (max_non_none_p > self.min_prob)
        
        if not keep.any():
            pred_set = PrimitiveSet([])
        else:
            # For the kept slots, assign the type with the highest probability among the non-NONE types
            non_none_probs = probs[keep, :TYPE_NONE]                    # (keep_count, 3)
            kept_types = non_none_probs.argmax(dim=-1).cpu().numpy()
            kept_params = params[0][keep].cpu().numpy()
            pred_set = decode_set(kept_types, kept_params,
                                  self.canvas.width, self.canvas.height)

        # Optional local algebraic refinement
        if self.refine_distance is not None and len(pred_set.primitives) > 0:
            # Get all inked points (0 is foreground/ink in our dataset format)
            ink_y, ink_x = np.nonzero(image < 128)
            if len(ink_y) > 0:
                ink_pts = np.column_stack([ink_x.astype(float), ink_y.astype(float)])  # (M, 2)
                refined_prims = []
                for p in pred_set.primitives:
                    poly_pts = np.asarray(flatten_primitive(p, 64), dtype=float)
                    dists = np.linalg.norm(ink_pts[:, None, :] - poly_pts[None, :, :], axis=2).min(axis=1)
                    local_pts = ink_pts[dists < self.refine_distance]
                    
                    if len(local_pts) < 5:
                        refined_prims.append(p)
                        continue
                    
                    try:
                        if p.type == "line":
                            a, b = _terminal_pair(local_pts)
                            refined_prims.append(Line(p1=a, p2=b))
                        elif p.type == "circle":
                            (cx, cy), r, _ = _fit_circle(local_pts)
                            refined_prims.append(Circle(center=(cx, cy), radius=r))
                        elif p.type == "arc":
                            (cx, cy), r, _ = _fit_circle(local_pts)
                            fit_p = _arc_from_points(local_pts, cx, cy)
                            if fit_p is not None and fit_p.is_valid():
                                refined_prims.append(fit_p)
                            else:
                                refined_prims.append(p)
                        else:
                            refined_prims.append(p)
                    except Exception:
                        refined_prims.append(p)
                pred_set = PrimitiveSet(refined_prims)

        return pred_set
