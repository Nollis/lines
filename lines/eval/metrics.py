"""Primitive-match metric.

Two complementary components, because parameter-only scoring misjudges
representationally-equivalent primitives (a straight arc and a line; two
parameterizations of the same circle):

1. **Geometric match** -- Hungarian-assign predicted to ground-truth primitives
   using a symmetric Chamfer distance between their flattened polylines
   (type-agnostic, image-localizable), then score type accuracy and mean
   normalized parameter (geometric) error over the matched pairs. Unmatched
   primitives become false positives / negatives.
2. **Render-based agreement** -- rasterize both sets and compute IoU over inked
   pixels. This resolves equivalence the parameter space cannot.

``evaluate`` returns a dict of components plus a transparent composite ``score``
in ``[0, 1]`` (higher is better). The composite weights are deliberately simple;
downstream code should rely on the documented properties (perfect on identical
sets, monotonic degradation, false-pos/neg penalties), not the exact weights.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment

from lines.datagen.render import flatten_primitive, render_primitives

# stroke-match tolerance (pixels): thin line-art is hypersensitive to sub-pixel
# misalignment, so render-IoU is computed on strokes dilated by this radius.
_IOU_TOLERANCE_PX = 2

Point = Tuple[float, float]

# composite weights
_W_RENDER = 0.4
_W_TYPE = 0.3
_W_GEOM = 0.3

_CLASSES = ("line", "arc", "circle")

# A predicted primitive is a TRUE match only if its Hungarian-assigned geometric
# error (normalized chamfer / canvas diagonal) is below this and its type is
# correct. ~0.05 of the diagonal ≈ 4.5px at 64px -- tight enough that a tangle of
# spurious lines cannot all be true matches.
_MATCH_THRESHOLD = 0.05


def _empty_per_class():
    return {k: {"n_gt": 0, "n_pred": 0, "n_matched": 0, "type_hits": 0,
                "geometric_error_sum": 0.0,
                "type_accuracy": None, "geometric_error": None} for k in _CLASSES}


def evaluate(pred, gt, canvas, match_threshold: float = _MATCH_THRESHOLD) -> dict:
    """Score a predicted primitive set against ground truth.

    The headline structural metric is ``f1`` (with ``precision`` / ``recall``):
    a predicted primitive is a TRUE match only if Hungarian-assigned to a GT
    primitive with normalized geometric error < ``match_threshold`` AND correct
    type. Over-prediction (tangles) tanks precision; missing primitives tank
    recall. ``render_iou`` and the legacy composite ``score`` are kept as
    diagnostics -- they are too forgiving of structural error on their own.

    The returned dict also includes a ``per_class`` breakdown.
    """
    diag = math.hypot(canvas.width, canvas.height)
    n_pred = len(pred.primitives)
    n_gt = len(gt.primitives)

    render_iou = _render_iou(pred, gt, canvas)
    per_class = _empty_per_class()
    for p in pred.primitives:
        if p.type in per_class:
            per_class[p.type]["n_pred"] += 1
    for g in gt.primitives:
        if g.type in per_class:
            per_class[g.type]["n_gt"] += 1

    if n_pred == 0 or n_gt == 0:
        n_matched = 0
        true_matches = 0
        type_accuracy = 0.0
        geometric_error = 1.0 if (n_pred or n_gt) else 0.0
    else:
        cost = np.zeros((n_pred, n_gt), dtype=float)
        for i, p in enumerate(pred.primitives):
            for j, g in enumerate(gt.primitives):
                cost[i, j] = _chamfer(_poly(p), _poly(g)) / diag
        rows, cols = linear_sum_assignment(cost)
        n_matched = len(rows)
        matched_errors = [cost[i, j] for i, j in zip(rows, cols)]
        geometric_error = float(np.mean(matched_errors))
        type_hits = sum(
            1 for i, j in zip(rows, cols)
            if pred.primitives[i].type == gt.primitives[j].type
        )
        type_accuracy = type_hits / n_matched
        # strict true match: close enough AND correct type
        true_matches = sum(
            1 for i, j in zip(rows, cols)
            if cost[i, j] < match_threshold
            and pred.primitives[i].type == gt.primitives[j].type
        )
        # per-class accumulation, keyed by the GT type of each matched pair
        for i, j in zip(rows, cols):
            kind = gt.primitives[j].type
            if kind not in per_class:
                continue
            per_class[kind]["n_matched"] += 1
            per_class[kind]["geometric_error_sum"] += float(cost[i, j])
            if pred.primitives[i].type == kind:
                per_class[kind]["type_hits"] += 1

    # finalize per-class rates (None when there's nothing to score)
    for kind, stats in per_class.items():
        if stats["n_matched"]:
            stats["type_accuracy"] = stats["type_hits"] / stats["n_matched"]
            stats["geometric_error"] = stats["geometric_error_sum"] / stats["n_matched"]

    false_positives = max(0, n_pred - n_matched)
    false_negatives = max(0, n_gt - n_matched)
    coverage = n_matched / max(n_pred, n_gt, 1)
    geom_score = 1.0 - min(1.0, geometric_error)

    # strict primitive precision / recall / F1 (the structural headline)
    if n_pred == 0:
        precision = 1.0 if n_gt == 0 else 0.0
    else:
        precision = true_matches / n_pred
    if n_gt == 0:
        recall = 1.0 if n_pred == 0 else 0.0
    else:
        recall = true_matches / n_gt
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    score = (
        _W_RENDER * render_iou
        + _W_TYPE * coverage * type_accuracy
        + _W_GEOM * coverage * geom_score
    )

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "true_matches": true_matches,
        "score": float(score),
        "render_iou": float(render_iou),
        "type_accuracy": float(type_accuracy),
        "geometric_error": float(geometric_error),
        "coverage": float(coverage),
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_matched": n_matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "per_class": per_class,
    }


# --- components ---------------------------------------------------------------

def _render_iou(pred, gt, canvas) -> float:
    a = render_primitives(pred, canvas.width, canvas.height) < 128
    b = render_primitives(gt, canvas.width, canvas.height) < 128
    if not a.any() and not b.any():
        return 1.0
    se = _disk(_IOU_TOLERANCE_PX)
    a = binary_dilation(a, structure=se)
    b = binary_dilation(b, structure=se)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def _poly(prim) -> List[Point]:
    return flatten_primitive(prim, 64)


def _chamfer(a: List[Point], b: List[Point]) -> float:
    """Symmetric mean nearest-neighbour distance between two point lists."""
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    # pairwise distances (small polylines -> dense is fine)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))
