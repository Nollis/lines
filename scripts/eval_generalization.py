"""Score a checkpoint (and the classical baseline) across the generalization
splits: in-distribution random content, trained-family structured content, and
held-out structured content -- each under both renderers.

Used to measure whether enriching the training generator with structured
layouts closes the content-shift gap, and whether it generalizes to held-out
structural relationships.
"""

from __future__ import annotations

import argparse

from lines.baselines.classical import ClassicalBaseline
from lines.datagen.dataset import Dataset
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor
from lines.models.set_predictor import load_set_predictor
from lines.train.predictor import ModelPredictor

SPLITS = {
    "random (in-dist)": "data/test128",
    "technical/ours": "data/probe_tech128_ours",
    "technical/cv2": "data/probe_tech128_cv2",
    "heldout/ours": "data/probe_heldout128_ours",
    "heldout/cv2": "data/probe_heldout128_cv2",
}


def _score(pred, canvas, label):
    print(f"--- {label} ---")
    for name, path in SPLITS.items():
        r = run_predictor(pred, Dataset(path), canvas)
        print(f"  {name:20s} score={r['mean_score']:.3f} iou={r['mean_render_iou']:.3f} "
              f"type={r['mean_type_accuracy']:.3f} geom={r['mean_geometric_error']:.3f} "
              f"cov={r['mean_coverage']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--none-threshold", type=float, default=0.50)
    ap.add_argument("--with-baseline", action="store_true")
    args = ap.parse_args()

    model, cfg = load_set_predictor(args.checkpoint)
    canvas = Canvas(cfg["canvas_side"], cfg["canvas_side"])
    if args.with_baseline:
        _score(ClassicalBaseline(), canvas, "BASELINE (classical)")
    _score(ModelPredictor(model, canvas, none_prob_threshold=args.none_threshold,
                          refine_distance=6.0),
           canvas,
           f"MODEL {args.checkpoint} (thr{args.none_threshold} + algebraic)")


if __name__ == "__main__":
    main()
