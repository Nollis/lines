"""Score a checkpoint (or the classical baseline) against a synthetic test split.

Usage::

    python -m lines.eval.score_checkpoint <checkpoint.pt>
    python -m lines.eval.score_checkpoint --baseline --canvas-side 64

The split is auto-located by canvas side at ``data/test{N}/``. If it's missing
the script generates 400 deterministic samples there. Results are printed and
saved next to the checkpoint as ``score.json`` for the ledger to pick up.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from lines.baselines.classical import ClassicalBaseline
from lines.datagen.dataset import Dataset, write_dataset
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor
from lines.models.set_predictor import load_set_predictor
from lines.train.predictor import ModelPredictor

# Deterministic split: a high seed so it never collides with training data.
_TEST_SEED = 900_000
_TEST_SAMPLES = 400


def ensure_test_split(canvas_side: int, n_samples: int = _TEST_SAMPLES,
                      root: str = "data") -> Path:
    """Return the path to ``data/test{canvas_side}/``, generating it if absent."""
    split = Path(root) / f"test{canvas_side}"
    if (split / "manifest.json").exists():
        return split
    canvas = Canvas(canvas_side, canvas_side)
    write_dataset(split, n_samples, seed=_TEST_SEED, canvas=canvas, randomize=False)
    return split


def score_checkpoint(checkpoint_path: Path, root: str = "data") -> dict:
    model, cfg = load_set_predictor(checkpoint_path)
    canvas_side = int(cfg.get("canvas_side", 64))
    canvas = Canvas(canvas_side, canvas_side)
    split = ensure_test_split(canvas_side, root=root)
    ds = Dataset(split)
    predictor = ModelPredictor(model, canvas)
    t0 = time.time()
    report = run_predictor(predictor, ds, canvas)
    elapsed = time.time() - t0
    return {
        "kind": "model",
        "checkpoint": str(checkpoint_path),
        "name": Path(checkpoint_path).parent.name,
        "canvas_side": canvas_side,
        "encoder_type": cfg.get("encoder_type"),
        "n_samples": len(ds),
        "elapsed_s": elapsed,
        "report": _strip_per_sample(report),
        "cfg": cfg,
    }


def score_baseline(canvas_side: int, root: str = "data") -> dict:
    canvas = Canvas(canvas_side, canvas_side)
    split = ensure_test_split(canvas_side, root=root)
    ds = Dataset(split)
    t0 = time.time()
    report = run_predictor(ClassicalBaseline(), ds, canvas)
    elapsed = time.time() - t0
    return {
        "kind": "baseline",
        "name": f"baseline_{canvas_side}",
        "canvas_side": canvas_side,
        "n_samples": len(ds),
        "elapsed_s": elapsed,
        "report": _strip_per_sample(report),
    }


def _strip_per_sample(report: dict) -> dict:
    out = {k: v for k, v in report.items() if k != "per_sample"}
    return out


def _format_summary(result: dict) -> str:
    r = result["report"]
    pc = r.get("per_class", {})
    line = pc.get("line", {})
    arc = pc.get("arc", {})
    circ = pc.get("circle", {})

    def _f(x, fmt="{:.3f}"):
        return "n/a" if x is None else fmt.format(x)

    s = [
        f"name           : {result['name']}",
        f"canvas         : {result['canvas_side']}x{result['canvas_side']}  samples={result['n_samples']}  elapsed={result['elapsed_s']:.1f}s",
        f"mean_score     : {r['mean_score']:.3f}",
        f"  render_iou   : {r['mean_render_iou']:.3f}",
        f"  type_acc     : {r['mean_type_accuracy']:.3f}",
        f"  geom_error   : {r['mean_geometric_error']:.3f}",
        f"  coverage     : {r['mean_coverage']:.3f}",
        "per-class (n_gt / n_pred / geometric_error / type_accuracy / recall):",
        f"  line   gt={line.get('n_gt',0):>4}  pred={line.get('n_pred',0):>4}  "
        f"geom={_f(line.get('geometric_error'))}  type={_f(line.get('type_accuracy'))}  "
        f"recall={_f(line.get('recall'))}",
        f"  arc    gt={arc.get('n_gt',0):>4}  pred={arc.get('n_pred',0):>4}  "
        f"geom={_f(arc.get('geometric_error'))}  type={_f(arc.get('type_accuracy'))}  "
        f"recall={_f(arc.get('recall'))}",
        f"  circle gt={circ.get('n_gt',0):>4}  pred={circ.get('n_pred',0):>4}  "
        f"geom={_f(circ.get('geometric_error'))}  type={_f(circ.get('type_accuracy'))}  "
        f"recall={_f(circ.get('recall'))}",
    ]
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?",
                    help="path to checkpoint .pt (omit when using --baseline)")
    ap.add_argument("--baseline", action="store_true",
                    help="score the classical baseline instead of a checkpoint")
    ap.add_argument("--canvas-side", type=int, default=64,
                    help="canvas side for --baseline (model uses its own cfg)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--save-json", action="store_true",
                    help="write score.json next to the checkpoint")
    args = ap.parse_args()

    if args.baseline:
        result = score_baseline(args.canvas_side, root=args.data_root)
    else:
        if not args.checkpoint:
            ap.error("checkpoint path required when --baseline is not set")
        result = score_checkpoint(Path(args.checkpoint), root=args.data_root)
        if args.save_json:
            out = Path(args.checkpoint).parent / "score.json"
            out.write_text(json.dumps(result, indent=2, default=str))

    print(_format_summary(result))
    print()


if __name__ == "__main__":
    main()
