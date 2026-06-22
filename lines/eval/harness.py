"""Run a predictor over a dataset split and aggregate metric scores.

A *predictor* is any callable ``image -> PrimitiveSet``. The baseline (Unit 4),
the model (Unit 5), and the refined model (Unit 6) all satisfy this signature,
so they are scored through the exact same path -- the comparison that answers
"does the model beat the baseline". The same harness runs the synthetic test
split and the real-image probe set.
"""

from __future__ import annotations

from typing import Callable

from lines.eval.metrics import evaluate

_MEAN_KEYS = ("f1", "precision", "recall", "score", "render_iou",
              "type_accuracy", "geometric_error", "coverage")


def run_predictor(predictor: Callable, dataset, canvas) -> dict:
    """Evaluate ``predictor`` over every sample in ``dataset``.

    Returns a report with per-sample metrics and ``mean_<key>`` aggregates.
    """
    per_sample = []
    for i in range(len(dataset)):
        image, gt = dataset[i]
        pred = predictor(image)
        metrics = evaluate(pred, gt, canvas)
        metrics["id"] = i
        per_sample.append(metrics)

    report = {"per_sample": per_sample, "n": len(per_sample)}
    for key in _MEAN_KEYS:
        if per_sample:
            report[f"mean_{key}"] = sum(s[key] for s in per_sample) / len(per_sample)
        else:
            report[f"mean_{key}"] = 0.0

    # per-image distribution: surfaces what mean-F1 averages away.
    # exact = "the whole drawing reconstructed" (F1 ~ 1.0)
    # near  = "at most ~1 wrong edge in ~10" (F1 >= 0.9)
    n_perfect = sum(1 for s in per_sample if s["f1"] >= 0.99)
    n_near = sum(1 for s in per_sample if s["f1"] >= 0.9)
    report["n_perfect"] = n_perfect
    report["n_near"] = n_near
    report["exact_match_rate"] = n_perfect / len(per_sample) if per_sample else 0.0
    report["near_match_rate"] = n_near / len(per_sample) if per_sample else 0.0

    # per-class aggregates: sum n_gt/n_pred/n_matched/type_hits/geom_error_sum
    # across the dataset, then derive rates -- not the mean-of-means, which
    # would over-weight samples with few primitives.
    classes = ("line", "arc", "circle")
    per_class = {
        k: {"n_gt": 0, "n_pred": 0, "n_matched": 0, "type_hits": 0,
            "geometric_error_sum": 0.0}
        for k in classes
    }
    for s in per_sample:
        for k in classes:
            pc = s["per_class"][k]
            per_class[k]["n_gt"] += pc["n_gt"]
            per_class[k]["n_pred"] += pc["n_pred"]
            per_class[k]["n_matched"] += pc["n_matched"]
            per_class[k]["type_hits"] += pc["type_hits"]
            per_class[k]["geometric_error_sum"] += pc["geometric_error_sum"]
    for k in classes:
        stats = per_class[k]
        nm = stats["n_matched"]
        stats["type_accuracy"] = stats["type_hits"] / nm if nm else None
        stats["geometric_error"] = stats["geometric_error_sum"] / nm if nm else None
        # recall = of GT primitives of this class, how many got any match
        stats["recall"] = nm / stats["n_gt"] if stats["n_gt"] else None
    report["per_class"] = per_class
    return report
