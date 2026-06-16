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

_MEAN_KEYS = ("score", "render_iou", "type_accuracy", "geometric_error", "coverage")


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
    return report
