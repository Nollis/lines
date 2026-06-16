# Classical Baseline Reference (the bar to beat)

Recorded by Unit 4 (`lines/baselines/classical.py`) via the Unit 3 harness.
The learned model (Unit 5) must beat **mean_score** on a comparable test split.

## Configuration

- Canvas: 128 × 128
- Samples: 200, seeds 900000–900199
- Primitive types: line, arc, circle (1–5 per image)
- Predictor: `ClassicalBaseline()` (skeletonize + line/circle fit, angular-coverage classification)
- Metric: `lines/eval/metrics.py` (Hungarian geometric match + render-IoU, 2px tolerance)

## Reference numbers

| Metric | Value |
|--------|-------|
| mean_score | **0.612** |
| mean_render_iou | 0.500 |
| mean_type_accuracy | 0.879 |
| mean_geometric_error | 0.024 |
| mean_coverage | 0.728 |
| exact primitive count | 0.430 |

## Interpretation

The baseline fits isolated strokes well (high type accuracy, low geometric error)
but loses **coverage and primitive count** when strokes cross or touch — multiple
primitives merge into a single skeleton component and are mis-counted. This is the
junction regime where the learned set-prediction model is expected to win.

Regenerate with the snippet in the Unit 4 section of the plan / project history.
