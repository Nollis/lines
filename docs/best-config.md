# Best known inference configurations

Confirmed on the full 400-sample held-out test splits (`data/test{64,128}`).
The metric is the composite primitive-match `score` (higher is better); the bar
is the classical baseline at the same resolution.

The auto-generated checkpoint ledger ([results.md](results.md)) scores every
checkpoint with the *default* predictor. This page records the best *inference
configuration* per model — threshold and refinement strategy matter, and the
optimum differs by resolution.

## Headline result: the model beats the classical baseline at both resolutions

| Resolution | Best model config | Score | Baseline | Margin |
|-----------|-------------------|-------|----------|--------|
| 64×64  | `v1_optimized`, threshold 0.85, **algebraic** refine | **0.620** | 0.551 | **+0.069** |
| 128×128 | `v1_warmstart_128`, threshold 0.50, **diffvg** refine | **0.636** | 0.618 | **+0.018** |

R3 (the brainstorm's gate — beat classical vectorization) is satisfied at both
resolutions. The 64 win is decisive; the 128 win is narrow and **requires
differentiable-render refinement** (the cheaper algebraic refine scores 0.590 at
128 and does *not* clear baseline).

## Reproduce

    # 64 (best is the default refine strategy)
    python -m lines.eval.score_checkpoint checkpoints/v1_optimized/model.pt \
        --none-threshold 0.85 --refine algebraic

    # 128 (needs the strict threshold + diffvg refinement; slow, ~19 min)
    python -m lines.eval.score_checkpoint checkpoints/v1_warmstart_128/model.pt \
        --none-threshold 0.50 --refine diffvg

## Why the configs differ by resolution

Two independent levers, each fixing a different deficit, and their optimum is
not the same at both sizes:

- **Threshold (coverage).** The 128 warm-start model *over-predicts*: at the
  default 0.85 it emits ~411 primitives for 271 ground-truth. Tightening to 0.50
  cuts that to ~302 and lifts coverage 0.671 → 0.776. The 64 model does **not**
  share this pathology — at 0.50 its score drops (0.620 → 0.533), so the strict
  threshold is a 128-specific fix, not a global one. **Do not change the
  predictor default.**
- **Refinement (precision).** diffvg (differentiable-render snapping, Unit 6)
  is the strongest precision pass. At 128 it lifts render-IoU to 0.541 (vs
  baseline 0.509) where algebraic re-fitting reaches only 0.438. It is the
  difference between losing (0.590) and winning (0.636) at 128.

## Cost note

diffvg refinement is per-image gradient descent: ~3 s/image at 128 (≈19 min for
the 400-sample split) vs ~0.015 s/image for algebraic. Use algebraic for fast
iteration; reserve diffvg for final scoring or when precision is the priority.

## Open lever

At 128, geometric error after diffvg is still 0.065 vs baseline's 0.023 — the
model's *placement* trails the classical least-squares fitter even after
snapping. The margin is carried by coverage and render-IoU. Closing the geometry
gap (more capacity at 128, or a sharper refinement objective) is where further
gains live.
