# Sim-to-real probe (Unit 7)

The entire project rests on one assumption (the brainstorm's R6): the model
must work on images whose source geometry is unavailable — i.e. it must
generalize beyond the exact renderer that produced its training data. Every
other number in this repo is measured on synthetic test images from the *same*
generator as the training set, so this is the one experiment that actually tests
the load-bearing claim.

## Method

The probe re-renders the **same primitive sets** as `data/test128` (identical
ground truth) through an **independent rasterizer** (OpenCV, native
circle/ellipse/line algorithms — a different AA and stroke model than the
training Pillow + supersampled-polyline path). A second variant adds JPEG
compression at quality 60, because the project's real inputs are jpgs.

Only the rasterizer changes, so any score drop isolates rasterization-domain
shift rather than a difference in geometry. Build with:

    python scripts/build_probe.py --src data/test128 --out data/probe128_cv2
    python scripts/build_probe.py --src data/test128 --out data/probe128_cv2_jpeg --jpeg-quality 60

## Result (full 400 samples, threshold 0.50 + algebraic refine)

| Predictor | in-dist (ours) | OpenCV | OpenCV + JPEG | Drop |
|-----------|----------------|--------|---------------|------|
| Classical baseline | 0.618 | 0.609 | 0.609 | −0.009 (−1.5%) |
| Model | 0.590 | 0.574 | 0.569 | −0.021 (−3.6%) |

## Findings

1. **The model generalizes; it does not collapse.** A completely different
   rasterizer plus JPEG compression costs only ~3.6% of score. The synthetic
   approach is not overfit to the training renderer's anti-aliasing — the
   central risk of the whole project is measured and it is modest.
2. **The model is ~2x more sensitive than the classical fitter** (−3.6% vs
   −1.5%), as expected: it learned pixel statistics where skeletonize+fit works
   on geometry. Still small in absolute terms.
3. **JPEG adds almost nothing** beyond the rasterizer shift (−0.005). Compression
   at q60 is not a meaningful threat for this task.
4. Geometric error *improved* on the probes (0.077 → 0.066): OpenCV draws bolder
   strokes, giving the local refinement more ink to fit.

## What this does and does not establish

Establishes: the model is robust to **rasterization-domain shift** — different
AA, stroke weight, and compression on the same kind of content.

Does **not** yet establish robustness to:
- genuinely different *content* (the primitives still come from our sampler's
  distribution);
- hand-authored or third-party vector art rasterized at scale;
- scanned / photographed inputs (a different sim-to-real regime entirely,
  explicitly out of scope for the "clean digital renders" target).

The natural next probe is a small set of **third-party clean renders** (e.g. an
SVG icon library rasterized independently) with hand-specified primitive ground
truth — content the generator never produced.
