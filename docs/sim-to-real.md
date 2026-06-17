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

---

# Sim-to-real probe v2: content-distribution shift

Probe v1 changed only the rasterizer. Probe v2 changes the **content
distribution**: an independent generator (`lines/datagen/technical_layout.py`)
produces structured *technical-drawing* layouts the training sampler never makes
— concentric circles, bolt-hole patterns, rectangles, crosshairs, tangent
circles, filleted corners — with exact ground truth. Each layout is rendered
through both the training renderer (isolates content shift) and OpenCV
(content + rasterizer shift). Build with:

    python scripts/build_technical_probe.py --out-root data/probe_tech128 --n 300

## Result (300 samples, threshold 0.50 + algebraic refine)

| Predictor | random/ours (in-dist) | technical/ours | technical/cv2 |
|-----------|----------------------|----------------|---------------|
| Classical baseline | 0.618 | 0.541 (−12%) | 0.538 (−13%) |
| Model | 0.590 | 0.460 (−22%) | 0.498 (−16%) |

## Findings

1. **Content shift hurts far more than rasterizer shift** — 12–22% here vs ~4%
   in probe v1, for *both* methods. Structured layouts are simply harder than
   random shape-soup.
2. **The model is more brittle to content shift than the classical baseline**
   (−22% vs −12% on technical/ours). It partly overfit to the training
   *distribution of arrangements*, not just the pixels.
3. **The two shifts are not additive** — the model scores *better* on
   technical/cv2 (0.498) than technical/ours (0.460), because OpenCV's bolder
   strokes give the local refinement more ink to fit (geom 0.108 → 0.053,
   coverage 0.657 → 0.714). A rasterizer change can partly *compensate* a
   content change.
4. **Qualitatively, the failures are structural relationships**, not shapes:
   concentric circles come out scattered (the model has no notion of a shared
   center), parallel-line clusters collapse into overlap. Tangent / independent
   circles — closest to the training distribution — are handled fine. See
   `data/preview/technical_predictions.png`.

## Implication (and why it's fixable)

The weakness is **content-distribution coverage**, not a fundamental limit.
Because the training data is *generated*, the fix is direct: enrich the training
sampler with structured arrangements (concentric, parallel, patterns,
tangencies) so the model sees these relationships during training. This is the
clearest compounding next step — it converts an OOD failure into in-distribution
training signal. It pairs naturally with scaling primitive complexity.

---

# Generator enrichment: closing the content gap

The training set was rebuilt as 59% random shape-soup + 41% structured layouts
(`scripts/build_mixed_train.py`, the 7 technical families on training-range
seeds disjoint from every probe). The 128 model was warm-started on it for 30
epochs (`checkpoints/v1_enriched_128`). To keep the measurement honest, a NEW
held-out probe with *different* structural relationships (nested squares, radial
bursts, circle chains — `lines/datagen/heldout_layout.py`) was used: those
families never appear in training.

## Before / after (threshold 0.50 + algebraic refine)

| Split | Baseline | Model before | Model after | Δ |
|-------|----------|--------------|-------------|---|
| random (in-dist)        | 0.618 | 0.590 | 0.581 | −0.009 |
| technical               | 0.541 | 0.460 | **0.795** | **+0.335** |
| technical/cv2           | 0.538 | 0.498 | 0.761 | +0.263 |
| held-out (never trained)| 0.377 | 0.511 | **0.558** | **+0.047** |
| held-out/cv2            | 0.287 | 0.530 | 0.563 | +0.033 |

## Findings

1. **The targeted weakness became a strength.** Technical content went from
   losing to the classical baseline (0.460 vs 0.541) to beating it decisively
   (0.795 vs 0.541). On technical content type accuracy reached 0.993, coverage
   0.928, geometric error 0.023 (~5x better than before). Concentric circles —
   the headline failure — are now rendered cleanly (see
   `data/preview/enrichment_before_after.png`).
2. **Generalization improved, not just memorization.** The held-out families,
   which never appear in training, also improved (+0.047 / +0.033). Learning the
   technical families taught a general notion of structured arrangement that
   transfers to unseen structure. Because the held-out families are disjoint
   from training, this is a real generalization gain, not leakage.
3. **No meaningful regression on random content** (0.590 → 0.581). The
   structured gains did not cost in-distribution performance.

This is the synthetic-data loop working as designed: a probe revealed a gap, the
gap identified exactly what to generate, and adding it to training converted an
out-of-distribution failure into both an in-distribution strength and a
generalization gain. The next probe (a third arrangement style) keeps the loop
honest.

