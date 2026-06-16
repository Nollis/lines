# Requirements: Vector Line Extraction from Renders (Synthetic-Trained)

**Date:** 2026-06-16
**Status:** Requirements / pre-planning
**Scope class:** Deep (research-y ML system) — but v1 deliberately scoped Tiny

---

## Problem

Turn a raster image of a part/shape into **clean, CAD/illustration-grade vector
lines** — true geometric primitives (lines, arcs, circles, Béziers), not pixel
traces. Classical vectorizers (Potrace, Illustrator Image Trace) over-segment,
double-line strokes, can't separate lines from shading, and recover no real
primitives (a circle comes back as a 200-point polyline). The goal is
illustration-quality, *semantic* line work suitable for technical illustration.

## Why ML (the value proposition — and the trap to avoid)

The training data is generated from geometry we control, so every example yields
perfectly aligned ground truth (shaded image + line-art raster + exact
primitives) with **no human labeling**. Primitive-level labels are essentially
unobtainable any other way — no human reliably annotates "arc, radius 23.4mm."

**Load-bearing constraint:** geometry is a *training-data generator only*. The
model earns its existence **only when the input is an image whose source
geometry is unavailable** (a drawing, a catalog render, a screenshot, a render
whose CAD file is gone). If we have the geometry at inference, we'd extract
lines directly and skip the model. Scope must not drift into "but sometimes we
have the geometry" — that path deletes the model.

## Users / use

Inference input is **clean digital renders** (crisp, high-contrast,
machine-generated line work or rendered views) — the smallest sim-to-real gap to
our synthetic training data. Output consumed as editable vector primitives for
technical illustration.

## Goals

- Produce **true geometric primitives** (line, arc, circle, Bézier) from a
  raster image — editable and clean, not dense polylines.
- Train entirely on **synthetic, self-labeled** pixel→primitive pairs.
- Beat a classical baseline (Potrace/Image Trace + primitive fitting) on a held
  out synthetic test set — measured, not assumed.
- Close the full **pixels → primitives loop end to end** on a trivial case
  first, then scale complexity on the same rails.

## Non-goals (v1)

- Shaded-render → line-art stage (deferred; v1 renders clean line art directly).
- Hidden/dashed lines, centerlines, section hatching, dimensions, annotations.
- Dense / realistic mechanical drawings.
- Hand-drawn, scanned, or photographed inputs (different sim-to-real regime).
- Real-time inference performance.

## Architecture: pipeline shape

Conceptual full pipeline (long-term):

1. Shaded 3D render → line-art raster (NPR line extraction)
2. Line-art raster → vector primitives (the core model)

**v1 collapses stage 1**: the data generator renders clean line art directly, so
v1 is *only* stage 2 (line-art raster → primitives). Stage 1 is added later,
trained and validated independently on the same data rails.

## v1 target (the walking skeleton)

- **Input:** clean 2D line-art raster of a simple composition (**1–5
  primitives**: lines, arcs, circles).
- **Output:** the exact set of primitives with parameters.
- **Done =** the end-to-end pixels→primitives loop works on tiny 2D cases and
  beats the classical baseline on the synthetic test set.

## Success criteria

- Quantitative metric defined for "primitives match" (e.g. primitive-type
  accuracy + parameter error / Chamfer or rendered-IoU between predicted and
  ground-truth vectors). *(Exact metric chosen in planning.)*
- v1 model beats classical baseline on that metric on a held-out synthetic set.
- A small, honest **real-image probe set** (a handful of genuine clean renders
  not from our generator) to watch the sim-to-real gap — pass/fail not required
  for v1, but tracked from day one so drift is visible.

## Recommended technical direction (high level — details are planning)

- **Core model:** Set prediction, DETR-style — N primitive queries, each emits
  `{type, params}`, Hungarian-matched to ground-truth primitives. Natural fit
  for a small bounded primitive set.
- **Refinement (optional):** Differentiable vector rendering (DiffVG) as a
  snapping pass — optimize predicted primitive params to match the input for
  CAD-grade precision. Doubles as a near-zero-training baseline to validate data
  + metrics before training.
- **Scale-path (documented, not built):** Autoregressive "CAD-as-language"
  sequence generation (DeepCAD / SketchGraphs lineage) for when bounded-N and
  drawing conventions outgrow set prediction.
- **Baseline-first:** Stand up Potrace / Image Trace + primitive fitting on the
  synthetic test set before training anything — it defines the bar ML must clear
  and operationalizes the value-proposition pressure-test.

## Key risks / open questions

- **Sim-to-real gap.** Even "clean renders" from elsewhere differ from our
  generator's renders (anti-aliasing, line weight, resolution, compression).
  Mitigation: domain randomization in the generator; real-image probe set.
- **Algorithm-imitation trap.** If ground-truth lines are produced by an
  existing deterministic extractor, ensure the inference value (pixels-only
  input) is real. (See value-proposition constraint above.)
- **Primitive count / ambiguity.** Bounded-N caps v1; symmetric shapes create
  matching ambiguity; arc parameterization (center/radius/angles vs endpoints)
  affects learnability — choose carefully in planning.
- **Metric design.** "Correct primitives" is non-trivial to score (a circle and
  a 360° arc render identically; near-duplicate primitives). Needs a deliberate
  metric.
- **Generator scope creep.** The data generator is the real engine; keep its v1
  as small as the model's v1.

## Prior art to review in planning

- Vectorization / line drawing: Mastering Sketching, Virtual Sketching,
  PolyVector Flow, DiffVG / Im2Vec.
- CAD / structured prediction: DeepCAD, SketchGraphs, PolyGen, CAD-as-language.
- Classical: Potrace, Illustrator Image Trace (baseline).

## Decisions captured

- Inference input = clean digital renders.
- Pipeline = shaded → line-art → primitives; **v1 = line-art → primitives only**.
- Output = **true geometric primitives** (not polylines, not stylized strokes).
- Geometry is training-only; never assumed at inference.
- v1 = tiny 2D, 1–5 primitives; baseline-first; set-prediction model.
