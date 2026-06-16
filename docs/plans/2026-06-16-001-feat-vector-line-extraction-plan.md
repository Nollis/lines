---
title: "feat: Synthetic-trained vector primitive extraction (v1 walking skeleton)"
type: feat
status: active
date: 2026-06-16
origin: docs/brainstorms/2026-06-16-vector-line-extraction-requirements.md
---

# feat: Synthetic-trained vector primitive extraction (v1 walking skeleton)

## Overview

Build the end-to-end **pixels → geometric primitives** loop on the smallest
useful case: a clean 2D line-art raster of 1–5 primitives (lines, arcs, circles)
goes in; the exact set of geometric primitives comes out. Everything is trained
on synthetic, self-labeled data generated from geometry we control. v1
deliberately drops the shaded-render→line-art stage and renders clean line art
directly, so the only learned mapping is line-art raster → primitives.

The plan front-loads two things that protect the project from quiet failure: a
**metric/eval harness defined before any model exists**, and a **classical
baseline** (skeletonize + primitive fitting) that the learned model must beat on
a held-out synthetic set. Later milestones scale the same rails toward synthetic
3D parts and the full shaded-render pipeline.

## Problem Frame

Classical vectorizers over-segment, double-line strokes, can't separate lines
from shading, and return no real primitives (a circle becomes a 200-point
polyline). The goal is illustration/CAD-grade **true primitives** from a raster
image whose source geometry is unavailable at inference. Synthetic generation is
the only practical way to get primitive-level ground truth — no human reliably
labels "arc, radius 23.4mm" (see origin:
docs/brainstorms/2026-06-16-vector-line-extraction-requirements.md).

## Requirements Trace

- R1. Output **true geometric primitives** (line, arc, circle, Bézier) from a raster image — not dense polylines. *(origin: Output fidelity)*
- R2. Train entirely on **synthetic, self-labeled** pixel→primitive pairs. *(origin: Why ML)*
- R3. **Beat a classical baseline** (skeletonize + primitive fitting) on a held-out synthetic set — measured, not assumed. *(origin: Goals, value-proposition pressure-test)*
- R4. Close the full **pixels→primitives loop end to end** on tiny 2D (1–5 primitives) first. *(origin: v1 target)*
- R5. Track the **sim-to-real gap** via a small real-image probe set from day one. *(origin: Success criteria)*
- R6. **Geometry is training-only** — never read at inference; the model consumes pixels alone. *(origin: load-bearing constraint)*

## Scope Boundaries

- No shaded-render → line-art stage in v1 (render clean line art directly).
- No hidden/dashed lines, centerlines, section hatching, dimensions, or annotations.
- No dense/realistic mechanical drawings.
- No hand-drawn, scanned, or photographed inputs.
- No real-time inference target; throughput is irrelevant for v1.
- No Bézier *generation* in the v1 data sampler beyond a minimal optional class — primary primitives are line/arc/circle (Bézier kept in the schema and metric so the model and eval are forward-compatible).

### Deferred to Separate Tasks

- Synthetic 3D part generator (CSG + Blender line-art): Phase 2, Unit 8.
- Shaded-render → line-art stage (full pipeline): Phase 2, Unit 9.
- Autoregressive "CAD-as-language" model for unbounded primitive counts: future iteration (documented scale-path, not built).

## Context & Research

### Relevant Code and Patterns

- Greenfield repo — no existing code to mirror. Establish conventions in Unit 1.

### Institutional Learnings

- None on disk (`docs/solutions/` absent).

### External References

- Set prediction: DETR (bipartite/Hungarian-matched set prediction) — architectural template for Unit 5.
- Differentiable vector rendering: `diffvg` — primitive-parameter refinement against the input raster (Unit 6).
- CAD/structured-prediction prior art (documented scale-path, not v1): DeepCAD, SketchGraphs, PolyGen.
- Classical baseline building blocks: morphological skeletonization, Hough/circle fitting, least-squares arc/line fitting.

> External-docs research (exact `diffvg` API, current DETR variants, line-art rendering libs) deferred — see "Open Questions → Deferred" and the deepening option at handoff.

## Key Technical Decisions

- **Python + PyTorch.** Standard stack for DETR-style set prediction and the `diffvg` differentiable renderer. *(Rationale: ecosystem fit; `diffvg` targets PyTorch.)*
- **Pure-Python 2D generator for v1; no Blender.** Sample primitives analytically and rasterize with a 2D vector renderer (cairo/skia-python). Blender enters only at the 3D milestone (Unit 8). *(Rationale: keep the v1 generator trivial and fully controlled.)*
- **Metric defined before models.** The eval harness (Unit 3) lands before baseline and model so all three are scored identically. *(Rationale: prevents moving goalposts; operationalizes R3.)*
- **Dual-component match metric.** (a) Hungarian-matched primitive **type accuracy + normalized parameter error**, and (b) **render-based** agreement (rasterize predicted vs GT primitives, compare via IoU/Chamfer). The render-based component resolves representational ambiguity (a circle and a 360° arc render identically). *(Rationale: parameter-only metrics misjudge equivalent representations.)*
- **Canonical primitive schema with normalized parameters.** One serializable representation shared by generator, baseline, model, metric, and refiner. Parameters normalized to canvas coordinates. *(Rationale: single source of truth prevents drift across components.)*
- **Arcs parameterized as `endpoints + signed bulge`** (`bulge = tan(θ/4)`, DXF convention), not center+radius+angles. *(Rationale: endpoints are directly localizable in the image while a center is often off-canvas and perceptually decoupled; bulge is a single signed scalar with no angle-wraparound discontinuity; `bulge = 0` degenerates to a straight line, so near-straight arcs regress smoothly. The bulge→∞ failure near a full circle is avoided because full curvature is the separate `Circle` primitive and the generator caps arc sweep below that regime.)*
- **Bounded-N set prediction for v1.** Fixed query count N (≥ max primitives, e.g. N=16 for a 1–5 cap) with a "none" class. *(Rationale: matches the tiny bounded target; the unbounded case is the documented scale-path.)*
- **DiffVG refinement is optional and additive.** Model output is usable on its own; refinement is a snap-to-pixels pass evaluated as an ablation. *(Rationale: de-risks v1 — the loop closes without it.)*

## Open Questions

### Resolved During Planning

- Tech stack? → Python + PyTorch + diffvg.
- 2D generator tooling? → analytic sampling + cairo/skia rasterization; Blender deferred to 3D.
- Where does the metric live in sequencing? → before baseline and model (Unit 3).
- How to score representationally-equivalent primitives? → render-based metric component.
- v1 primitive set? → line, arc, circle primary; Bézier present in schema/metric, minimal in sampler.
- Arc parameterization? → `endpoints + signed bulge` (`tan(θ/4)`); full circles use the `Circle` primitive; generator caps arc sweep below the bulge blow-up regime. *(See Key Technical Decisions for rationale.)*

### Deferred to Implementation
- Image encoder backbone (small CNN vs ViT) and N/decoder depth — tune in Unit 5; start small.
- Rasterization library specifics (`pycairo` vs `skia-python`) and anti-aliasing settings — settle in Unit 2.
- Exact `diffvg` primitive-parameter wiring and optimizer schedule — Unit 6.
- Tolerance thresholds for the baseline fitter and for "match" in the metric — calibrate against generated data in Units 2–4.

## Output Structure

    lines/
      __init__.py
      primitives.py            # canonical primitive schema + (de)serialization + normalization
      datagen/
        sampler2d.py           # sample 1–5 primitives within canvas constraints
        render.py              # rasterize primitives -> clean line-art image
        randomize.py           # domain-randomization knobs (line weight, AA, resolution)
        dataset.py             # write image+manifest pairs; torch Dataset loader
      eval/
        metrics.py             # Hungarian match + param error + render-based IoU/Chamfer
        harness.py             # run a predictor over a split, aggregate + report
      baselines/
        classical.py           # skeletonize + primitive fitting -> primitives
      models/
        set_predictor.py       # encoder + transformer decoder, N primitive queries
        matcher.py             # Hungarian matching for loss
        losses.py              # type CE + parameter regression loss
      train/
        config.py
        train.py               # training loop, checkpointing, eval hook
      refine/
        diffvg_refine.py       # optional differentiable-render snap pass
    tests/
      test_primitives.py
      test_sampler2d.py
      test_render.py
      test_metrics.py
      test_classical.py
      test_set_predictor.py
      test_diffvg_refine.py
    configs/
      v1_tiny2d.yaml
    data/                      # generated datasets (gitignored)
    docs/

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
                 (training only — geometry never seen at inference, R6)
 sampler2d ──► primitives (ground truth set) ──► render.py ──► line-art raster
     │                                                              │
     └────────────► manifest (image_path, primitive list) ◄─────────┘

 inference / eval path:
   line-art raster ──► [ classical baseline ]      ─┐
                       [ set-prediction model ]     ─┼─► predicted primitive set
                       [ model + diffvg refine ]    ─┘            │
                                                                  ▼
   ground-truth set ───────────────────► eval/metrics.py ──► score
       (Hungarian match: type acc + param error)   +   (render both, IoU/Chamfer)
```

The metric consumes only `(predicted_primitives, ground_truth_primitives)`, so
the baseline, the model, and the refined model are scored through the exact same
function — the comparison that answers R3.

## Implementation Units

### Phase 1 — v1 walking skeleton (tiny 2D, line-art → primitives)

- [x] **Unit 1: Primitive schema + project scaffold**

**Goal:** Establish the canonical primitive representation and a minimal Python package every later unit imports.

**Requirements:** R1, R6

**Dependencies:** None

**Files:**
- Create: `lines/__init__.py`, `lines/primitives.py`, `pyproject.toml`, `configs/v1_tiny2d.yaml`, `.gitignore`
- Test: `tests/test_primitives.py`

**Approach:**
- Define `Line` (endpoints), `Arc` (endpoints + signed bulge), `Circle` (center + radius), `Bezier` (control points), each with a `type` tag, plus a `PrimitiveSet` container.
- Provide an arc↔(center, radius, start/end angle) conversion helper for fitting/rendering, while keeping endpoints+bulge as the stored/learned form.
- Provide normalization to/from canvas coordinates and JSON (de)serialization round-trip.
- No model/render logic here — pure data + math.

**Patterns to follow:**
- None (greenfield); set conventions (typing, module layout) other units mirror.

**Test scenarios:**
- Happy path: build each primitive type, serialize to JSON and back → equal object.
- Happy path: normalize then denormalize a primitive against a canvas size → original params within float tolerance.
- Edge case: degenerate inputs (zero-length line, zero-radius circle, zero-sweep arc) → flagged invalid by a validator, not silently accepted.
- Happy path: arc round-trips endpoints+bulge → (center, radius, angles) → endpoints+bulge within tolerance, including a negative-bulge (opposite sweep direction) case.
- Edge case: `bulge = 0` arc renders/derives as the straight segment between its endpoints (line/arc unification holds).
- Edge case: a near-full-circle shape is representable as a `Circle`, and an arc whose sweep exceeds the generator cap is flagged rather than stored with an exploding bulge.

**Verification:** All primitive types round-trip through JSON and normalization; validator rejects degenerate cases.

- [x] **Unit 2: Synthetic 2D data generator**

**Goal:** Generate clean line-art raster + exact primitive ground truth pairs, reproducibly, with domain-randomization knobs.

**Requirements:** R2, R4, R6

**Dependencies:** Unit 1

**Files:**
- Create: `lines/datagen/sampler2d.py`, `lines/datagen/render.py`, `lines/datagen/randomize.py`, `lines/datagen/dataset.py`
- Test: `tests/test_sampler2d.py`, `tests/test_render.py`

**Approach:**
- `sampler2d` draws 1–5 primitives within canvas bounds, seeded for determinism. Arc sweep is capped below the bulge blow-up regime (e.g. ≤ ~270°); shapes beyond that are emitted as a `Circle` or split, so stored bulges stay bounded.
- `render` rasterizes the primitive set to a clean black-on-white line-art image (cairo/skia), with controllable line weight, anti-aliasing, and resolution from `randomize`.
- `dataset` writes `(image, manifest)` pairs and exposes a torch `Dataset`. The manifest stores the exact primitives — this is the free ground truth.

**Execution note:** Build the sampler test-first so the determinism/bounds contract is pinned before rendering is wired in.

**Patterns to follow:**
- `lines/primitives.py` schema as the manifest format.

**Test scenarios:**
- Happy path: generate a single circle → manifest has exactly one circle whose center/radius match the requested params; rendered image is non-empty and within canvas.
- Happy path: same seed → byte-identical image and identical manifest (reproducibility).
- Edge case: max count (5) with overlapping primitives → all 5 recorded; render does not crash.
- Edge case: sampler never emits primitives outside canvas bounds over many seeds.
- Edge case: degenerate samples are rejected/resampled (ties to Unit 1 validator).
- Integration: a written pair reloads through the torch `Dataset` with image tensor shape and manifest matching what was written.

**Verification:** A small generated dataset (image+manifest pairs) loads through the `Dataset`; manifests reconstruct the drawn primitives.

- [x] **Unit 3: Eval + metric harness**

**Goal:** Score any predictor's primitive output against ground truth, identically for baseline and model.

**Requirements:** R3, R5

**Dependencies:** Unit 1 (schema), Unit 2 (data + render for the render-based component)

**Files:**
- Create: `lines/eval/metrics.py`, `lines/eval/harness.py`
- Test: `tests/test_metrics.py`

**Approach:**
- `metrics`: (a) Hungarian-match predicted↔GT primitives, scoring type accuracy + normalized parameter error and penalizing unmatched (false pos/neg); (b) render both sets and compute IoU/Chamfer to resolve representational equivalence.
- `harness`: run a predictor over a split, aggregate per-primitive and per-image scores, emit a report; supports the synthetic test split and the real-image probe set (R5).

**Patterns to follow:**
- `lines/datagen/render.py` for the render-based metric component.

**Test scenarios:**
- Happy path: identical predicted and GT sets → perfect score (all matched, zero param error, IoU = 1).
- Edge case: predicted 360° arc vs GT circle (identical rendering) → render-based component scores them equivalent even though param matching differs.
- Edge case: one extra predicted primitive (false positive) → score penalized via unmatched count.
- Edge case: one missing primitive (false negative) → score penalized.
- Happy path: predicted circle radius off by a known delta → parameter-error component increases monotonically with delta.
- Edge case: empty prediction vs non-empty GT → worst-case score, no crash.

**Verification:** Metric returns perfect score on identical sets, equivalent score on the circle/360-arc case, and degrades monotonically as predictions worsen.

- [x] **Unit 4: Classical baseline**

**Goal:** A non-learned skeletonize + primitive-fitting predictor that defines the bar the model must beat.

**Requirements:** R3

**Dependencies:** Unit 1, Unit 2, Unit 3

**Files:**
- Create: `lines/baselines/classical.py`
- Test: `tests/test_classical.py`

**Approach:**
- Skeletonize the raster, segment strokes, fit line/arc/circle primitives (least-squares + circle/Hough fitting), emit a `PrimitiveSet`.
- Run through the Unit 3 harness on the synthetic test split; record the baseline score as the reference number.

**Patterns to follow:**
- Predictor interface implicit in `lines/eval/harness.py` (takes an image, returns a `PrimitiveSet`).

**Test scenarios:**
- Happy path: clean single-circle raster → fitter returns one circle (not a polyline) within tolerance.
- Happy path: two crossing lines → returns two line primitives; junction does not collapse them into one.
- Edge case: single arc (partial circle) → returns an arc, not a full circle.
- Edge case: empty image → returns empty set, no crash.
- Integration: baseline runs through the harness end to end and produces a recorded reference score.

**Verification:** Baseline produces a stable, recorded score on the synthetic test split that later model runs compare against.

- [ ] **Unit 5: Set-prediction model + training loop**

**Goal:** Train a DETR-style model to predict the primitive set from a raster, and beat the baseline on the held-out synthetic split.

**Requirements:** R1, R2, R3, R4, R6

**Dependencies:** Unit 1–4

**Files:**
- Create: `lines/models/set_predictor.py`, `lines/models/matcher.py`, `lines/models/losses.py`, `lines/train/config.py`, `lines/train/train.py`
- Test: `tests/test_set_predictor.py`

**Approach:**
- Image encoder (start small — compact CNN) → transformer decoder with N primitive queries; each query emits a type distribution (line/arc/circle/Bézier/none) and parameter vector.
- Hungarian matcher assigns queries to GT primitives; loss = type cross-entropy + parameter regression on matched pairs.
- Training loop consumes the Unit 2 `Dataset`, checkpoints, and runs the Unit 3 harness as an eval hook each epoch.

**Execution note:** Land a deliberate overfit-a-tiny-batch test first — it is the fastest proof the matcher + loss + heads are wired correctly before any real training.

**Patterns to follow:**
- DETR set-prediction structure (queries + bipartite matching).
- `lines/eval/harness.py` for the eval hook.

**Test scenarios:**
- Happy path (sanity): overfit a tiny fixed batch (e.g. 8 samples) → training loss approaches ~0 and predictions match those samples.
- Unit: Hungarian matcher returns the known-optimal assignment on a hand-constructed predicted/GT pair.
- Happy path: type head classifies line/arc/circle/none correctly on simple single-primitive inputs.
- Edge case: image with fewer primitives than N → surplus queries predict "none".
- Integration (success gate): trained model scored through the Unit 3 harness exceeds the Unit 4 baseline on the held-out synthetic split.

**Verification:** Model overfits a tiny batch; on the held-out split it beats the recorded classical baseline score (R3 satisfied).

- [ ] **Unit 6: Optional DiffVG refinement (ablation)**

**Goal:** Snap predicted primitive parameters to the pixels via differentiable vector rendering, and measure whether it helps.

**Requirements:** R1, R3

**Dependencies:** Unit 5

**Files:**
- Create: `lines/refine/diffvg_refine.py`
- Test: `tests/test_diffvg_refine.py`

**Approach:**
- Initialize from the model's predicted primitives; render with `diffvg`; optimize parameters to minimize image difference vs the input raster.
- Compare model-only vs model+refine through the Unit 3 harness as an ablation.

**Patterns to follow:**
- `lines/eval/harness.py` for scoring; `lines/primitives.py` for parameter handling.

**Test scenarios:**
- Happy path: primitives perturbed slightly from ground truth → refinement reduces render IoU/parameter error toward the target.
- Edge case: already-correct init → refinement is stable (no divergence / minimal change).
- Edge case: degenerate primitive during optimization → handled without NaN/crash.
- Integration (ablation): model+refine score ≥ model-only score on the held-out split.

**Verification:** Refinement improves or matches model-only on the metric; never degrades a good initialization into a worse one.

### Phase 2 — Scale toward synthetic 3D parts (later milestones)

- [ ] **Unit 7: Domain randomization + real-image probe set**

**Goal:** Widen the generator's appearance distribution and stand up the sim-to-real tracking set.

**Requirements:** R5

**Dependencies:** Unit 2, Unit 3

**Files:**
- Modify: `lines/datagen/randomize.py`, `lines/eval/harness.py`
- Create: `data/probe/` (a handful of genuine clean renders not from our generator, with hand-specified primitive targets)
- Test: extend `tests/test_render.py`

**Approach:**
- Expand randomization (line weight, AA, resolution, slight noise/contrast) so the model doesn't overfit one render style.
- Curate a tiny real-image probe set and run it through the harness each eval to watch the gap (pass/fail not required for v1, but tracked).

**Test scenarios:**
- Happy path: randomization parameters stay within configured ranges over many seeds.
- Integration: probe set runs through the harness and reports a (lower) score that is tracked over time.

**Verification:** Probe-set score is reported alongside synthetic-split score on every eval run.

- [ ] **Unit 8: Synthetic 3D part generator**

**Goal:** Generate CSG-style 3D parts and emit projected 2D primitives as ground truth.

**Requirements:** R1, R2, R6

**Dependencies:** Unit 1, Unit 2 (manifest format), Unit 3

**Files:**
- Create: `lines/datagen/parts3d.py`, `lines/datagen/blender_lineart.py`
- Test: `tests/test_parts3d.py`

**Approach:**
- Compose primitives in 3D (boxes + cylinders + holes), render line art via Blender Freestyle / Line Art, and extract the projected 2D primitive ground truth (silhouettes, feature edges; circles project to ellipses/arcs).
- Reuse the Unit 1 schema and Unit 3 metric unchanged; only the data source is new.

**Test scenarios:**
- Happy path: a single cylinder → projected primitives include the expected silhouette lines and elliptical/arc edges in the manifest.
- Edge case: hole/occlusion → occluded edges handled per the chosen convention; manifest consistent with the rendered line art.
- Integration: 3D-generated pairs load through the existing `Dataset` and score through the existing harness.

**Verification:** 3D pairs flow through the unchanged schema/dataset/metric; manifests align with rendered line art.

- [ ] **Unit 9: Shaded-render → line-art stage**

**Goal:** Complete the full pipeline by learning (or rendering) line art from shaded 3D renders.

**Requirements:** R1

**Dependencies:** Unit 8

**Files:**
- Create: `lines/models/shaded_to_lineart.py`, `lines/datagen/shaded_render.py`
- Test: `tests/test_shaded_to_lineart.py`

**Approach:**
- Render shaded views alongside line art (Blender), then train an image-to-line model so the v1 vectorizer can consume shaded renders via a two-stage pipeline.
- Validate stage 1 independently before chaining to Unit 5's vectorizer.

**Test scenarios:**
- Happy path: shaded render of a simple part → predicted line-art raster close to the rendered ground-truth line art.
- Integration: stage1 output fed into the Unit 5 vectorizer produces primitives scored through the Unit 3 harness (full-pipeline check).

**Verification:** Two-stage pipeline (shaded → line art → primitives) runs end to end and is scored through the existing harness.

## System-Wide Impact

- **Interaction graph:** The `PrimitiveSet` schema (Unit 1) and the metric/harness (Unit 3) are the two hubs every other unit depends on; changes there ripple everywhere. Keep both stable once Phase 1 lands.
- **Predictor contract:** Baseline (Unit 4), model (Unit 5), and refiner (Unit 6) all implement the same "image → PrimitiveSet" shape so the harness treats them interchangeably.
- **State lifecycle risks:** Generated datasets are large and reproducible-by-seed; keep `data/` gitignored and regenerate from seeds rather than committing artifacts.
- **Unchanged invariants:** The schema and metric defined in Phase 1 must carry into Phase 2 unchanged — the whole point of Units 8–9 is that only the *data source* changes, not how primitives are represented or scored.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| **Algorithm-imitation trap** — model just re-learns the renderer's edge algorithm with no inference value | Hold R6 firm (geometry training-only); judge success on the real-image probe set (Unit 7), not just synthetic recall |
| **Sim-to-real gap** — clean renders elsewhere differ from our generator | Domain randomization (Unit 7); probe set tracked from the first eval |
| **Metric mis-scores equivalent primitives** (circle vs 360° arc) | Render-based metric component (Unit 3) explicitly tested for this case |
| **Bounded-N caps generalization** as drawings densify | Accepted for v1; documented scale-path is the autoregressive model |
| **Arc parameterization hurts training stability** | Resolved: `endpoints + signed bulge` keeps params image-localizable, drops angle-wraparound, and degenerates to a line at bulge 0; generator caps sweep so bulge stays bounded |
| **`diffvg` build/integration friction** | Refinement is optional and additive — the v1 loop closes without it (Unit 6 can slip) |
| **Baseline too weak/too strong to be meaningful** | Calibrate fitter tolerances against generated data; record the baseline as a fixed reference before model work |

## Sources & References

- **Origin document:** [2026-06-16-vector-line-extraction-requirements.md](docs/brainstorms/2026-06-16-vector-line-extraction-requirements.md)
- External (to confirm during implementation): DETR (set prediction), `diffvg` (differentiable vector rendering), DeepCAD / SketchGraphs (scale-path), Blender Freestyle / Line Art (3D milestone).
