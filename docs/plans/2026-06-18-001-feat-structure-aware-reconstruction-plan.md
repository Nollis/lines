---
title: "feat: Structure-aware reconstruction (honest metric + relationship-capable architecture)"
type: feat
status: active
date: 2026-06-18
origin: docs/3d-roadmap.md (Stage 1 result)
---

# feat: Structure-aware reconstruction

## Overview

Stage 1 (3D boxes) exposed two coupled gaps in the system, both now blocking
further 3D work:

1. **The metric is blind to structural error.** A 12.9-line tangle over an
   8.8-edge box scores 0.629 because render-IoU rewards rough ink coverage.
2. **The architecture cannot represent inter-primitive relationships.**
   Bounded-N (DETR-style) set prediction emits N independent primitives with no
   notion of shared corners (boxes) or shared centers (concentric circles), so
   connected drawings come out as tangles.

This plan scopes the work to fix both, in the only order that is sound:
**metric first** (you cannot evaluate an architecture change while a tangle
still scores 0.63), then the cheapest structural fix, then an architecture
change only if the cheap fix is insufficient.

## Problem Frame

Across the project the deepest failures have all been *relationships the model
cannot represent* — 2D concentric circles (shared center), then 3D box edges
(shared corners). Set prediction treats every primitive independently. The
render-IoU-weighted metric has masked this because "ink in roughly the right
place" scores well regardless of whether the primitive graph is correct (see
origin: docs/3d-roadmap.md, Stage 1 result).

## Requirements Trace

- R1. A metric under which a tangle (over-prediction, disconnected edges)
  scores poorly, and a clean 1:1 reconstruction scores well.
- R2. Recalibrate every prior result under the honest metric so historical
  numbers are comparable and not over-stated.
- R3. A cheap structural post-process that recovers as much clean structure as
  possible from existing model output, measured against R1's metric.
- R4. If R3 is insufficient, a relationship-capable architecture that models
  shared endpoints/centers, evaluated on both 2D structured content and 3D
  boxes under the honest metric.
- R5. Decisions gated by measurement, not assumption (no architecture rewrite
  before the cheap fix is shown to be insufficient).

## Scope Boundaries

- Not building Stage 2 (cylinders/ellipses) until this lands — it would inherit
  the same flaw.
- Not committing to a specific architecture up front; the prototype (Unit A2)
  decides.
- No change to the 2D primitive vocabulary here (ellipses are Stage 2).

### Deferred to Separate Tasks

- 3D Stage 2 (ellipse vocabulary): resumes after this plan, on the chosen
  architecture.
- GPU enablement: likely required for the architecture experiments (Unit A3);
  scoped separately.

## Context & Research

### Relevant Code and Patterns

- `lines/eval/metrics.py` — current composite metric (render-IoU dominant);
  the per-class machinery and Hungarian matcher are reused by the new metric.
- `lines/train/predictor.py` — already does per-primitive algebraic/diffvg
  refinement; the structural post-process (Unit A1) extends this stage.
- `lines/models/set_predictor.py`, `losses.py`, `matcher.py` — the current
  set-prediction model the architecture work would replace or augment.
- `lines/datagen/box_scene.py`, `technical_layout.py` — structured content the
  honest metric and new architecture must handle.

### Institutional Learnings

- The compound loop (probe → fold into training → re-measure on held-out) is
  validated and reused; a held-out structural probe gates each architecture
  step.
- Render-IoU was deliberately chosen early to tolerate representational
  equivalence; Stage 1 shows it is too forgiving for connected structure.

### External References (confirm during implementation)

- SketchGraphs, DeepCAD, PolyGen — primitive-graph and CAD-as-language
  architectures (the relationship-capable options).

## Key Technical Decisions

- **Metric before architecture.** The honest metric is the gate; nothing
  downstream is evaluable without it.
- **Strict primitive-F1 as the new headline metric.** A predicted primitive is
  a *true match* only if Hungarian-assigned to a GT primitive with normalized
  geometric error < τ AND correct type. Report precision / recall / F1.
  Over-prediction tanks precision; missing edges tank recall. Render-IoU and
  mean geometric error are kept as *diagnostics*, demoted from the score.
- **Cheapest structural fix first (post-process), architecture only if needed.**
  Junction-merge + dedup on existing output may recover much of the tangle with
  no retraining — try it before any rewrite (YAGNI hedge against a large,
  premature architecture change).
- **Architecture choice is prototyped, not assumed.** Two real options
  (vertex-graph vs autoregressive CAD-as-language) are compared on a small
  prototype before committing.

## Open Questions

### Resolved During Planning

- Order of work? → metric, then post-process, then (gated) architecture.
- New headline metric? → strict primitive-F1 (type + geometric-error threshold).

### Deferred to Implementation

- Match threshold τ for "true match" — calibrate so a perfect reconstruction
  scores ~1.0 and the Stage-1 tangle scores low; choose against real outputs.
- Whether a dedicated junction/corner-graph metric term is needed beyond F1 —
  decide after seeing F1 on the box tangle.
- Vertex-graph vs autoregressive — decided by the Unit A2 prototype.
- Whether curves (circles/arcs) need a hybrid representation under a
  vertex-graph architecture.

## Implementation Units

- [ ] **Unit M1: Strict primitive-F1 metric**

**Goal:** A structure-honest headline metric where a tangle scores poorly.

**Requirements:** R1

**Dependencies:** None

**Files:**
- Modify: `lines/eval/metrics.py` (add strict matching + precision/recall/F1),
  `lines/eval/harness.py` (aggregate the new fields)
- Test: `tests/test_metrics.py` (extend)

**Approach:**
- Reuse the Hungarian assignment; a match counts as *true* only if normalized
  geometric error < τ and predicted type == GT type.
- precision = true / n_pred, recall = true / n_gt, F1 = harmonic mean.
- Keep render-IoU and mean geom error in the report as diagnostics; make F1 the
  headline `score` (or add `f1` alongside and shift downstream reporting to it).

**Test scenarios:**
- Happy path: identical sets → precision = recall = F1 = 1.0.
- Edge: over-prediction (2× spurious primitives) → precision ≈ 0.5, F1 drops.
- Edge: half the GT primitives missing → recall ≈ 0.5.
- Edge: correct positions but wrong type → not a true match (F1 penalized).
- Edge: a "tangle" fixture (many overlapping near-duplicate lines over a few GT
  lines) scores low F1 despite high render-IoU.

**Verification:** The Stage-1 box tangle scores low F1; a hand-built clean
reconstruction of the same box scores near 1.0.

- [ ] **Unit M2: Recalibrate all prior results under F1**

**Goal:** Re-score historical checkpoints so numbers are comparable and honest.

**Requirements:** R2

**Dependencies:** Unit M1

**Files:**
- Modify: `scripts/eval_generalization.py`, `scripts/update_results.py` (emit F1)
- Modify: `docs/results.md`, `docs/best-config.md` (add F1 columns / caveats)

**Approach:**
- Re-run the existing splits + checkpoints, reporting F1 next to the old score.
- Expect large drops (especially boxes); document that render-IoU overstated.

**Test scenarios:** Test expectation: none -- reporting/scripts; covered by M1.

**Verification:** `docs/results.md` shows F1 alongside the legacy score for every
checkpoint; the box model's F1 is far below its 0.629 render-score.

- [ ] **Unit A1: Structural post-process (junction merge + dedup)**

**Goal:** Recover clean structure from existing set-prediction output without
retraining; quantify how far cheap post-processing gets.

**Requirements:** R3, R5

**Dependencies:** Unit M1 (to measure the gain honestly)

**Files:**
- Create: `lines/refine/structure.py` (merge near-coincident endpoints to shared
  corners; remove/merge near-duplicate and collinear-overlapping primitives)
- Modify: `lines/train/predictor.py` (optional post-process stage)
- Test: `tests/test_structure_postprocess.py`

**Approach:**
- Cluster predicted endpoints; snap clustered endpoints to a shared corner.
- Drop primitives that are near-duplicates or sub-segments of another.
- Apply to the Stage-1 box model output; measure F1 before/after.

**Test scenarios:**
- Happy path: two lines with near-coincident endpoints → endpoints snapped to a
  single shared corner.
- Edge: two near-duplicate overlapping lines → merged to one.
- Edge: a clean input is left unchanged (idempotent).
- Integration: box-model tangle → post-process raises F1 measurably.

**Verification:** Post-process raises the box model's F1 (decision input for
Gate 2); clean inputs are unchanged.

- [ ] **Unit A2: Architecture prototype + decision (gated)**

**Goal:** If A1 is insufficient, compare relationship-capable architectures on a
small prototype and choose one.

**Requirements:** R4, R5

**Dependencies:** Unit A1 (only proceed if A1 leaves a large gap)

**Files:**
- Create: `lines/models/vertex_graph.py` *or* `lines/models/autoregressive.py`
  (prototype of whichever is built first), plus a small training entry point
- Test: prototype-level shape/overfit tests

**Approach (directional — pick via prototype, not pre-committed):**
- *Option (a) vertex-graph:* predict corner points as a set, then classify edge
  adjacency between detected corners. Natural for boxes/polygons; needs a
  hybrid path for curves.
- *Option (b) autoregressive CAD-as-language:* generate primitives as a
  sequence where each conditions on prior ones (shared endpoints become
  references). Models relationships + unbounded counts; needs canonical
  ordering and teacher forcing.
- Overfit a tiny box batch with each prototype; compare clean-reconstruction F1.

**Test scenarios:**
- Happy path (sanity): prototype overfits a tiny box batch to high F1 (proving
  it *can* represent the junction structure set prediction could not).

**Verification:** A documented comparison (F1, complexity, curve-handling) and a
chosen architecture.

- [ ] **Unit A3: Implement chosen architecture + compound-loop evaluation**

**Goal:** Build the chosen architecture, train it, and measure on 2D structured
content and 3D boxes under the honest metric with a held-out structural probe.

**Requirements:** R4

**Dependencies:** Unit A2

**Files:**
- Create/replace model + loss + training wiring for the chosen architecture
- Modify: training/eval scripts as needed
- Test: model/loss tests + an overfit sanity test

**Approach:**
- Train on the existing structured + box content; evaluate F1 on held-out
  structural families (the compound-loop discipline) and on boxes.
- Compare against set-prediction + A1 post-process baselines.

**Test scenarios:**
- Happy path: overfit-tiny-batch sanity (architecture wiring correct).
- Integration: clean box wireframes recovered (F1 markedly above set-prediction
  + post-process); 2D structured content not regressed.

**Verification:** On held-out boxes and structured 2D, the new architecture
beats set-prediction + A1 on F1, with visibly clean (non-tangled)
reconstructions.

## Decision Gates

| Gate | After | Decision |
|------|-------|----------|
| 1 | M1 | The honest metric must rank the tangle low and a clean reconstruction high. If not, fix the metric before anything else. |
| 2 | A1 | **DECIDED → proceed to A2.** Post-process moved box F1 only 0.262 → 0.282 (dedup fixed the count 12.9 → 8.3 and precision 0.23 → 0.30, but recall *fell* 0.315 → 0.273 — merging can't create accurate edges the model never predicted). 0.28 is far from the ~0.9 target, so the architecture rewrite is justified, not premature. |
| 3 | A2 | **DECIDED → autoregressive.** Bake-off (`scripts/a2_overfit_bench.py`, 64 samples × 600 steps each, ~7 min CPU): autoregressive overfit *boxes* to mean F1 0.998 (63/64 exact) and *concentric circles* to mean F1 0.995 (63/64 exact). Both relationship-failure cases the project hit (shared corners AND shared centers) are now representable in one architecture. Vertex-graph not built -- it structurally cannot represent the concentric case, so the bake-off ruled it out without needing implementation. |

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Architecture rewrite is large and may not be necessary | A1 (cheap post-process) is the YAGNI hedge; Gate 2 can stop here |
| Autoregressive adds ordering/teacher-forcing/inference-speed cost | Prototype (A2) surfaces this before committing |
| Vertex-graph handles curves poorly | Evaluate curve handling in A2; consider a hybrid before A3 |
| New metric makes historical numbers look worse | That is the point (R2); document the recalibration clearly |
| Architecture training needs more compute than CPU allows | Flag GPU as a prerequisite for A3; keep A2 prototypes tiny |

## Sources & References

- **Origin:** [docs/3d-roadmap.md](docs/3d-roadmap.md) (Stage 1 result)
- Related: [docs/sim-to-real.md](docs/sim-to-real.md) (the relationship-coverage
  lesson from 2D), [docs/best-config.md](docs/best-config.md)
- External (to confirm): SketchGraphs, DeepCAD, PolyGen.
