---
title: "feat: Stage 2 — ellipse primitive + cylinder reconstruction"
type: feat
status: active
date: 2026-06-22
origin: docs/3d-roadmap.md (Stage 2), docs/a2-prototype-scope.md (autoregressive bake-off)
---

# feat: Stage 2 — ellipse primitive + cylinder reconstruction

## Overview

Stage 1 shipped boxes with the autoregressive architecture (F1 = 0.980,
exact-match = 0.932 on 400 held-out boxes). The natural next geometric content
is **cylinders**, which project to **2 silhouette lines + 2 elliptical rims** —
introducing the first new primitive type the schema hasn't carried since v1.

This plan covers everything that touches the addition of an ``Ellipse``
primitive end-to-end: schema, encoding, tokenizer, renderer, differentiable
renderer, metric, and a new 3D cylinder scene generator. The autoregressive
bake-off (`docs/a2-prototype-scope.md`) already overfit *concentric circles* to
F1 0.995, which is the closest existing relative to "shared-center curved
geometry" — so we have prior evidence the architecture can represent ellipses
once the vocabulary admits them.

## Why this is the right next step

- *Architecture is proven for boxes.* No retraining recipe risk; the question is
  vocabulary, not capacity.
- *Cylinders are the smallest content extension that forces ellipses.* They are
  also the most common technical-drawing feature after rectangular parts.
- *The roadmap explicitly flagged Stage 2 as the next content jump* and tagged
  it the largest single investment (it touches schema, model, *and* metric in
  one go). Every later stage (CSG, hidden lines, realistic parts) inherits
  whatever ellipse representation we land here, so investing carefully is worth
  it.

## Requirements Trace

- E1. An `Ellipse(center, semi_major, semi_minor, rotation)` primitive in the
  canonical schema, with serialization, normalization, validation, and
  approx-equality the same way the other primitives have.
- E2. The tokenizer can serialize/parse an ellipse; sequences with ellipses
  round-trip exactly, including canonical ordering.
- E3. The training renderer (`render_primitives`) draws ellipses correctly,
  including rotated axes.
- E4. `flatten_primitive` produces a faithful polyline for ellipses so Chamfer
  matching in the metric and the diffvg refiner work unchanged.
- E5. The differentiable renderer (`soft_render` / `diffvg_refine`) supports
  ellipses so any future training-time render loss or refinement covers them.
- E6. A 3D cylinder scene generator that produces (image, primitive set) pairs
  analytically — same compute-the-projection-don't-extract-from-pixels insight
  as box_scene.
- E7. A cylinder data split (train + held-out) and an empirical result on it
  under the strict primitive-F1 metric.

## Scope Boundaries

- One new primitive: `Ellipse`. *Not* extending to general conics, B-splines,
  or polygonal regions.
- Cylinders only — no cones (frustums), spheres, or CSG.
- Hidden-line removal stays at the box level (back-face culling for convex
  solids). General hidden-line removal is Stage 3.
- The Stage 1 box checkpoint is *not* warm-started into Stage 2 — vocabulary
  changes mean the token embedding and head are new shapes. Mixing boxes +
  cylinders in training is the natural curriculum extension; it falls out for
  free if both generators feed the existing training entry point.

### Deferred to Separate Tasks

- Cone / frustum primitives: future stage.
- Sphere silhouettes (visible outline = a circle from any orientation) —
  technically reducible to `Circle`, but worth its own thinking.

## Context & Research

### Relevant Code and Patterns

- `lines/primitives.py` — the existing schema. `Arc` (endpoints + signed bulge)
  is the closest analogue; ellipses are stored differently because they have
  no "endpoints" axis the way an arc does.
- `lines/models/encoding.py` — `N_TYPES = 4`, `N_PARAMS = 5`. Adding ellipse
  means `N_TYPES = 5`. Crucially, **an ellipse already fits in 5 params:**
  `[cx, cy, semi_major, semi_minor, rotation_rad]` — the param head doesn't
  need to grow.
- `lines/models/seq_tokenizer.py` — currently 134 tokens (3 special + 3 type
  + 64 coord + 64 bulge). Add `ELLIPSE` type token + an angle quantization
  block, vocabulary grows to ~200.
- `lines/datagen/projection.py` — projection core already there; cylinders need
  a new mesh type *and* analytic rim projection (a circular rim on an oblique
  plane projects to an ellipse with known parameters).
- `lines/datagen/box_scene.py` — the box scene generator is the template
  for `cylinder_scene.py`.
- `lines/models/autoregressive.py` — model is shape-agnostic; only the token
  embedding grows when vocab grows. No architectural changes.
- `lines/refine/diffvg_refine.py` — currently handles line / arc / circle in
  `_polyline`. Add an ellipse polyline branch (sample around the ellipse with
  the rotation applied).
- `lines/eval/metrics.py` — Chamfer over polylines; works unchanged once
  `flatten_primitive` knows ellipse.

### Institutional Learnings

- *Per-image exact-match rate is the real metric* (Stage 1 lesson). Carry it
  forward unchanged — works for ellipses out of the box because it's
  type-agnostic at the F1 level.
- *Autoregressive doesn't need beam search at sufficient capacity* (Stage 1).
  Greedy is fine.
- *Periodic checkpointing every N epochs* — keep the same recipe.
- *Generate, don't extract* — analytic projection of the circular rim gives
  exact ellipse parameters; no fitting from a rendered image.

## Key Technical Decisions

- **Ellipse representation: `(cx, cy, a, b, theta)` in canvas coords** —
  center + two semi-axes + rotation in radians. Five floats; fits the
  existing `N_PARAMS = 5` exactly.
- **Circle stays as its own type.** `Circle` is `a = b, theta = 0`; you could
  redundantly represent every circle as an ellipse, but keeping them distinct
  preserves the existing tokenizer/metric path and lets the model choose the
  cleaner output when one fits. The 2D circle results we already have stay
  valid.
- **Tokenizer: ellipse is `TYPE_ELLIPSE cx cy a b theta_token`.** Six tokens
  per primitive vs four for circle. Theta needs its own quantization block
  (range `[0, π)` since ellipse is symmetric under θ + π). Reuse the existing
  coord-quantization for `cx, cy, a, b`.
- **Canonical-ordering rule for ellipses:** if `b > a` (i.e. theta unusable
  because semi-major and semi-minor are swapped), swap and rotate by π/2.
  This keeps the same drawing always producing the same token sequence —
  load-bearing for the autoregressive bet.
- **3D cylinder rim projection is analytic, not numerical.** Project the unit
  circle on the rim's plane to image coords using the camera basis: the
  result is *exactly* an ellipse with closed-form `(a, b, theta)`. No fitting.
- **Cylinders in training data: mix with boxes.** A 50/50 mix (or
  `train_samples` from each generator) is the simplest curriculum extension.
  The model already handles type prediction; this just gives it ellipses to
  predict on cylinder inputs.

## Open Questions

### Resolved During Planning

- One new type or extend `Circle`? → New type (`Ellipse`); circle stays.
- 5 params or 6? → 5 (the existing slot count is sufficient).
- Warm-start from box checkpoint? → No (vocab changed; token embedding incompat).

### Deferred to Implementation

- Theta quantization granularity (64 bins covering `[0, π)` ≈ 2.8° each)
  vs (128 bins) — calibrate against rim-fit tolerance.
- Mix ratio cylinder:box in training (50/50 default; tune if cylinders
  under-learn).
- Whether `flatten_primitive` should sample uniformly in angle or
  arc-length for ellipses (angle is simpler; arc-length is more uniform).

## Implementation Units

The order is *the same* shape that worked for boxes (schema → tokenizer →
renderer/metric → 3D generator → train), but with a vocabulary extension
explicitly visible.

- [ ] **Unit E1: Ellipse primitive in the canonical schema**

**Goal:** add `Ellipse` to `lines/primitives.py` with the same surface as the
other primitives (dict round-trip, normalize/denormalize, validate,
approx_equal, canonical form).

**Requirements:** E1

**Files:**
- Modify: `lines/primitives.py`
- Test: `tests/test_primitives.py` (extend) or `tests/test_ellipse_primitive.py`

**Test scenarios:**
- Happy path: construct an ellipse, dict round-trip preserves it within tol.
- Edge: `b > a` canonicalizes to `(a', b', theta + π/2)` with `a' = b, b' = a`.
- Edge: theta wraps mod π (theta and theta+π represent the same ellipse).
- Edge: degenerate ellipse (a=0 or b=0) flagged invalid.
- Normalization: ellipse with `cx, cy, a, b` in canvas coords normalizes to
  unit-square coords, theta unchanged.

- [ ] **Unit E2: Tokenizer support for ellipses**

**Goal:** serialize/parse ellipses with the same canonical-ordering guarantee.

**Requirements:** E2

**Files:**
- Modify: `lines/models/seq_tokenizer.py`
- Test: `tests/test_seq_tokenizer.py` (extend)

**Test scenarios:**
- Vocab size matches: `3 + 4 type + 64 coord + 64 bulge + 64 theta`.
- Round-trip: ellipse → tokens → ellipse within bin tolerance.
- Canonical: ellipse with swapped axes (b>a) tokenizes identically to its
  canonicalized form.
- Mixed sets containing ellipse + line + circle round-trip.

- [ ] **Unit E3: Renderer + flatten support for ellipses**

**Goal:** `render_primitives` draws ellipses (including rotation), and
`flatten_primitive` produces a polyline good enough for Chamfer matching.

**Requirements:** E3, E4

**Files:**
- Modify: `lines/datagen/render.py`
- Test: `tests/test_render.py` (extend)

**Approach:**
- `flatten_primitive` for `Ellipse`: sample `n` points along the parametric
  ellipse `(cx + a cos t cos θ − b sin t sin θ, cy + a cos t sin θ + b sin t cos θ)`.
- `render_primitives` uses the flattened polyline path (same as it does for
  arcs / circles via line-list drawing).

**Test scenarios:**
- An ellipse with `theta=0` and `a=b` renders identically to a `Circle` of
  the same center/radius (within AA tolerance).
- A rotated ellipse renders with the major axis at the expected angle.
- `flatten_primitive(ellipse)` returns the right number of points; first and
  last point coincide (closed curve).

- [ ] **Unit E4: Differentiable renderer + diffvg refinement for ellipses**

**Goal:** the existing soft-render and per-image gradient refinement paths
handle ellipses.

**Requirements:** E5

**Files:**
- Modify: `lines/refine/diffvg_refine.py` (add `_ellipse_polyline`)
- Modify (if used): `lines/models/soft_render.py` for any train-time render
  loss

**Test scenarios:**
- Perturbed ellipse (semi-axis off by ε) refines toward the target.
- Identity case: a correctly-placed ellipse stays put.

- [ ] **Unit E5: Cylinder 3D scene generator**

**Goal:** projected 3D cylinders → 2 lines + 2 ellipses with exact ground truth.

**Requirements:** E6

**Files:**
- Create: `lines/datagen/cylinder_scene.py`
- Test: `tests/test_cylinder_scene.py`

**Approach:**
- Cylinder mesh: center axis + radius + height. Visible edges =
  - 2 silhouette lines (parallel to the axis, perpendicular to the projection
    of the axis on the image plane);
  - top rim ellipse (always visible);
  - bottom rim ellipse if the camera sees it (back-face culled on view angle).
- Rim ellipse from projection: project the rim's circle plane to image; the
  result is the parametric ellipse with known `(cx, cy, a, b, theta)` from
  the camera basis vectors.

**Test scenarios:**
- Side-on cylinder: 2 silhouette lines + 1 visible rim (top) + 1 ellipse
  degenerate to a near-line (the far rim, seen edge-on) — or omitted entirely
  if behind silhouette.
- Top-down cylinder: only the top circle (degenerate ellipse with `a = b`).
- Oblique cylinder: 2 lines + 2 ellipses (top and bottom rims), majors
  perpendicular to the axis projection.

- [ ] **Unit E6: Cylinder data split + training run**

**Goal:** train the existing autoregressive model on a cylinder split and
measure F1 / exact-match on a held-out set.

**Requirements:** E7

**Files:**
- Create: `scripts/build_cylinder_data.py` (mirror of `build_box_data.py`)
- Modify: notebook to add a "train on cylinders" variant (or a `data_kind`
  selector that picks boxes vs cylinders vs mixed)
- Test: `tests/test_train_autoregressive.py` smoke test with cylinder dir

**Approach:**
- Generate `train64_cyl` (10k) + `test64_cyl` (400) using the same
  big-preset recipe that worked for boxes.
- Run the autoregressive trainer (Colab T4, ~15–20 min).
- Score under strict F1 + exact-match; visualize.

**Success criterion:** exact-match ≥ 0.8 on held-out cylinders.

- [ ] **Unit E7: Mixed train (boxes + cylinders) — the real Stage 2 ship**

**Goal:** one model that does both content types competently. Demonstrates the
vocabulary extension doesn't break the box result.

**Requirements:** E7

**Files:**
- Modify: notebook for the mixed-training variant

**Approach:**
- Generate `train64_mixed` (10k total, 50/50 boxes + cylinders).
- Train; score on three held-out splits: boxes-only, cylinders-only, mixed.
- Compare against the pure-box Stage 1 result and the pure-cylinder Unit E6
  result.

**Success criterion:** mixed model's exact-match on boxes is within 5pp of
the pure-box Stage 1 number (≥ 0.88) **and** exact-match on cylinders is ≥
0.8. No catastrophic interference.

## Decision Gates

| Gate | After | Decision |
|------|-------|----------|
| 1 | E2 | Tokenizer round-trips ellipses exactly; canonical-ordering deterministic. If not, fix before any model code. |
| 2 | E5 | Cylinder ground truth is analytically correct (visual check + a couple of hand-computed test cases). If not, no model can learn from broken labels. |
| 3 | E6 | Cylinder-only exact-match ≥ 0.8. If yes, proceed to E7 (mixed). If 0.5–0.8, more epochs / capacity. If < 0.5, ellipse representation is harder than predicted — revisit param encoding (maybe arc-length parameterization, maybe a different tokenization). |
| 4 | E7 | Mixed model holds boxes (≥ 0.88) AND cylinders (≥ 0.8). If yes, **Stage 2 ships**; resume Stage 3. If boxes regress, try larger model or split-conditioned training. |

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Theta quantization too coarse for narrow ellipses | Calibrate vs visual at Unit E3; bump bins to 128 if needed |
| Mixed training causes catastrophic interference (model gets worse at boxes when cylinders are added) | Gate 4 catches it; mitigation is bigger model / type-conditioned queries / separate models |
| Differentiable ellipse rendering is numerically fragile near `a == b` (degenerate to circle) | Clamp `|a − b|` away from zero in the refinement, or fall back to circle path when that close |
| Cylinder generator produces near-degenerate ellipses on near-axis-aligned views | Cap view angle / reject samples where `b / a < threshold`; mirrors the arc-sweep cap we used in 2D |

## Sources & References

- **Origin (roadmap):** [docs/3d-roadmap.md](../3d-roadmap.md) Stage 2 section
- **A2 bake-off (capacity proof on concentric circles):** [docs/a2-prototype-scope.md](../a2-prototype-scope.md)
- **Stage 1 ship config:** [docs/best-config.md](../best-config.md), AR big preset.
- **Honest metric used throughout:** Unit M1 of
  [the structure-aware reconstruction plan](2026-06-18-001-feat-structure-aware-reconstruction-plan.md).
