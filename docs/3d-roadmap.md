# 3D Roadmap: from boxes to mechanical drawings

The 2D walking skeleton is done and the data-improvement loop is proven. 3D is
the next content frontier (plan Unit 8). This roadmap stages it so each step is
small, measurable, and reuses as much as possible.

## Guiding decisions (fixed)

1. **Analytic projection, not Blender.** We control the 3D geometry, so we
   compute projected edges directly (project vertices, determine visibility) and
   *know* each 2D primitive exactly. Blender's Freestyle would re-introduce the
   "what primitive is this stroke?" problem we avoid in 2D. No heavy dependency,
   cleaner ground truth. Same load-bearing insight as the 2D generator.
2. **Orthographic projection** — the technical-illustration standard; no
   perspective foreshortening to model.
3. **Same compound loop every stage:** build generator → probe to measure the
   gap → fold into training → retrain → measure on a *fresh* held-out probe →
   document. This is the method we just validated; 3D is new content for it.
4. **Reuse schema / metric / model unchanged until a stage forces a change**,
   then extend deliberately.

## The vocabulary problem (why staging matters)

A box's edges are all straight → project to **lines** (existing vocabulary,
zero change). A cylinder's circular rim viewed at an angle projects to an
**ellipse** — which line/arc/circle *cannot represent*. So cylinders force a
schema + model + metric extension. Boxes don't. That split defines the stages.

## Stages

### Stage 0 — Projection core (infrastructure, no model)
- 3D mesh representation (vertices, edges, faces) for parametric parts.
- Orthographic camera: random view direction, project 3D → 2D.
- Visibility for convex solids: back-face culling; silhouette = edge between a
  front- and back-facing face.
- Output: visible edges as 2D segments with exact endpoints.
- Pure numpy, fully TDD. Reused by every later stage.
- **Done when:** a known cube at a known view projects to the expected visible
  edge set (verified against hand-computed geometry).

### Stage 1 — Boxes → lines (3D walking skeleton)
- Random boxes in random orientation → visible edges as `Line` primitives.
- **Only change:** bump `n_queries` (a generic box view shows ~9 visible edges
  > the current 8). Go to 16; the architecture already parameterizes this.
- Reuses schema, metric, encoding, refinement unchanged.
- Run the compound loop: probe the current 2D model on box renders (measure the
  gap), then fold boxes into training and retrain.
- **Milestone:** model recovers box wireframes from a single projected view.
- **Note:** the model must be retrained (new query count + new content); the 2D
  checkpoint warm-starts everything except the query embeddings.

### Stage 2 — Cylinders → lines + ellipses (vocabulary extension)
- Cylinder projects to 2 silhouette lines + 2 rim **ellipses** (or elliptical
  arcs when a rim is partly hidden).
- **Schema:** add `Ellipse(center, semi_major, semi_minor, rotation)` — 5
  params, which fits the existing `N_PARAMS = 5` exactly. Circle becomes the
  degenerate a = b, θ = 0 case (keep both for clean output).
- **Encoding/model:** new type id (`N_TYPES` 4 → 5: line/arc/circle/ellipse/
  none); the 5-slot param head is unchanged in size.
- **Metric/refinement:** `flatten_primitive` and the differentiable renderer
  gain an ellipse case; Chamfer/IoU then work unchanged.
- **Generator:** project the circle's plane analytically to get the ellipse
  parameters directly (no fitting needed).
- **Milestone:** ellipse primitive learned end-to-end; cylinders reconstructed.
- This is the largest single investment — it touches schema, model, and metric.

### Stage 3 — CSG features + hidden lines
- Subtractive features: holes (cylinder − box), slots, counterbores.
- **General hidden-line removal** (occlusion against all faces, not just
  back-face culling) — the hardest geometry here.
- Technical convention: hidden edges become **dashed**, not omitted → a
  line-style attribute on primitives (a new schema dimension).
- **Architecture inflection likely here:** primitive counts climb (15–30+),
  which strains bounded-N set prediction. This is where the plan's documented
  scale-path — autoregressive "CAD-as-language" sequence generation — may need
  to replace the DETR-style head. Decide based on Stage-2 count limits.

### Stage 4 — Realistic parts & drawing conventions
- Multi-feature parts, centerlines, the mechanical-drawing endgame. Out of scope
  until Stage 3 lands.

## Cross-cutting flags

- **Compute.** 3D generation + higher primitive counts + a possibly larger model
  intensify the CPU bottleneck already noted. A GPU becomes the practical
  enabler around Stage 2–3.
- **Held-out discipline carries over.** Each stage gets a fresh held-out probe
  (e.g. a box family the training never used) so generalization stays honestly
  measured — exactly as in the 2D loop.
- **Metric scores visible primitives only.** Since ground truth = visible
  (drawn) edges, the existing metric needs no change for occlusion.

## Recommended starting point

Stages 0 + 1 (projection core + boxes) are the minimal, zero-vocabulary-change
walking skeleton. They prove the 3D projection pipeline end-to-end and slot
straight into the existing model and loop. Commit to Stage 2 (ellipses) only
once boxes work — it's the real investment.
