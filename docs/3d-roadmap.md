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

#### Stage 1 result (measured)

The pipeline works end to end; clean reconstruction does not. A 16-query model
trained on 4000 projected box scenes, scored on 400 held-out boxes (64px,
threshold 0.50 + algebraic refine):

| Predictor | Score | RenderIoU | TypeAcc | Coverage |
|-----------|-------|-----------|---------|----------|
| classical baseline | 0.200 | 0.420 | 0.013 | 0.114 |
| 2D model (8q, untrained on boxes) | 0.299 | 0.450 | 0.236 | 0.336 |
| **box model (16q)** | **0.629** | 0.565 | 1.000 | 0.696 |

**Two-sided finding:**

- *Pipeline validated.* Analytic projection → exact Line ground truth → train →
  measure runs end to end, and the box model roughly doubles the best
  before-number (0.299 → 0.629), with perfect type accuracy.
- *Reconstruction is NOT clean.* Qualitatively the predictions are a tangle of
  overlapping segments around the silhouette, not a 9-edge wireframe with
  corners that meet. The model over-predicts (~12.9 lines for an ~8.8-edge box).
  Render-IoU rewards rough ink coverage and **masks** the structural error.

**Root cause — the architecture inflection, arriving early.** A box is 9 lines
meeting at 7 shared corners; bounded-N set prediction has no mechanism for that
edge-adjacency (the same class of failure as 2D concentric circles — a
*relationship* the model cannot represent). This was predicted for Stage 3; it
shows up at Stage 1.

**Decision implication.** Before Stage 2 (cylinders/ellipses), address the
architecture and metric:
- *Architecture:* explicit connectivity/junction modeling, or the documented
  scale-path (autoregressive "CAD-as-language" sequence generation), rather than
  pushing bounded-N set prediction further.
- *Metric:* add a structure-aware term (junction/corner correctness, stronger
  over-prediction penalty) so a tangle cannot score 0.63. Render-IoU alone is
  too forgiving for connected wireframes.

Also exposed and fixed this stage: training now checkpoints every N epochs
(a sleep killed the first run at epoch 26 with nothing saved).

#### Stage 1 — RESOLVED (autoregressive recipe)

The architecture-inflection above triggered the M1/A1/A2/A3 program in
`docs/plans/2026-06-18-001-feat-structure-aware-reconstruction-plan.md`: a
strict primitive-F1 metric (M1), a YAGNI structural post-process that was
proven insufficient (A1, Gate 2 → architecture rewrite justified), the A2
bake-off that picked autoregressive over vertex-graph, and A3 which built the
training entry point and validated on a CPU dry-run before going to GPU.

Final box result, GPU on free Colab T4 (~15 min), `data/test64_box` 400 held-out:

| Predictor | F1 | Exact-match | Near-match |
|-----------|-----|-------------|------------|
| classical baseline | 0.000 | 0.000 | 0.000 |
| set predictor + structure post-process | 0.282 | n/a | n/a |
| AR small preset (d=192, 3L, 50ep, 4k train) | 0.899 | 0.642 | 0.650 |
| **AR big preset (d=256, 5L, 100ep, 10k train)** | **0.980** | **0.932** | **0.932** |

373 / 400 boxes reconstructed *exactly*. The bimodal failure mode (~33%
"catastrophes") observed in the small preset is gone: exact and near are now
equal, meaning remaining failures are genuine "one wrong edge" cases, not
"fell off the rails entirely." Beam-3 added +0.5pp exact -- nominal; greedy
suffices once the model has the capacity. **Stage 1 ships at this recipe.**

The single empirical lesson: render-IoU-weighted mean F1 was always a partial
truth; per-image `exact_match_rate` is the metric that matched the visual.
This is folded permanently into `lines/eval/harness.py`.

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

#### Stage 2 Unit E6 — RESOLVED (cylinder-only ship)

Same recipe as Stage 1 (d=256, 5 decoder layers, 100 epochs, 10k samples) on
`data/train64_cyl` (3 primitives per scene: 2 silhouette lines + 1 rim ellipse).
Held-out 400 cylinders:

| Predictor | F1 | Exact-match | Near-match |
|-----------|-----|-------------|------------|
| classical baseline | 0.000 | 0.000 | 0.000 |
| AR boxes (Stage 1 ship, recorded) | 0.980 | 0.932 | 0.932 |
| **AR cylinders (Stage 2 Unit E6 ship)** | **0.995** | **0.988** | **0.988** |

395 / 400 cylinders reconstructed *exactly*. Greedy decoding; beam-3 added
**+0.000** -- the model is confident enough that decoder strategy is
irrelevant. Stage 2 Unit E6 ships at this recipe.

The empirical insight Stage 1 *predicted* and this run *confirms*: the
difficulty of structured reconstruction scales with the **number of inter-
primitive relationships**, not the **number of primitive types**. Cylinders
have 3 primitives (2 corner-pairs); boxes have 9 primitives (7 shared corners).
Cylinders are *easier*, even with the new Ellipse vocabulary, because there is
less structural commitment to coordinate. The Ellipse extension itself worked
end-to-end on the first try (no retry, no recipe tuning) -- the canonical-form
rule (E1), theta quantization (E2), and SVD-based analytic rim projection (E5)
all held up exactly as designed.

What's next: Unit E7 -- one model on a mixed 50/50 boxes + cylinders training
set, to verify the same architecture does both content types without
catastrophic interference. Gate 4: cylinders exact >= 0.80 AND boxes exact
within 5pp of 0.932 (so >= 0.88).

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
