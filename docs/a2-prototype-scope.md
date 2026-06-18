# A2: Architecture prototype bake-off (scope)

Sub-scope of [the structure-aware reconstruction plan](plans/2026-06-18-001-feat-structure-aware-reconstruction-plan.md),
Unit A2. Gate 2 ruled out the cheap fix (post-process: box F1 0.262 → 0.282,
far from a ~0.9 target). Set prediction cannot represent inter-primitive
relationships (shared corners, shared centers), so a relationship-capable
architecture is needed. A2 *chooses* that architecture via a cheap, decisive
bake-off before A3 builds it.

## Guiding principle: test representation capacity, not full training

The question A2 must answer is **"can this architecture even represent clean
structured output?"** — not "what's its production accuracy." That makes the
prototype an **overfit-a-tiny-set test**: a small model on ~100–300 samples for
a few hundred steps. If an architecture cannot overfit 200 boxes to high F1, it
cannot represent the junction structure and is disqualified — cheaply, in
minutes, on CPU. Full-scale training (and GPU) is A3, *after* the choice.

This keeps A2 doable now without GPU, and makes the decision evidence-based
rather than a paper comparison.

## Candidates

### Candidate V — Vertex-graph

Predict the drawing as a graph: a set of **vertices** (corners/junctions), then
**edges** as connections between them.

- *Vertex head:* `N_v` point queries → each predicts `(exists, x, y)` (DETR-for-
  points / keypoint detection).
- *Edge head:* for each ordered/unordered pair of detected vertices, a bilinear
  or small-MLP scorer predicts "edge present" → an adjacency matrix.
- *Decode:* each present edge between two vertices → a `Line` primitive sharing
  exact endpoints with its neighbors.

**Strengths:** junction structure is *built in* — edges share vertices by
construction, so a box's corners meet exactly and the tangle is impossible.
Lines are exact (vertex-to-vertex). Elegant for polygonal line art.

**Weaknesses:** **curves don't fit** — a circle has no vertices, so circles/
arcs/ellipses need a separate parallel primitive channel (a hybrid). And it does
**not** model the 2D concentric-circle relationship (shared *center*, not shared
vertex). So it solves boxes but not the broader relationship problem the project
has hit twice.

### Candidate A — Autoregressive (CAD-as-language)

Generate primitives as a **token sequence** where each primitive conditions on
all previously emitted ones.

- *Tokenizer:* quantize coordinates to bins (e.g. 64 bins / 64px) and serialize
  a primitive set in a **canonical order** as
  `[START, LINE, x1,y1,x2,y2, LINE, ..., CIRCLE, cx,cy,r, ..., END]`.
- *Model:* a GPT-style transformer decoder over the token sequence,
  image-conditioned via cross-attention to the existing CNN encoder.
- *Train:* teacher-forced next-token prediction. *Infer:* autoregressive
  sampling to `END`.

**Strengths:** models **any** relationship — shared corners, shared centers,
anything — because every token sees the full history. Quantization makes a
re-used corner land in the *same bin* → the *same token*, so shared points come
out exact. Unifies all primitive types in one vocabulary (lines/arcs/circles,
and later **ellipses** — the Stage-2 extension comes nearly for free). Handles
unbounded primitive counts. The proven approach (DeepCAD, PolyGen, SketchGraphs).

**Weaknesses:** more machinery — coordinate quantization, **canonical ordering**
(the same drawing must always serialize identically, or the model cannot learn),
teacher forcing, slow autoregressive inference, sequence-level loss. Quantization
caps sub-pixel precision (recoverable via the existing refinement pass).

## The two discriminating tests

One test each candidate should pass, one that separates them:

1. **Box overfit (junction structure).** Overfit ~200 box scenes. *Both* should
   reach high F1 if they can represent edge-adjacency. If V passes cleanly but A
   struggles, that is signal toward V.
2. **Concentric-circle overfit (non-vertex relationship + curves).** Overfit
   ~200 concentric-circle scenes (from the existing `technical_layout`). **V is
   expected to fail** (no vertices; shared-center is not a graph edge); **A is
   expected to pass** (it reuses the center coordinate token). This is the test
   that decides generality.

The project has *already* been bitten by both failure modes — boxes (shared
corners) and concentric circles (shared center). An architecture that only fixes
one is a local patch. Test 2 is therefore the load-bearing comparison.

## Decision rubric

| Criterion | Vertex-graph (V) | Autoregressive (A) |
|-----------|------------------|--------------------|
| Box junction structure | built-in (strong) | learned (should be ok) |
| Concentric / shared-center | not modeled | modeled |
| Curves (circle/arc/ellipse) | needs separate channel | native in vocabulary |
| Stage-2 ellipse extension | extra work | ~free (new token) |
| Unbounded primitive count | capped by `N_v` | native |
| Implementation cost | medium | higher |
| Inference speed | one pass | autoregressive (slower) |
| Sub-pixel precision | exact | quantized (+ refine) |

**Hypothesis (to confirm, not assume):** A is the more general answer — it
addresses every relationship the project has hit and makes Stage 2 nearly free —
and the bake-off's job is to confirm it is *tractable* (overfits boxes) rather
than to discover whether V is more elegant on boxes alone. V remains the pick
only if A cannot overfit boxes or is disproportionately harder to train.

## Prototype build scope (minimal)

Build only enough to run the two overfit tests, smallest viable:

- Candidate A: `lines/models/seq_tokenizer.py` (quantize + canonical serialize +
  parse-back) and `lines/models/autoregressive.py` (image-conditioned decoder).
  TDD the tokenizer round-trip first (it is the load-bearing, bug-prone part:
  serialize → parse must be exact; canonical order must be deterministic).
- Candidate V: `lines/models/vertex_graph.py` (vertex point head + edge-adjacency
  head) and a box-graph ground-truth builder (box scene → vertices + adjacency).
- A tiny shared harness: overfit N samples, report train-set F1 (capacity) and
  held-out F1 on a few samples, using the M1 metric.

No new training infrastructure beyond a minimal overfit loop. Tiny models
(d≈64–128, 2 layers). ~100–300 samples. A few hundred steps.

## Budget & gating

- **CPU-feasible, minutes-scale per overfit run.** No GPU for A2.
- Decide via the rubric + the two test results. Record the choice and rationale
  (Gate 3 in the parent plan).
- **GPU is for A3** (full training of the chosen architecture), scoped after the
  decision.

## Fallback

If A overfits boxes but its precision is poor and V is clearly better on boxes,
consider a **hybrid**: vertex-graph for line/edge structure + an autoregressive
or set channel for curves. More complex than either pure option — only if the
bake-off shows neither pure candidate suffices.

## Exit criteria

A2 is done when there is: (1) a passing tokenizer round-trip + a working overfit
of at least the favored candidate on both box and concentric tests, (2) a filled
rubric with measured F1s, and (3) a recorded architecture decision feeding A3.
