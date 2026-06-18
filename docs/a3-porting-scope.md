# A3: porting + first-run scope

Sub-scope of [the structure-aware reconstruction plan](plans/2026-06-18-001-feat-structure-aware-reconstruction-plan.md),
Unit A3. Gate 3 chose autoregressive; A2 proved capacity (boxes F1 0.998,
concentric F1 0.995 on overfit-the-set). A3 builds it at training scale and
measures honest F1 on *held-out* content.

The point of *this* doc is to make the work doable so that when GPU is needed,
opening Colab/Modal/etc. is a 30-minute setup, not half a day. Everything below
is TDD-able locally on CPU before paying for any GPU time.

## Two truths to keep in mind

1. **The bake-off model is small** (d=128, 2 layers, ~4 min for 600 steps over
   64 samples on CPU). Production A3 is bigger (d≈192, 3-4 layers, full 4000
   samples for ~30-50 epochs). A first real run might still fit on CPU in a few
   hours — GPU is the *enabler*, not strictly required for Round 1.
2. **Autoregressive inference is slow.** Greedy decoding is per-token; eval on
   400 samples × ~50 tokens × forward pass each is *minutes*, not seconds. GPU
   matters more for inference throughput than for training in this regime.

## Units

- [ ] **Unit P1: device-agnostic refactor**

**Goal:** every training/inference path accepts `--device {cpu,cuda}` and runs
identically on either. Local CPU tests still pass; GPU tests are skipped on
CPU-only systems.

**Files:**
- Modify: `lines/train/train.py`, `lines/train/predictor.py`,
  `lines/refine/diffvg_refine.py`, `lines/models/autoregressive.py`,
  `lines/models/set_predictor.py` (already mostly there)
- Test: extend existing model/loss tests with a `cuda` parametrize guarded by
  `pytest.skipif(not torch.cuda.is_available())`

**Approach:**
- Every `.to(...)`, every `torch.zeros(...)` that needs gradients, every
  registered buffer (`SoftRenderer.grid` already does this — good) must carry a
  device. The `HungarianMatcher` boundary (which calls `scipy.linear_sum_assignment`)
  already does `.cpu().numpy()` — leave as-is.
- `diffvg_refine` builds a fresh `_build_grid` per call; move that to the input
  image's device (currently CPU-only). The biggest GPU win lives here — DiffVG
  refinement is currently the long pole on eval (~3s/img at 128).
- `Dataset.cache_images=True` keeps images in CPU RAM — fine, batches transfer
  to GPU at `__getitem__` time.

**Verification:** `pytest -q` passes on CPU; adding `--device cuda` to a tiny
overfit run trains the autoregressive prototype to ~0 loss on 8 boxes.

- [ ] **Unit P2: autoregressive training entry point**

**Goal:** a sibling to `lines/train/train.py` for the autoregressive model.
Periodic checkpointing (we learned this lesson), warm-start, device flag.

**Files:**
- Create: `lines/train/train_autoregressive.py`
- Test: `tests/test_train_autoregressive.py` — 1-epoch smoke test (same shape
  as `test_train.py`)

**Approach:**
- Reuse `Dataset` and the existing manifest format; just swap the model + loss
  for `AutoregressiveModel` + `teacher_forced_loss`.
- Tokenize each batch on-the-fly (cheap; same as the bake-off script).
- Pad to the batch's max sequence length; use `PAD` ignore-index in the loss.
- AdamW, cosine schedule over `epochs * steps_per_epoch`, grad clip 1.0 — the
  same recipe that worked for the set predictor.
- Checkpoint every N epochs (we know why; same as `train.py`).
- Final eval: greedy-sample each held-out image, decode tokens via the
  tokenizer, run `evaluate()` from `lines.eval.metrics` (the F1 metric).

**Verification:** 1-epoch smoke test runs end to end on 8 samples and writes a
checkpoint that can be reloaded.

- [ ] **Unit P3: A3 first-run config + dry run on CPU**

**Goal:** establish the recipe locally so the GPU run is mechanical.

**Files:**
- Create: `configs/a3_box64.yaml` (recipe), `scripts/a3_first_run.sh` or
  README snippet

**Approach (starting point, expect to tune):**
- Data: existing `data/train64_box` (4000 boxes), `data/test64_box` (400).
- Architecture: d=192, 3-4 decoder layers, 4 heads (slight bump over the
  prototype that overfit 64 boxes).
- Optim: AdamW, lr=3e-4 (lower than the prototype's 3e-3 — production data is
  bigger and overfits slower), wd=1e-4, batch=32.
- Schedule: 30-50 epochs to start (anneal cosine).
- *Dry run:* train for 2 epochs on CPU, confirm loss is dropping, then stop.
  This catches any wiring/recipe bugs before paying for GPU.

**Verification:** 2-epoch CPU dry-run shows loss curve heading down; the
checkpoint loads back; greedy-sample on a held-out image returns a valid (if
not yet accurate) token stream.

- [ ] **Unit P4: the GPU run + measurement (Colab / Modal / wherever)**

**Goal:** train to convergence and measure honest F1 on held-out boxes (and
on the concentric-circle probe as a bonus relationship test).

**Approach:**
- Git pull + `pip install -r` on the GPU box, run `train_autoregressive.py`
  with the P3 config + `--device cuda`.
- Eval with `scripts/eval_generalization.py` extended for the autoregressive
  predictor — needs a small wrapper analogous to `ModelPredictor` that calls
  `greedy_sample` + `Tokenizer.decode`.
- Record F1 in `docs/results.md` alongside the existing numbers.

**Reference points (the bar to beat):**

| Predictor on test64_box (400 held-out) | F1 (already measured) |
|---|---|
| classical baseline | 0.000 |
| set predictor (16q) | 0.262 |
| set predictor + structure post-process | 0.282 |
| autoregressive (overfit on 64 train samples) | 0.998 |
| **autoregressive (A3, *held-out* generalization)** | **?** — the real test |

**Verification:** held-out F1 markedly above 0.282 (set predictor + post-process)
is the threshold for "A3 worked"; F1 near the bake-off's 0.998 would say "the
generalization is as clean as the capacity."

## Risks specific to A3

| Risk | Mitigation |
|------|------------|
| **Slow greedy inference** dominates eval time | Batched parallel greedy (one image at a time but many images in parallel); GPU helps a lot here |
| **Exposure bias** — teacher-forced training but its own predictions at inference | Document but defer; classical mitigation is scheduled sampling, only worth it if eval F1 << train F1 |
| **Long sequences** for dense content (Stage-2 ellipses, CSG parts later) | Token budget is fine for boxes (~50) and concentric (~16); revisit at Stage 2 |
| **First-run hyperparameters wrong** | The CPU dry-run in P3 catches the worst miscalibrations cheaply |
| **Colab/etc. disconnects mid-run** | Periodic checkpointing already lands in P2; can resume via `--init-from` |

## How to start

1. P1 + P2 + P3 are all CPU-local TDD work — couple hours, no spend.
2. Run the P3 dry-run; if it learns, push to git.
3. Open Colab Pro or whichever runtime, `git pull`, run the P4 command. The
   bake-off's 64-sample run took 4 min on CPU; the 4000-sample run at ~60×
   the data and ~3× the model is plausibly 1-3h on a T4. Likely overnight on
   CPU if you don't want to pay yet.

## Success criteria for A3 as a whole

- Held-out box F1 ≥ 0.7 (close to bake-off capacity, well above the
  set-predictor+post-process 0.282) ⇒ A3 succeeded; resume Stage-2 (ellipses)
  on this architecture.
- Held-out box F1 between 0.282 and 0.7 ⇒ partial win; investigate exposure
  bias, sampling temperature, sequence augmentation.
- Held-out box F1 ≤ 0.282 ⇒ unexpected; revisit either the data scale, the
  training recipe, or the canonical-ordering choice (the tokenizer's quietest
  load-bearing assumption).
