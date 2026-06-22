# A3-P4: GPU handoff — the "open Colab and run" page

P1 (device refactor) and P2 (autoregressive trainer) are committed and tested.
P3 (CPU dry-run) is the recipe sanity check we just passed. P4 is the real
training run; this page exists so it's mechanical when you sit down to it.

## P3 dry-run result (the recipe works)

3 epochs on `data/train64_box` (4000 boxes), `d_model=192`, 3 decoder layers,
`lr=3e-4`, batch 32, CPU:

| Metric | After 3 epochs |
|--------|----------------|
| Mean loss (train) | 2.85 → 1.75 (−39%) |
| **Held-out F1** | **0.105** |
| Held-out precision | 0.111 |
| Held-out recall | 0.103 |
| Held-out render-IoU | 0.260 |

Already non-trivial after <3 min of training. Reference points from the
plan:

| Predictor | held-out box F1 |
|-----------|-----------------|
| classical baseline | 0.000 |
| set predictor (16q) | 0.262 |
| set predictor + structure post-process | 0.282 |
| autoregressive @ 3 epochs (dry-run) | 0.105 |
| **target after full training** | **≥ 0.7** |

Throughput observed: ~40 s/epoch CPU. A 50-epoch run is ~33 min CPU,
plausibly ~3–5 min on a Colab T4.

## The one command

After `git pull` on a fresh Colab/Modal/RunPod box with a CUDA PyTorch:

```bash
pip install -e .                   # picks up pyproject deps
python -m lines.train.train_autoregressive \
    --canvas-side 64 \
    --train-dir data/train64_box \
    --test-dir data/test64_box \
    --out-dir checkpoints/ar_box64_gpu \
    --epochs 50 \
    --batch-size 64 \
    --d-model 192 \
    --n-decoder-layers 3 \
    --lr 3e-4 \
    --device cuda
```

Notes that matter:

- **The data lives in the repo's `data/` dir, which is gitignored**.
  Either commit it temporarily for the Colab run, or regenerate it with
  `scripts/build_box_data.py` (deterministic — same seeds give the same data).
- **Periodic checkpointing is on**, so a disconnect mid-run costs at most 5
  epochs. Resume with `--init-from checkpoints/ar_box64_gpu/model.pt`.
- **Watch the loss in the first 5 epochs**: if it doesn't fall below 1.5 in 5
  epochs at GPU speed, the recipe is wrong, not just under-trained — lower the
  lr or check `--d-model` (must be a multiple of `n-heads`).

## Score it through the existing scoreboard

After training finishes, the existing eval script needs a small extension to
recognize the autoregressive arch tag in the checkpoint — until then, score
manually:

```python
from lines.datagen.dataset import Dataset
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor
from lines.models.autoregressive import AutoregressiveModel
from lines.train.predictor_ar import AutoregressivePredictor
import torch

ck = torch.load("checkpoints/ar_box64_gpu/model.pt", map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = AutoregressiveModel(canvas_side=cfg["canvas_side"], d_model=cfg["d_model"],
                            n_heads=cfg["n_heads"], n_decoder_layers=cfg["n_decoder_layers"],
                            max_seq_len=cfg["max_seq_len"])
model.load_state_dict(ck["model"])
C = Canvas(cfg["canvas_side"], cfg["canvas_side"])
ds = Dataset("data/test64_box")
rep = run_predictor(AutoregressivePredictor(model, C, max_tokens=cfg["max_seq_len"]),
                    ds, C)
print({k: rep[k] for k in ("mean_f1", "mean_precision", "mean_recall", "mean_render_iou")})
```

## Success rubric

- **F1 ≥ 0.7** ⇒ A3 succeeded; resume Stage 2 (cylinders/ellipses) on this
  architecture, add an `arch: "autoregressive"` branch to the scoreboard CLI.
- **F1 between 0.282 and 0.7** ⇒ partial win; first thing to try is more
  epochs (the dry-run was nowhere near converged at 3). Then sampling
  temperature / nucleus, then more data, then exposure-bias mitigations.
- **F1 ≤ 0.282** ⇒ unexpected; check tokenizer canonical-ordering against
  what the model emits, and verify the CPU dry-run baseline reproduces.
