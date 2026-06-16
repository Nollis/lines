# Vector Primitive Extraction AutoResearch Program

Optimize the set-prediction model to extract CAD/illustration-grade geometric vector primitives from line-art raster images.

## Goal
Improve the model's performance on the test split, measured by `mean_score` (type accuracy + normalized parameter error + render-based IoU).
The model must beat the classical baseline reference of **`mean_score = 0.612`**.

## Editable Files
You are permitted to modify the following files to optimize the model architecture, hyperparameters, losses, and learning process:
- [lines/models/set_predictor.py](file:///e:/Projekt/Lines/lines/models/set_predictor.py) — Model architecture (CNN backbone, Transformer decoder layers, queries, etc.)
- [lines/models/losses.py](file:///e:/Projekt/Lines/lines/models/losses.py) — Loss weights and definitions
- [lines/train/train.py](file:///e:/Projekt/Lines/lines/train/train.py) — Hyperparameters, learning rate, and optimizer settings

## Evaluation Loop

To ensure quick iterations (under 2 minutes on CPU), you should run a **fast proxy task** for your experimental runs, and only run the full 120-epoch training to verify final candidates.

### 1. Fast Proxy Task (Run this for proposing and validating changes)
Run a short training run of 15 epochs on a subset of 1000 samples with memory caching and downsampled 32x32 rendering loss:
```bash
python -m lines.train.train --epochs 15 --train-samples 1000 --test-samples 100 --render-canvas-size 32
```
Verify the output log:
* Look at the training loss trajectory (it should show steady convergence).
* Look at the final evaluation report on the test set. Monitor the **`mean_score`** metric.
* Keep changes if the proxy `mean_score` improves over your previous proxy baseline.

### 2. Full Evaluation (Run this to verify final optimal configurations)
Once you find a high-performing configuration, run a full 120-epoch training on the complete dataset to verify against the classical baseline:
```bash
python -m lines.train.train --epochs 120 --train-samples 4000 --test-samples 400 --render-canvas-size 64
```
Target: Beat the classical baseline score of **`mean_score = 0.612`**.

## Key Rules & Guidelines
1. **Never game the metric**: Keep the dataset creation and evaluation harness unchanged (`prepare.py` is read-only).
2. **CPU-friendly**: Optimize for execution on CPU. Keep backbones small (e.g. compact CNNs) and decoder layers shallow to avoid memory/time blowups.
3. **Save progress**: When a proxy run shows an improvement, git commit the changes before proposing the next iteration. If a proposal degrades performance, git reset/revert to the last stable commit.
