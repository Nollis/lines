"""Train the set-prediction model on a generated dataset.

Run as a script: ``python -m lines.train.train --help`` for options.

The loop is deliberately small (CPU-friendly): 64x64 canvas, batch 32, AdamW
with cosine schedule. The eval harness (:mod:`lines.eval.harness`) runs the
trained model against the same metric used to score the classical baseline --
that comparison is the empirical gate the plan calls for.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from lines.datagen.dataset import Dataset, write_dataset
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor
from lines.models.encoding import N_PARAMS, encode_set
from lines.models.losses import SetPredictionLoss
from lines.models.matcher import HungarianMatcher
from lines.models.set_predictor import (
    SetPredictor, build_warm_started, required_feature_size,
)
from lines.train.predictor import ModelPredictor


@dataclass
class TrainConfig:
    canvas_side: int = 64
    render_canvas_size: int | None = 32
    train_samples: int = 4000
    test_samples: int = 400
    train_seed: int = 0
    test_seed: int = 900_000
    epochs: int = 25
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-4
    n_queries: int = 8
    d_model: int = 128
    n_decoder_layers: int = 3
    n_heads: int = 4
    class_weight_none: float = 0.2
    param_weight: float = 5.0
    render_weight: float = 1.0
    checkpoint_every: int = 5   # save model.pt every N epochs (crash/sleep safety)
    seed: int = 0


# --- dataset adapter ----------------------------------------------------------

class _TrainView(TorchDataset):
    def __init__(self, ds: Dataset, canvas: Canvas):
        self.ds = ds
        self.canvas = canvas

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, pset = self.ds[i]
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        t, p = encode_set(pset, self.canvas.width, self.canvas.height)
        return x, torch.from_numpy(t), torch.from_numpy(p)


def _collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    gt_types = [b[1] for b in batch]
    gt_params = [b[2] for b in batch]
    return images, gt_types, gt_params


# --- train --------------------------------------------------------------------

def train(cfg: TrainConfig, train_dir: Path, test_dir: Path, out_dir: Path,
          log=print, init_from: Path | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(cfg.canvas_side, cfg.canvas_side)

    if not (train_dir / "manifest.json").exists():
        log(f"generating {cfg.train_samples} train samples -> {train_dir}")
        write_dataset(train_dir, cfg.train_samples, seed=cfg.train_seed, canvas=canvas)
    if not (test_dir / "manifest.json").exists():
        log(f"generating {cfg.test_samples} test samples -> {test_dir}")
        write_dataset(test_dir, cfg.test_samples, seed=cfg.test_seed, canvas=canvas,
                      randomize=False)

    train_ds = Dataset(train_dir, max_samples=cfg.train_samples, cache_images=True)
    test_ds = Dataset(test_dir, max_samples=cfg.test_samples, cache_images=True)
    loader = DataLoader(_TrainView(train_ds, canvas), batch_size=cfg.batch_size,
                        shuffle=True, num_workers=0, collate_fn=_collate)

    if init_from is not None:
        model, src_cfg = build_warm_started(init_from, cfg.canvas_side)
        log(f"warm-started from {init_from} "
            f"(src canvas {src_cfg.get('canvas_side')} -> {cfg.canvas_side}, "
            f"encoder={src_cfg.get('encoder_type')})")
        # inherit the source architecture so cfg recorded in the checkpoint is accurate
        cfg.n_queries = int(src_cfg.get("n_queries", cfg.n_queries))
        cfg.d_model = int(src_cfg.get("d_model", cfg.d_model))
        cfg.n_heads = int(src_cfg.get("n_heads", cfg.n_heads))
        cfg.n_decoder_layers = int(src_cfg.get("n_decoder_layers", cfg.n_decoder_layers))
    else:
        model = SetPredictor(
            n_queries=cfg.n_queries, d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_decoder_layers=cfg.n_decoder_layers,
            feature_size=required_feature_size(cfg.canvas_side),
        )
    crit = SetPredictionLoss(matcher=HungarianMatcher(),
                             class_weight_none=cfg.class_weight_none,
                             param_weight=cfg.param_weight,
                             render_weight=cfg.render_weight,
                             canvas_size=cfg.canvas_side,
                             render_canvas_size=cfg.render_canvas_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs * steps_per_epoch)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for images, gt_types, gt_params in loader:
            logits, params = model(images)
            out = crit(logits, params, gt_types, gt_params, gt_images=images)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += out.loss.item()
            n_batches += 1
        mean_loss = running / n_batches
        elapsed = time.time() - t0
        log(f"epoch {epoch:02d}/{cfg.epochs}  loss={mean_loss:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "loss": mean_loss, "elapsed_s": elapsed})

        # periodic checkpoint so a crash/sleep never loses the whole run
        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "epoch": epoch}, out_dir / "model.pt")

    # final eval against the same metric used for the baseline
    model.eval()
    predictor = ModelPredictor(model, canvas)
    eval_report = run_predictor(predictor, test_ds, canvas)
    log(f"\n=== Test set ({cfg.test_samples} samples, {cfg.canvas_side}x{cfg.canvas_side}) ===")
    for k in ("mean_score", "mean_render_iou", "mean_type_accuracy",
              "mean_geometric_error", "mean_coverage"):
        log(f"  {k:22s} {eval_report[k]:.3f}")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
               out_dir / "model.pt")
    return {"history": history, "eval": eval_report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas-side", type=int, default=64)
    ap.add_argument("--train-dir", default="data/train64")
    ap.add_argument("--test-dir", default="data/test64")
    ap.add_argument("--out-dir", default="checkpoints/v1")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--train-samples", type=int, default=4000)
    ap.add_argument("--test-samples", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--param-weight", type=float, default=5.0)
    ap.add_argument("--render-weight", type=float, default=1.0)
    ap.add_argument("--render-canvas-size", type=int, default=32)
    ap.add_argument("--n-queries", type=int, default=8,
                    help="number of primitive queries (raise for content with "
                         "more primitives, e.g. 16 for projected boxes)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from this checkpoint (e.g. a model trained "
                         "at a lower resolution); pos_embed is interpolated")
    args = ap.parse_args()
    cfg = TrainConfig(canvas_side=args.canvas_side,
                      epochs=args.epochs,
                      train_samples=args.train_samples,
                      test_samples=args.test_samples,
                      lr=args.lr,
                      batch_size=args.batch_size,
                      n_queries=args.n_queries,
                      param_weight=args.param_weight,
                      render_weight=args.render_weight,
                      render_canvas_size=args.render_canvas_size)
    init_from = Path(args.init_from) if args.init_from else None
    train(cfg, Path(args.train_dir), Path(args.test_dir), Path(args.out_dir),
          init_from=init_from)


if __name__ == "__main__":
    main()
