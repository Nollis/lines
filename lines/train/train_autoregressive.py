"""Train the autoregressive (CAD-as-language) model (A3-P2).

Sibling to ``lines.train.train`` (the set predictor). Tokenizes each
:class:`PrimitiveSet` on the fly, teacher-forces next-token prediction, and
evaluates by greedy-sampling on the held-out split under the strict F1 metric.

Periodic checkpointing (we learned this lesson) + ``--device`` + ``--init-from``
are mirrored from the set-predictor trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from lines.datagen.dataset import Dataset, write_dataset
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor
from lines.models.autoregressive import AutoregressiveModel, teacher_forced_loss
from lines.models.seq_tokenizer import EOS, PAD, Tokenizer
from lines.train.predictor_ar import AutoregressivePredictor


# ----------------------------------------------------------------------------- config

@dataclass
class ARTrainConfig:
    canvas_side: int = 64
    train_samples: int = 4000
    test_samples: int = 400
    train_seed: int = 0
    test_seed: int = 900_000
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    d_model: int = 192
    n_decoder_layers: int = 3
    n_heads: int = 4
    max_seq_len: int = 128
    checkpoint_every: int = 5
    device: str = "cpu"
    seed: int = 0


# ----------------------------------------------------------------------------- dataset

class _ARView(TorchDataset):
    def __init__(self, ds: Dataset, tok: Tokenizer):
        self.ds = ds
        self.tok = tok

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, pset = self.ds[i]
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        seq = torch.tensor(self.tok.encode(pset), dtype=torch.long)
        return x, seq


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    seqs = [b[1] for b in batch]
    max_len = max(s.numel() for s in seqs)
    targets = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, :s.numel()] = s
    return imgs, targets


# ----------------------------------------------------------------------------- train

def train_autoregressive(cfg: ARTrainConfig, train_dir: Path, test_dir: Path,
                         out_dir: Path, log=print,
                         init_from: Path | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(cfg.canvas_side, cfg.canvas_side)
    device = torch.device(cfg.device)

    if not (train_dir / "manifest.json").exists():
        log(f"generating {cfg.train_samples} train samples -> {train_dir}")
        write_dataset(train_dir, cfg.train_samples, seed=cfg.train_seed, canvas=canvas)
    if not (test_dir / "manifest.json").exists():
        log(f"generating {cfg.test_samples} test samples -> {test_dir}")
        write_dataset(test_dir, cfg.test_samples, seed=cfg.test_seed, canvas=canvas,
                      randomize=False)

    train_ds = Dataset(train_dir)
    test_ds = Dataset(test_dir)
    tok = Tokenizer(canvas_side=cfg.canvas_side)
    loader = DataLoader(_ARView(train_ds, tok), batch_size=cfg.batch_size,
                        shuffle=True, num_workers=0, collate_fn=_collate)

    model = AutoregressiveModel(
        canvas_side=cfg.canvas_side, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_decoder_layers=cfg.n_decoder_layers,
        max_seq_len=cfg.max_seq_len,
    ).to(device)

    if init_from is not None:
        state = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        log(f"warm-started from {init_from}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = teacher_forced_loss(model, images, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_batches += 1
        mean_loss = running / n_batches
        elapsed = time.time() - t0
        log(f"epoch {epoch:02d}/{cfg.epochs}  loss={mean_loss:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "loss": mean_loss, "elapsed_s": elapsed})

        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "epoch": epoch, "arch": "autoregressive"},
                       out_dir / "model.pt")

    # final eval: F1 via greedy-sample predictor through the standard harness
    model.eval()
    predictor = AutoregressivePredictor(model, canvas,
                                        max_tokens=cfg.max_seq_len, device=cfg.device)
    eval_report = run_predictor(predictor, test_ds, canvas)
    log(f"\n=== Test set ({cfg.test_samples} samples, {cfg.canvas_side}x{cfg.canvas_side}) ===")
    for k in ("mean_f1", "mean_precision", "mean_recall", "mean_render_iou"):
        log(f"  {k:22s} {eval_report[k]:.3f}")

    (out_dir / "history.json").write_text(json.dumps(
        {"history": history, "eval": _strip(eval_report)}, indent=2, default=str))
    return {"history": history, "eval": eval_report}


def _strip(report: dict) -> dict:
    return {k: v for k, v in report.items() if k != "per_sample"}


# ----------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas-side", type=int, default=64)
    ap.add_argument("--train-dir", default="data/train64_box")
    ap.add_argument("--test-dir", default="data/test64_box")
    ap.add_argument("--out-dir", default="checkpoints/ar_box64")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-samples", type=int, default=4000)
    ap.add_argument("--test-samples", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-decoder-layers", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--device", default="cpu", help='"cpu" or "cuda"')
    args = ap.parse_args()

    cfg = ARTrainConfig(
        canvas_side=args.canvas_side, epochs=args.epochs,
        train_samples=args.train_samples, test_samples=args.test_samples,
        lr=args.lr, batch_size=args.batch_size, d_model=args.d_model,
        n_decoder_layers=args.n_decoder_layers, max_seq_len=args.max_seq_len,
        device=args.device,
    )
    init_from = Path(args.init_from) if args.init_from else None
    train_autoregressive(cfg, Path(args.train_dir), Path(args.test_dir),
                         Path(args.out_dir), init_from=init_from)


if __name__ == "__main__":
    main()
