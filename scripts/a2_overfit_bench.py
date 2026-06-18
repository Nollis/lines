"""A2 bake-off: overfit-a-tiny-set capacity test for the autoregressive prototype.

For each content type (boxes / concentric circles) we:
  1. generate N small samples,
  2. teacher-force train the model on the *same* N samples for K steps,
  3. greedily decode each one and compute F1 against ground truth.

The point is *representation capacity*, not generalization -- can this
architecture even produce clean structured output for the relationships the
project has been bitten by? This is the scope doc's gate before A3 builds it
out at full scale.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from lines.datagen.box_scene import sample_box_scene
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas
from lines.datagen.technical_layout import sample_technical_set
from lines.eval.metrics import evaluate
from lines.models.autoregressive import (
    AutoregressiveModel, greedy_sample, teacher_forced_loss,
)
from lines.models.seq_tokenizer import EOS, Tokenizer

CANVAS = Canvas(64, 64)
TOK = Tokenizer(canvas_side=64)


def _render_sample(pset):
    return render_primitives(pset, 64, 64, line_width=2.0)


def _box_sampler(seed):
    return sample_box_scene(seed, CANVAS)


def _concentric_sampler(seed):
    # force concentric circles only -- the structural relationship that broke
    # set prediction in 2D and the vertex-graph alternative cannot model
    return sample_technical_set(seed, CANVAS, _force="concentric")


def overfit_and_score(sampler_name: str, sampler, n_samples: int,
                      steps: int, lr: float):
    torch.manual_seed(0)
    np.random.seed(0)
    psets, images, seqs = [], [], []
    for i in range(n_samples):
        ps = sampler(i)
        psets.append(ps)
        images.append(torch.from_numpy(_render_sample(ps).astype(np.float32) / 255.0).unsqueeze(0))
        seqs.append(TOK.encode(ps))

    max_len = max(len(s) for s in seqs)
    targets = torch.full((n_samples, max_len), EOS, dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    batch_img = torch.stack(images, dim=0)

    model = AutoregressiveModel(canvas_side=64, d_model=128,
                                n_heads=4, n_decoder_layers=3, max_seq_len=max_len + 4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    t0 = time.time()
    losses = []
    for step in range(steps):
        loss = teacher_forced_loss(model, batch_img, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            losses.append((step, loss.item()))
    elapsed = time.time() - t0

    # decode each training image and score against its own GT (capacity test)
    f1s, precs, recalls = [], [], []
    n_valid_decode = 0
    for img_t, gt in zip(images, psets):
        try:
            toks = greedy_sample(model, img_t.unsqueeze(0), max_len=max_len + 4)
            pred = TOK.decode(toks)
            n_valid_decode += 1
        except Exception:
            from lines.primitives import PrimitiveSet
            pred = PrimitiveSet([])
        m = evaluate(pred, gt, CANVAS)
        f1s.append(m["f1"]); precs.append(m["precision"]); recalls.append(m["recall"])

    print(f"\n=== {sampler_name} (N={n_samples}, {steps} steps, {elapsed:.0f}s) ===")
    print("  loss trajectory:", ", ".join(f"step{s}:{l:.3f}" for s, l in losses[:6]))
    if len(losses) > 6:
        print(f"  ... final step{losses[-1][0]}:{losses[-1][1]:.3f}")
    print(f"  decode succeeded: {n_valid_decode}/{n_samples}")
    print(f"  mean F1={np.mean(f1s):.3f}  precision={np.mean(precs):.3f}  recall={np.mean(recalls):.3f}")
    print(f"  perfect (F1=1.0): {sum(1 for f in f1s if f > 0.99)}/{n_samples}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--only", choices=["box", "concentric"], default=None)
    args = ap.parse_args()

    if args.only in (None, "box"):
        overfit_and_score("BOXES", _box_sampler, args.n, args.steps, args.lr)
    if args.only in (None, "concentric"):
        overfit_and_score("CONCENTRIC CIRCLES", _concentric_sampler,
                          args.n, args.steps, args.lr)


if __name__ == "__main__":
    main()
