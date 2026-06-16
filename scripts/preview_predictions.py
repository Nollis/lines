"""Render a side-by-side grid of (input, model prediction, ground truth)
for a handful of test samples, to make it easy to eyeball what the model
actually learned. Outputs ``data/preview/predictions.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lines.datagen.dataset import Dataset
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas
from lines.models.set_predictor import SetPredictor, required_feature_size
from lines.train.predictor import ModelPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-dir", default="data/test64")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--out", default="data/preview/predictions.png")
    ap.add_argument("--none-threshold", type=float, default=0.85)
    ap.add_argument("--refine-distance", type=float, default=6.0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    canvas = Canvas(cfg["canvas_side"], cfg["canvas_side"])
    model = SetPredictor(
        n_queries=cfg["n_queries"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_decoder_layers=cfg["n_decoder_layers"],
        feature_size=required_feature_size(cfg["canvas_side"]),
    )
    model.load_state_dict(ckpt["model"])
    predictor = ModelPredictor(model, canvas, none_prob_threshold=args.none_threshold,
                               refine_distance=args.refine_distance)

    ds = Dataset(args.test_dir)
    S = canvas.width
    pad = 6
    cell_h = S
    cell_w = S * 3 + pad * 2          # input | pred | gt
    rows = args.n
    sheet = np.full(((cell_h + pad) * rows + pad, cell_w + pad * 2, 3),
                    220, dtype=np.uint8)

    for k in range(rows):
        img, gt = ds[k]
        pred = predictor(img)
        gt_img = render_primitives(gt, S, S)
        pred_img = render_primitives(pred, S, S)
        y = pad + k * (cell_h + pad)

        triple = np.concatenate([
            _to_rgb(img),
            np.full((S, pad, 3), 220, np.uint8),
            _to_rgb(pred_img),
            np.full((S, pad, 3), 220, np.uint8),
            _to_rgb(gt_img),
        ], axis=1)
        sheet[y:y + cell_h, pad:pad + triple.shape[1]] = triple

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, "RGB").save(args.out)
    print(f"saved {args.out} -- columns: input | prediction | ground truth")


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray] * 3, axis=-1)


if __name__ == "__main__":
    main()
