"""Build sim-to-real probe v2: structured technical-drawing content.

Generates N layouts from the independent technical-layout generator and renders
each through BOTH the training renderer (Pillow) and the independent OpenCV
renderer, writing two Dataset-compatible splits that share ground truth:

* ``*_ours``  -> content shift only (familiar pixels, unfamiliar arrangements)
* ``*_cv2``   -> content shift + rasterizer shift

Comparing the two against the random-content test set decomposes the gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lines.datagen.probe_render import render_cv2
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas
from lines.datagen.technical_layout import sample_technical_set

_LINE_WIDTH = 2.0


def build(out_root: str, n_samples: int, seed0: int, canvas_side: int):
    canvas = Canvas(canvas_side, canvas_side)
    ours_dir = Path(f"{out_root}_ours")
    cv2_dir = Path(f"{out_root}_cv2")
    (ours_dir / "images").mkdir(parents=True, exist_ok=True)
    (cv2_dir / "images").mkdir(parents=True, exist_ok=True)

    ours_entries, cv2_entries = [], []
    for i in range(n_samples):
        pset = sample_technical_set(seed0 + i, canvas)
        prim_dicts = pset.to_dict()["primitives"]
        rel = f"images/{i:06d}.png"

        img_ours = render_primitives(pset, canvas_side, canvas_side, line_width=_LINE_WIDTH)
        Image.fromarray(img_ours, "L").save(ours_dir / rel)
        img_cv2 = render_cv2(pset, canvas_side, canvas_side, line_width=int(_LINE_WIDTH))
        Image.fromarray(img_cv2, "L").save(cv2_dir / rel)

        entry = {"id": i, "seed": seed0 + i, "image": rel,
                 "line_width": _LINE_WIDTH, "supersample": 4, "primitives": prim_dicts}
        ours_entries.append(dict(entry))
        cv2_entries.append(dict(entry))

    meta = {"width": canvas_side, "height": canvas_side}
    (ours_dir / "manifest.json").write_text(json.dumps({"canvas": meta, "samples": ours_entries}, indent=2))
    (cv2_dir / "manifest.json").write_text(json.dumps({"canvas": meta, "samples": cv2_entries}, indent=2))
    print(f"wrote {ours_dir} and {cv2_dir} ({n_samples} samples each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/probe_tech128")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=700_000)
    ap.add_argument("--canvas-side", type=int, default=128)
    args = ap.parse_args()
    build(args.out_root, args.n, args.seed0, args.canvas_side)


if __name__ == "__main__":
    main()
