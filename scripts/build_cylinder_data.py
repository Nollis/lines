"""Build train/test datasets of projected 3D cylinder wireframes (Stage 2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lines.datagen.cylinder_scene import sample_cylinder_scene
from lines.datagen.randomize import default_render_params, sample_render_params
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas


def build(out_dir: str, n: int, seed0: int, canvas_side: int, randomize: bool):
    canvas = Canvas(canvas_side, canvas_side)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n):
        pset = sample_cylinder_scene(seed0 + i, canvas)
        rp = sample_render_params(np.random.default_rng([seed0 + i, 7])) if randomize \
            else default_render_params()
        img = render_primitives(pset, canvas_side, canvas_side,
                                line_width=rp.line_width, supersample=rp.supersample)
        rel = f"images/{i:06d}.png"
        Image.fromarray(img, "L").save(out / rel)
        entries.append({"id": i, "seed": seed0 + i, "image": rel,
                        "line_width": rp.line_width, "supersample": rp.supersample,
                        "primitives": pset.to_dict()["primitives"]})
    (out / "manifest.json").write_text(json.dumps(
        {"canvas": {"width": canvas_side, "height": canvas_side}, "samples": entries}, indent=2))
    print(f"wrote {out}: {n} cylinder scenes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--canvas-side", type=int, default=64)
    ap.add_argument("--no-randomize", action="store_true")
    args = ap.parse_args()
    build(args.out, args.n, args.seed0, args.canvas_side, not args.no_randomize)


if __name__ == "__main__":
    main()
