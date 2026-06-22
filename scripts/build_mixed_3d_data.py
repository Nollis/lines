"""Build a mixed train/test dataset combining 3D boxes and cylinders (E7).

Sampling: deterministic ``random.Random(choice_seed).random() < fraction_cyl``
per sample picks cylinder vs box. Each generator gets its own seed sub-stream
(box and cylinder seed ranges stay disjoint so probes never see this content).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from lines.datagen.box_scene import sample_box_scene
from lines.datagen.cylinder_scene import sample_cylinder_scene
from lines.datagen.randomize import default_render_params, sample_render_params
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas


def build(out_dir: str, n: int, seed0: int, canvas_side: int,
          fraction_cyl: float, randomize: bool):
    canvas = Canvas(canvas_side, canvas_side)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # disjoint seed sub-streams: boxes use seed0.., cylinders use seed0+500k..
    # (test splits live further out, so no risk of overlap)
    choice_rng = random.Random(seed0 + 12_345)
    entries = []
    n_cyl = 0
    for i in range(n):
        is_cyl = choice_rng.random() < fraction_cyl
        if is_cyl:
            pset = sample_cylinder_scene(seed0 + 500_000 + i, canvas)
            kind = "cylinder"
            n_cyl += 1
        else:
            pset = sample_box_scene(seed0 + i, canvas)
            kind = "box"
        rp = (sample_render_params(np.random.default_rng([seed0 + i, 7]))
              if randomize else default_render_params())
        img = render_primitives(pset, canvas_side, canvas_side,
                                line_width=rp.line_width, supersample=rp.supersample)
        rel = f"images/{i:06d}.png"
        Image.fromarray(img, "L").save(out / rel)
        entries.append({"id": i, "seed": seed0 + i, "image": rel,
                        "line_width": rp.line_width, "supersample": rp.supersample,
                        "kind": kind,
                        "primitives": pset.to_dict()["primitives"]})

    (out / "manifest.json").write_text(json.dumps(
        {"canvas": {"width": canvas_side, "height": canvas_side},
         "samples": entries}, indent=2))
    print(f"wrote {out}: {n} scenes "
          f"({n_cyl} cyl + {n - n_cyl} box, {n_cyl / n:.0%} cylinder)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--canvas-side", type=int, default=64)
    ap.add_argument("--fraction-cyl", type=float, default=0.5,
                    help="fraction of samples that are cylinders (rest are boxes)")
    ap.add_argument("--no-randomize", action="store_true")
    args = ap.parse_args()
    build(args.out, args.n, args.seed0, args.canvas_side,
          args.fraction_cyl, not args.no_randomize)


if __name__ == "__main__":
    main()
