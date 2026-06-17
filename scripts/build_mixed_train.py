"""Build an enriched training set: random shape-soup + structured layouts.

The model previously trained only on random independent primitives and failed
on structured arrangements (concentric circles, parallel clusters). Since the
data is generated, the fix is to put those arrangements INTO training. This
builder mixes the random sampler with the technical-layout families, rendered
through the training renderer with domain randomization.

Seeds are disjoint from every probe split so evaluation stays honest:
random training uses 0.., technical training uses 500_000.., while the technical
probe uses 700_000.. and the held-out probe 800_000...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lines.datagen.randomize import sample_render_params
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas, sample_primitive_set
from lines.datagen.technical_layout import sample_technical_set


def build(out_dir: str, n: int, canvas_side: int, tech_fraction: float,
          rand_seed0: int = 0, tech_seed0: int = 500_000):
    canvas = Canvas(canvas_side, canvas_side)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    choice_rng = np.random.default_rng(12_345)   # deterministic source selection

    entries = []
    n_tech = 0
    for i in range(n):
        if choice_rng.random() < tech_fraction:
            pset = sample_technical_set(tech_seed0 + i, canvas)
            source = "technical"
            n_tech += 1
        else:
            pset = sample_primitive_set(rand_seed0 + i, canvas=canvas)
            source = "random"
        rp = sample_render_params(np.random.default_rng([i, 7]))
        img = render_primitives(pset, canvas_side, canvas_side,
                                line_width=rp.line_width, supersample=rp.supersample)
        rel = f"images/{i:06d}.png"
        Image.fromarray(img, "L").save(out / rel)
        entries.append({"id": i, "image": rel, "line_width": rp.line_width,
                        "supersample": rp.supersample, "source": source,
                        "primitives": pset.to_dict()["primitives"]})

    (out / "manifest.json").write_text(json.dumps(
        {"canvas": {"width": canvas_side, "height": canvas_side}, "samples": entries}, indent=2))
    print(f"wrote {out}: {n} samples, {n_tech} technical ({n_tech / n:.0%}), "
          f"{n - n_tech} random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train128_mixed")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--canvas-side", type=int, default=128)
    ap.add_argument("--tech-fraction", type=float, default=0.4)
    args = ap.parse_args()
    build(args.out, args.n, args.canvas_side, args.tech_fraction)


if __name__ == "__main__":
    main()
