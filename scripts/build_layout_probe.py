"""Build a structured-layout probe split (technical or held-out families).

Renders generated layouts through both the training renderer (content shift
only) and OpenCV (content + rasterizer shift), writing two Dataset-compatible
splits that share ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from lines.datagen.heldout_layout import sample_heldout_set
from lines.datagen.heldout2_layout import sample_heldout2_set
from lines.datagen.probe_render import render_cv2
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas
from lines.datagen.technical_layout import sample_technical_set

_GENERATORS = {
    "technical": sample_technical_set,
    "heldout": sample_heldout_set,
    "heldout2": sample_heldout2_set,
}
_LINE_WIDTH = 2.0


def build(out_root: str, source: str, n: int, seed0: int, canvas_side: int):
    gen = _GENERATORS[source]
    canvas = Canvas(canvas_side, canvas_side)
    ours_dir, cv2_dir = Path(f"{out_root}_ours"), Path(f"{out_root}_cv2")
    for d in (ours_dir, cv2_dir):
        (d / "images").mkdir(parents=True, exist_ok=True)

    ours_entries, cv2_entries = [], []
    for i in range(n):
        pset = gen(seed0 + i, canvas)
        prim_dicts = pset.to_dict()["primitives"]
        rel = f"images/{i:06d}.png"
        Image.fromarray(render_primitives(pset, canvas_side, canvas_side,
                                          line_width=_LINE_WIDTH), "L").save(ours_dir / rel)
        Image.fromarray(render_cv2(pset, canvas_side, canvas_side,
                                   line_width=int(_LINE_WIDTH)), "L").save(cv2_dir / rel)
        entry = {"id": i, "seed": seed0 + i, "image": rel, "line_width": _LINE_WIDTH,
                 "supersample": 4, "primitives": prim_dicts}
        ours_entries.append(dict(entry))
        cv2_entries.append(dict(entry))

    meta = {"width": canvas_side, "height": canvas_side}
    (ours_dir / "manifest.json").write_text(json.dumps({"canvas": meta, "samples": ours_entries}, indent=2))
    (cv2_dir / "manifest.json").write_text(json.dumps({"canvas": meta, "samples": cv2_entries}, indent=2))
    print(f"wrote {ours_dir} and {cv2_dir} ({n} {source} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(_GENERATORS), required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--canvas-side", type=int, default=128)
    args = ap.parse_args()
    build(args.out_root, args.source, args.n, args.seed0, args.canvas_side)


if __name__ == "__main__":
    main()
