"""Build a sim-to-real probe split from an existing test manifest.

Re-renders the SAME primitive sets (identical ground truth) through the
independent OpenCV rasterizer, optionally followed by JPEG compression. The
resulting folder is Dataset-compatible, so the eval harness scores it
unchanged. The score drop vs the original (Pillow-rendered) split is the
sim-to-real gap, decomposed by which shift was applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lines.datagen.probe_render import render_cv2, jpeg_roundtrip
from lines.primitives import PrimitiveSet


def build_probe(src_dir: Path, out_dir: Path, jpeg_quality: int | None = None) -> Path:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    manifest = json.loads((src_dir / "manifest.json").read_text())
    canvas = manifest["canvas"]
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    entries = []
    for entry in manifest["samples"]:
        pset = PrimitiveSet.from_dict({"primitives": entry["primitives"]})
        lw = max(1, int(round(entry.get("line_width", 2.0))))
        img = render_cv2(pset, canvas["width"], canvas["height"], line_width=lw)
        if jpeg_quality is not None:
            img = jpeg_roundtrip(img, jpeg_quality)
        rel = entry["image"]
        Image.fromarray(img, "L").save(out_dir / rel)
        new_entry = dict(entry)
        new_entry["image"] = rel
        entries.append(new_entry)

    (out_dir / "manifest.json").write_text(
        json.dumps({"canvas": canvas, "samples": entries}, indent=2))
    return out_dir / "manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/test128")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jpeg-quality", type=int, default=None)
    args = ap.parse_args()
    path = build_probe(Path(args.src), Path(args.out), args.jpeg_quality)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
