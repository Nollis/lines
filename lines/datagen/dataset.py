"""Write and reload synthetic (image, primitive-set) datasets.

A dataset is a directory containing ``manifest.json`` and an ``images/``
folder. The manifest is the free, perfectly-aligned ground truth: each entry
stores the exact primitives plus the render parameters used, so any image is
reproducible from its manifest. The loader is framework-agnostic (returns a
numpy image + :class:`PrimitiveSet`); a torch ``Dataset`` wrapper is added with
the training unit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from lines.datagen.randomize import default_render_params, sample_render_params
from lines.datagen.sampler2d import Canvas, sample_primitive_set
from lines.datagen.render import render_primitives
from lines.primitives import PrimitiveSet


def write_dataset(
    out_dir,
    n_samples: int,
    seed: int = 0,
    canvas: Canvas = Canvas(256, 256),
    min_n: int = 1,
    max_n: int = 5,
    max_arc_sweep_deg: float = 270.0,
    types=("line", "arc", "circle"),
    randomize: bool = True,
) -> Path:
    """Generate ``n_samples`` pairs under ``out_dir`` and return the manifest path."""
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(n_samples):
        sample_seed = seed + i
        pset = sample_primitive_set(
            sample_seed, canvas=canvas, min_n=min_n, max_n=max_n,
            max_arc_sweep_deg=max_arc_sweep_deg, types=types,
        )
        if randomize:
            rp = sample_render_params(np.random.default_rng([sample_seed, 1]))
        else:
            rp = default_render_params()

        img = render_primitives(
            pset, canvas.width, canvas.height,
            line_width=rp.line_width, supersample=rp.supersample,
        )
        rel_path = f"images/{i:06d}.png"
        Image.fromarray(img, mode="L").save(out_dir / rel_path)

        entries.append({
            "id": i,
            "seed": sample_seed,
            "image": rel_path,
            "line_width": rp.line_width,
            "supersample": rp.supersample,
            "primitives": pset.to_dict()["primitives"],
        })

    manifest = {
        "canvas": {"width": canvas.width, "height": canvas.height},
        "samples": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


class Dataset:
    """Reload a dataset written by :func:`write_dataset`."""

    def __init__(self, root, max_samples: int | None = None, cache_images: bool = True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.samples = self.manifest["samples"]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.canvas = self.manifest["canvas"]

        self.cache_images = cache_images
        self.cached_images = {}
        if self.cache_images:
            for idx, entry in enumerate(self.samples):
                img = np.asarray(Image.open(self.root / entry["image"]).convert("L"), dtype=np.uint8)
                self.cached_images[idx] = img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        if self.cache_images and i in self.cached_images:
            img = self.cached_images[i]
        else:
            entry = self.samples[i]
            img = np.asarray(Image.open(self.root / entry["image"]).convert("L"), dtype=np.uint8)
        pset = PrimitiveSet.from_dict({"primitives": self.samples[i]["primitives"]})
        return img, pset

    def line_width(self, i: int) -> float:
        return self.samples[i]["line_width"]

    def supersample(self, i: int) -> int:
        return self.samples[i]["supersample"]

