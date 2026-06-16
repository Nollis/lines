"""Adapt a trained :class:`SetPredictor` to the predictor protocol the eval
harness expects: ``image (numpy HxW uint8) -> PrimitiveSet``.

A "none" probability threshold filters out empty queries; the rest are decoded
into primitives in the requested canvas coordinates.
"""

from __future__ import annotations

import numpy as np
import torch

from lines.datagen.sampler2d import Canvas
from lines.models.encoding import N_TYPES, TYPE_NONE, decode_set
from lines.primitives import PrimitiveSet


class ModelPredictor:
    def __init__(self, model, canvas: Canvas, none_prob_threshold: float = 0.5,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.canvas = canvas
        self.threshold = none_prob_threshold
        self.device = device

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> PrimitiveSet:
        x = torch.from_numpy(image.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)             # (1, 1, H, W)
        logits, params = self.model(x)                              # (1, N, K), (1, N, P)
        probs = logits.softmax(dim=-1)[0]                           # (N, K)
        types = probs.argmax(dim=-1)                                # (N,)
        none_p = probs[:, TYPE_NONE]
        keep = (types != TYPE_NONE) & (none_p < self.threshold)
        if not keep.any():
            return PrimitiveSet([])
        kept_types = types[keep].cpu().numpy()
        kept_params = params[0][keep].cpu().numpy()
        return decode_set(kept_types, kept_params,
                          self.canvas.width, self.canvas.height)
