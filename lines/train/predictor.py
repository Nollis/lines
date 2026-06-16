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
    def __init__(self, model, canvas: Canvas, none_prob_threshold: float = 0.85,
                 non_none_prob_threshold: float = 0.15, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.canvas = canvas
        self.threshold = none_prob_threshold
        self.min_prob = non_none_prob_threshold
        self.device = device

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> PrimitiveSet:
        x = torch.from_numpy(image.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)             # (1, 1, H, W)
        logits, params = self.model(x)                              # (1, N, K), (1, N, P)
        probs = logits.softmax(dim=-1)[0]                           # (N, K)
        
        # Keep slots where the probability of NONE is below self.threshold,
        # AND at least one non-NONE class is somewhat confident (above self.min_prob)
        none_p = probs[:, TYPE_NONE]
        max_non_none_p = probs[:, :TYPE_NONE].max(dim=-1).values
        keep = (none_p < self.threshold) & (max_non_none_p > self.min_prob)
        
        if not keep.any():
            return PrimitiveSet([])
            
        # For the kept slots, assign the type with the highest probability among the non-NONE types
        non_none_probs = probs[keep, :TYPE_NONE]                    # (keep_count, 3)
        kept_types = non_none_probs.argmax(dim=-1).cpu().numpy()
        kept_params = params[0][keep].cpu().numpy()
        return decode_set(kept_types, kept_params,
                          self.canvas.width, self.canvas.height)
