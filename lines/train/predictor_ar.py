"""Inference wrapper for the autoregressive model: image -> PrimitiveSet.

Greedy-decodes a token sequence with :func:`greedy_sample`, then parses it
through the tokenizer. Honors the harness contract (callable with a numpy
``HxW`` uint8 image) so existing eval scripts work unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from lines.datagen.sampler2d import Canvas
from lines.models.autoregressive import AutoregressiveModel, greedy_sample
from lines.models.seq_tokenizer import Tokenizer
from lines.primitives import PrimitiveSet


class AutoregressivePredictor:
    def __init__(self, model: AutoregressiveModel, canvas: Canvas,
                 max_tokens: int = 128, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.canvas = canvas
        self.tok = Tokenizer(canvas_side=canvas.width)
        self.max_tokens = max_tokens
        self.device = device

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> PrimitiveSet:
        x = torch.from_numpy(image.astype(np.float32) / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)
        tokens = greedy_sample(self.model, x, max_len=self.max_tokens)
        try:
            return self.tok.decode(tokens)
        except Exception:
            return PrimitiveSet([])
