"""Tests for the model-as-predictor wrapper (Unit 5)."""

import numpy as np
import torch

from lines.datagen.sampler2d import Canvas
from lines.models.set_predictor import SetPredictor, required_feature_size
from lines.train.predictor import ModelPredictor


def test_untrained_predictor_returns_empty_set_for_blank_image():
    torch.manual_seed(0)
    canvas = Canvas(64, 64)
    model = SetPredictor(n_queries=8, d_model=64, n_heads=4, n_decoder_layers=2,
                         feature_size=required_feature_size(64))
    pred = ModelPredictor(model, canvas)
    blank = np.full((64, 64), 255, dtype=np.uint8)
    out = pred(blank)
    # head_type init biases toward "none" -> should predict no primitives
    assert out.primitives == []


def test_predictor_returns_primitive_set_with_valid_primitives():
    canvas = Canvas(64, 64)
    model = SetPredictor(n_queries=8, d_model=64, n_heads=4, n_decoder_layers=2,
                         feature_size=required_feature_size(64))
    # force at least one query to predict a line by overriding heads
    with torch.no_grad():
        model.head_type.bias.zero_()
        model.head_type.bias[0] = 10.0   # TYPE_LINE wins for every query
    pred = ModelPredictor(model, canvas)
    img = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    out = pred(img)
    assert all(p.is_valid() for p in out.primitives)
