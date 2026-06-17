"""Compat tests: legacy 'plain' encoder checkpoints must still load."""

import torch

from lines.models.set_predictor import (
    SetPredictor, detect_encoder_type, required_feature_size,
)


def test_detect_residual_from_state_dict():
    m = SetPredictor(d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=4, encoder_type="residual")
    assert detect_encoder_type(m.state_dict()) == "residual"


def test_detect_plain_from_state_dict():
    m = SetPredictor(d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=4, encoder_type="plain")
    assert detect_encoder_type(m.state_dict()) == "plain"


def test_plain_encoder_forward_shape():
    m = SetPredictor(n_queries=4, d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=required_feature_size(64), encoder_type="plain")
    logits, params = m(torch.zeros(1, 1, 64, 64))
    assert logits.shape == (1, 4, 4)   # (B, N, K)
    assert params.shape == (1, 4, 5)   # (B, N, P)


def test_plain_state_dict_round_trips():
    a = SetPredictor(d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=4, encoder_type="plain")
    b = SetPredictor(d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=4, encoder_type="plain")
    b.load_state_dict(a.state_dict())   # must not raise
