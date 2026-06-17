"""Tests for cross-resolution warm-starting (64 -> 128 fine-tune setup)."""

import torch

from lines.models.set_predictor import (
    SetPredictor, warm_start_state_dict, required_feature_size,
)


def _model(feature_size, d_model=64):
    return SetPredictor(n_queries=8, d_model=d_model, n_heads=4,
                        n_decoder_layers=2, feature_size=feature_size,
                        encoder_type="residual")


def test_pos_embed_is_resized_to_target_token_count():
    src = _model(feature_size=4)                      # 16 tokens
    adapted = warm_start_state_dict(src.state_dict(), 4, 8, d_model=64)
    assert adapted["pos_embed"].shape == (64, 64)     # 8*8 tokens, d=64


def test_non_pos_embed_weights_are_copied_unchanged():
    src = _model(feature_size=4)
    sd = src.state_dict()
    adapted = warm_start_state_dict(sd, 4, 8, d_model=64)
    for key, tensor in sd.items():
        if key == "pos_embed":
            continue
        assert torch.equal(adapted[key], tensor), f"{key} was modified"


def test_adapted_state_dict_loads_into_target_model():
    src = _model(feature_size=4)
    adapted = warm_start_state_dict(src.state_dict(), 4, 8, d_model=64)
    dst = _model(feature_size=8)
    dst.load_state_dict(adapted)   # must not raise


def test_same_resolution_warm_start_is_identity():
    src = _model(feature_size=8)
    adapted = warm_start_state_dict(src.state_dict(), 8, 8, d_model=64)
    for key, tensor in src.state_dict().items():
        assert torch.equal(adapted[key], tensor)


def test_warm_started_model_runs_forward_at_new_resolution():
    src = _model(feature_size=required_feature_size(64))
    adapted = warm_start_state_dict(src.state_dict(),
                                    required_feature_size(64),
                                    required_feature_size(128), d_model=64)
    dst = _model(feature_size=required_feature_size(128))
    dst.load_state_dict(adapted)
    logits, params = dst(torch.zeros(1, 1, 128, 128))
    assert logits.shape == (1, 8, 4)
    assert params.shape == (1, 8, 5)
