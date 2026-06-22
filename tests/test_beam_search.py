"""Tests for beam-search sampling on the autoregressive model.

Greedy is myopic: once it picks a bad token, the rest cascades. Beam search
explores multiple paths and reduces that catastrophe rate. Contract pinned
here so we can compare predictors apples-to-apples.
"""

import torch

from lines.models.autoregressive import AutoregressiveModel, beam_sample, greedy_sample
from lines.models.seq_tokenizer import EOS, SOS

torch.manual_seed(0)


def _tiny():
    return AutoregressiveModel(canvas_side=64, d_model=64, n_heads=4, n_decoder_layers=2)


def test_beam_sample_returns_valid_sequence():
    m = _tiny()
    image = torch.zeros(1, 1, 64, 64)
    tokens = beam_sample(m, image, max_len=20, beam_size=3)
    assert tokens[0] == SOS
    assert tokens[-1] == EOS or len(tokens) == 20


def test_beam_size_one_matches_greedy():
    m = _tiny()
    image = torch.zeros(1, 1, 64, 64)
    g = greedy_sample(m, image, max_len=20)
    b = beam_sample(m, image, max_len=20, beam_size=1)
    assert g == b


def test_beam_sample_is_deterministic_given_seed():
    m = _tiny()
    image = torch.zeros(1, 1, 64, 64)
    a = beam_sample(m, image, max_len=20, beam_size=4)
    b = beam_sample(m, image, max_len=20, beam_size=4)
    assert a == b


def test_beam_sample_only_emits_valid_token_ids():
    m = _tiny()
    image = torch.zeros(1, 1, 64, 64)
    tokens = beam_sample(m, image, max_len=20, beam_size=3)
    from lines.models.seq_tokenizer import vocab_size
    V = vocab_size()
    assert all(0 <= t < V for t in tokens)
