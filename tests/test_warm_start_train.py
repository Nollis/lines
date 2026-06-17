"""Smoke test: warm-start fine-tuning at a new resolution runs end-to-end."""

from pathlib import Path

import torch

from lines.datagen.dataset import write_dataset
from lines.datagen.sampler2d import Canvas
from lines.train.train import TrainConfig, train


def test_warm_start_64_to_128_runs(tmp_path: Path):
    # 1. train a tiny 64px model
    cfg64 = TrainConfig(canvas_side=64, train_samples=8, test_samples=4, epochs=1,
                        batch_size=4, n_queries=8, d_model=64, n_decoder_layers=2,
                        render_weight=0.0)
    train(cfg64, tmp_path / "tr64", tmp_path / "te64", tmp_path / "m64",
          log=lambda *a, **k: None)
    src = tmp_path / "m64" / "model.pt"
    assert src.exists()

    # 2. warm-start fine-tune it at 128px
    cfg128 = TrainConfig(canvas_side=128, train_samples=8, test_samples=4, epochs=1,
                         batch_size=4, render_weight=0.0)
    result = train(cfg128, tmp_path / "tr128", tmp_path / "te128", tmp_path / "m128",
                   log=lambda *a, **k: None, init_from=src)
    assert (tmp_path / "m128" / "model.pt").exists()
    assert 0.0 <= result["eval"]["mean_score"] <= 1.0

    # 3. the warm-started checkpoint inherits the source architecture (d_model=64)
    ck = torch.load(tmp_path / "m128" / "model.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["d_model"] == 64
    assert ck["cfg"]["canvas_side"] == 128
