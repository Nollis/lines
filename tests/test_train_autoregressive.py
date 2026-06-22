"""End-to-end smoke test for the autoregressive training entry point (A3-P2).

Trains for 1 epoch on a tiny generated split to verify: build the dataset,
tokenize+teacher-force, save a checkpoint that reloads, and run greedy-sample
eval through the harness producing valid metrics.
"""

from pathlib import Path

import torch

from lines.train.train_autoregressive import ARTrainConfig, train_autoregressive


def test_one_epoch_runs_end_to_end(tmp_path: Path):
    cfg = ARTrainConfig(
        canvas_side=64, train_samples=8, test_samples=4, epochs=1,
        batch_size=4, d_model=64, n_decoder_layers=2,
    )
    result = train_autoregressive(
        cfg, tmp_path / "train", tmp_path / "test", tmp_path / "out",
        log=lambda *a, **k: None,
    )
    ck = tmp_path / "out" / "model.pt"
    assert ck.exists()
    assert len(result["history"]) == 1
    assert result["history"][0]["loss"] > 0    # finite, computed

    # checkpoint reloads cleanly
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    assert "model" in payload and "cfg" in payload

    # eval ran and produced metrics
    rep = result["eval"]
    assert "mean_f1" in rep
    assert 0.0 <= rep["mean_f1"] <= 1.0
    assert rep["n"] == 4


def test_train_data_is_generated_if_missing(tmp_path: Path):
    cfg = ARTrainConfig(canvas_side=64, train_samples=4, test_samples=2,
                        epochs=1, batch_size=2, d_model=64, n_decoder_layers=2)
    train_autoregressive(
        cfg, tmp_path / "train", tmp_path / "test", tmp_path / "out",
        log=lambda *a, **k: None,
    )
    assert (tmp_path / "train" / "manifest.json").exists()
    assert (tmp_path / "test" / "manifest.json").exists()
