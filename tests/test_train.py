"""Smoke test for the training loop (Unit 5).

Trains for 1 epoch on 8 tiny samples to verify the loop runs end-to-end:
dataset write, training step, checkpoint save, eval through the harness.
Does NOT assert any quality target -- that's the empirical gate run via
the CLI on a real-sized configuration.
"""

from pathlib import Path

from lines.train.train import TrainConfig, train


def test_training_loop_runs_one_epoch_end_to_end(tmp_path: Path):
    cfg = TrainConfig(
        canvas_side=64, train_samples=8, test_samples=4, epochs=1,
        batch_size=4, n_queries=8, d_model=64, n_decoder_layers=2,
    )
    result = train(cfg, tmp_path / "train", tmp_path / "test", tmp_path / "out",
                   log=lambda *a, **kw: None)
    assert (tmp_path / "out" / "model.pt").exists()
    assert len(result["history"]) == 1
    rep = result["eval"]
    assert 0.0 <= rep["mean_score"] <= 1.0
    assert rep["n"] == 4
