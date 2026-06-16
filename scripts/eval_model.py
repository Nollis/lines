
import torch
import argparse
from pathlib import Path
from lines.models.set_predictor import SetPredictor, required_feature_size
from lines.train.predictor import ModelPredictor
from lines.datagen.dataset import Dataset
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor

def evaluate_checkpoint(ckpt_path: str, test_dir: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    canvas = Canvas(cfg["canvas_side"], cfg["canvas_side"])
    model = SetPredictor(
        n_queries=cfg["n_queries"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_decoder_layers=cfg["n_decoder_layers"],
        feature_size=required_feature_size(cfg["canvas_side"]),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    predictor = ModelPredictor(model, canvas)
    ds = Dataset(test_dir)
    
    report = run_predictor(predictor, ds, canvas)
    print(f"\n=== Evaluation Report: {ckpt_path} ===")
    for k in ("mean_score", "mean_render_iou", "mean_type_accuracy",
              "mean_geometric_error", "mean_coverage"):
        print(f"  {k:22s} {report[k]:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v1_120/model.pt")
    ap.add_argument("--test-dir", default="data/test64")
    args = ap.parse_args()
    evaluate_checkpoint(args.checkpoint, args.test_dir)
