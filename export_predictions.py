"""Export transaction scores and 2-D embeddings from a Lorentz GNN checkpoint.

This script runs inference only; it does not train or retrain the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.models.detector import ForensicDetector
from src.utils.real_data_loader import load_elliptic_data


def metric_dict(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    result = {
        "threshold": float(threshold),
        "n_evaluated": int(len(y_true)),
        "illicit_count": int(y_true.sum()),
        "predicted_illicit_count": int(pred.sum()),
        "precision_illicit": float(precision_score(y_true, pred, zero_division=0)),
        "recall_illicit": float(recall_score(y_true, pred, zero_division=0)),
        "f1_illicit": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc_illicit": float(average_precision_score(y_true, scores)),
    }
    if np.unique(y_true).size == 2:
        result["roc_auc_illicit"] = float(roc_auc_score(y_true, scores))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Elliptic directory or parent directory")
    parser.add_argument("--checkpoint", required=True, help="Lorentz model forensic_detector.pt")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but no CUDA device is available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "original_lorentz_hyperbolic_gat":
        raise ValueError(
            "This checkpoint is not marked as the original Lorentz GNN. "
            "Train the original model with this repository's train.py first; do not use the older Poincare checkpoint."
        )
    data = load_elliptic_data(args.data_dir).to(device)
    model = ForensicDetector(int(checkpoint["in_dim"]), int(checkpoint["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    with torch.inference_mode():
        embedding, logits = model(data.x, data.edge_index)
        if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
            raise FloatingPointError("Model produced NaN/Inf values; no export was written")
        scores = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
        embedding_np = embedding.cpu().numpy()

    labels = data.y.cpu().numpy()
    timesteps = data.timestep.cpu().numpy()
    tx_ids = data.tx_id.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    degree = np.bincount(np.concatenate((edge_index[0], edge_index[1])), minlength=len(labels))
    known_actual = np.where(labels == 0, "Illicit", np.where(labels == 1, "Licit", "Unknown"))
    threshold = float(checkpoint.get("validation_threshold", 0.5))
    q90, q99 = np.quantile(scores, [0.90, 0.99])
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame({
        "tx_id": tx_ids.astype(str),
        "timestep": timesteps,
        "actual_label": known_actual,
        "illicit_score": scores.astype(np.float64),
        "predicted_label": np.where(scores >= threshold, "Illicit", "Licit"),
        "risk_band": np.where(scores >= q99, "Top 1% score", np.where(scores >= q90, "Top 10% score", "Remaining")),
        "degree": degree.astype(np.int64),
        "embedding_x": embedding_np[:, 0],
        "embedding_y": embedding_np[:, 1],
    })
    frame.to_csv(output / "predictions.csv", index=False)

    val_mask = (timesteps >= 35) & (timesteps <= 41) & (labels >= 0)
    test_mask = (timesteps >= 42) & (timesteps <= 49) & (labels >= 0)
    y_val = (labels[val_mask] == 0).astype(np.int64)
    y_test = (labels[test_mask] == 0).astype(np.int64)
    metrics = {
        "model_name": "Original Lorentz-model Hyperbolic GAT",
        "architecture": "original_lorentz_hyperbolic_gat",
        "threshold": threshold,
        "threshold_selection": "Selected on labelled validation timesteps 35-41 during training",
        "validation_metrics": metric_dict(y_val, scores[val_mask], threshold),
        "test_timestep_range": [42, 49],
        "test_metrics_at_validation_threshold": metric_dict(y_test, scores[test_mask], threshold),
        "test_metrics_at_0_5": metric_dict(y_test, scores[test_mask], 0.5),
        "n_nodes_exported": int(len(labels)),
        "n_edges_used": int(edge_index.shape[1]),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "evaluation_note": "All graph edges used for message passing. This is transductive chronological-label evaluation, not strict future-isolated testing. Scores are ranking signals, not calibrated probabilities.",
    }
    (output / "deployment_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Exported real predictions: {output / 'predictions.csv'}")
    print(f"Exported real evaluation metrics: {output / 'deployment_metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
