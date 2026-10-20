"""Train the original Lorentz-model Hyperbolic GNN on the Elliptic graph.

Example:
  python train.py --data-dir data/elliptic --epochs 30 --patience 6 --output-dir artifacts

Label supervision is chronological (timesteps 1-34 train, 35-41 validation,
42-49 test), while all graph edges are available for message passing. This is a
transductive research evaluation, not a strict future-isolated graph split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

from src.models.detector import ForensicDetector
from src.utils.real_data_loader import load_elliptic_data


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


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
    parser.add_argument("--data-dir", default="data/elliptic", help="Elliptic folder or parent folder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but no CUDA device is available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Training original Lorentz-model Hyperbolic GNN on {device}")

    data = load_elliptic_data(args.data_dir)
    times = data.timestep.cpu().numpy()
    labels_np = data.y.cpu().numpy()
    train_np = (times >= 1) & (times <= 34) & (labels_np >= 0)
    val_np = (times >= 35) & (times <= 41) & (labels_np >= 0)
    test_np = (times >= 42) & (times <= 49) & (labels_np >= 0)
    if not train_np.any() or not val_np.any() or not test_np.any():
        raise ValueError("Train/validation/test masks are empty; check dataset timestep and labels")

    data = data.to(device)
    train_mask = torch.as_tensor(train_np, dtype=torch.bool, device=device)
    val_mask = torch.as_tensor(val_np, dtype=torch.bool, device=device)
    test_mask = torch.as_tensor(test_np, dtype=torch.bool, device=device)
    y_val = (labels_np[val_np] == 0).astype(np.int64)
    y_test = (labels_np[test_np] == 0).astype(np.int64)

    model = ForensicDetector(in_dim=int(data.x.shape[1]), hidden_dim=args.hidden_dim).to(device)
    counts = torch.bincount(data.y[train_mask], minlength=2).float().clamp_min(1.0)
    class_weights = counts.sum() / (2.0 * counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "forensic_detector.pt"
    history = []
    best_val_ap = -float("inf")
    best_threshold, best_val_f1 = 0.5, 0.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=class_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            _, validation_logits = model(data.x, data.edge_index)
            scores_all = torch.softmax(validation_logits, dim=1)[:, 0].cpu().numpy()
        val_scores = scores_all[val_np]
        threshold, val_f1 = choose_threshold(y_val, val_scores)
        val_ap = float(average_precision_score(y_val, val_scores))
        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "pr_auc_illicit": val_ap,
            "validation_threshold": float(threshold),
            "validation_f1_at_threshold": float(val_f1),
            "n_evaluated": int(len(y_val)),
        }
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={loss.item():.5f} | validation PR-AUC={val_ap:.4f} | F1={val_f1:.4f} at threshold {threshold:.6f}")

        if val_ap > best_val_ap:
            best_val_ap, best_threshold, best_val_f1, stale = val_ap, threshold, val_f1, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_dim": int(data.x.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                "architecture": "original_lorentz_hyperbolic_gat",
                "validation_threshold": float(best_threshold),
                "best_validation_f1_at_threshold": float(best_val_f1),
                "seed": int(args.seed),
                "evaluation_note": "All graph edges used for message passing; chronological label split is transductive.",
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: validation PR-AUC did not improve for {args.patience} epochs")
                break

    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        _, logits = model(data.x, data.edge_index)
        scores = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()

    val_metrics = metric_dict(y_val, scores[val_np], best_threshold)
    test_metrics = metric_dict(y_test, scores[test_np], best_threshold)
    test_half = metric_dict(y_test, scores[test_np], 0.5)
    metrics = {
        "model_name": "Original Lorentz-model Hyperbolic GAT",
        "architecture": "original_lorentz_hyperbolic_gat",
        "threshold": float(best_threshold),
        "threshold_selection": "Maximize illicit-class F1 on validation timesteps 35-41",
        "best_validation_pr_auc_illicit": float(best_val_ap),
        "best_validation_f1_at_threshold": float(best_val_f1),
        "validation_metrics": val_metrics,
        "test_timestep_range": [42, 49],
        "test_metrics_at_validation_threshold": test_metrics,
        "test_metrics_at_0_5": test_half,
        "n_nodes": int(data.num_nodes),
        "n_edges": int(data.edge_index.shape[1]),
        "epochs_completed": len(history),
        "evaluation_note": "All graph edges are available for message passing. This is transductive chronological-label evaluation, not strict future-isolated testing. Scores are ranking signals, not calibrated probabilities.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nTraining complete; checkpoint and metrics saved:")
    print(json.dumps(metrics, indent=2))
    print(f"Checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
