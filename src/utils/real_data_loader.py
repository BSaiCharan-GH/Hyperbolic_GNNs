"""Load the Elliptic transaction graph for the Lorentz-model research pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

REQUIRED_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)


def resolve_dataset_dir(data_dir: str | Path) -> Path:
    """Accept the dataset directory itself or a parent containing the CSV files."""
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if all((root / name).is_file() for name in REQUIRED_FILES):
        return root

    matches = [
        p.parent
        for p in root.rglob(REQUIRED_FILES[0])
        if all((p.parent / name).is_file() for name in REQUIRED_FILES)
    ]
    # Preserve order while deduplicating.
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        print(f"Found Elliptic dataset under: {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            "Found multiple Elliptic dataset folders. Pass the intended one directly: "
            + ", ".join(str(p) for p in matches)
        )
    raise FileNotFoundError(
        f"Could not find all three Elliptic CSV files below {root}. Required: {REQUIRED_FILES}"
    )


def load_elliptic_data(data_dir: str | Path = "data/elliptic") -> Data:
    """Create a PyG Data object with 165 features and chronological metadata.

    Original Elliptic labels are mapped to 0=illicit, 1=licit, -1=unknown.
    Timestep is retained for split masks but is not included in model features.
    StandardScaler is fitted only on timesteps 1-34 to reduce preprocessing leakage.
    """
    root = resolve_dataset_dir(data_dir)
    feature_path, class_path, edge_path = (root / name for name in REQUIRED_FILES)
    print(f"Reading features: {feature_path}")

    # First CSV has no header: txId, timestep, then 165 numeric features.
    features_df = pd.read_csv(feature_path, header=None, low_memory=False)
    if features_df.shape[1] != 167:
        raise ValueError(
            "Expected 167 columns in elliptic_txs_features.csv (txId + timestep + 165 features); "
            f"got {features_df.shape[1]}"
        )
    tx_ids = features_df.iloc[:, 0].to_numpy()
    timesteps = features_df.iloc[:, 1].to_numpy(dtype=np.int64)
    raw_features = features_df.iloc[:, 2:].to_numpy(dtype=np.float32)
    del features_df

    if raw_features.shape[1] != 165:
        raise ValueError(f"Expected 165 model features, got {raw_features.shape[1]}")
    if not np.isfinite(raw_features).all():
        raise ValueError("Feature file contains NaN or infinite values")

    train_period = (timesteps >= 1) & (timesteps <= 34)
    if not train_period.any():
        raise ValueError("No training-period transactions found (timesteps 1-34)")
    scaler = StandardScaler().fit(raw_features[train_period])
    # Lorentz exp/log maps are numerically sensitive to large tangent norms.
    # Scaling the standardized vector keeps the original geometry pipeline stable.
    x = np.clip(scaler.transform(raw_features), -8.0, 8.0).astype(np.float32) / 10.0
    del raw_features

    # IDs are numeric in the released Elliptic CSVs; normalize to integers to make
    # CSV joins robust to pandas selecting a different numeric dtype.
    tx_ids_int = pd.to_numeric(pd.Series(tx_ids), errors="raise").astype(np.int64).to_numpy()
    id_to_index = {int(tx_id): idx for idx, tx_id in enumerate(tx_ids_int)}

    classes = pd.read_csv(class_path, dtype={"class": str})
    if not {"txId", "class"}.issubset(classes.columns):
        raise ValueError(f"Classes file needs txId and class columns; got {list(classes.columns)}")
    class_ids = pd.to_numeric(classes["txId"], errors="raise").astype(np.int64)
    class_map = dict(zip(class_ids, classes["class"].astype(str).str.strip().str.lower()))
    original_labels = pd.Series(tx_ids_int).map(class_map).fillna("unknown")
    labels = original_labels.map({"1": 0, "2": 1}).fillna(-1).to_numpy(dtype=np.int64)

    print(f"Reading edges: {edge_path}")
    edges = pd.read_csv(edge_path)
    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError(f"Edge file needs txId1 and txId2 columns; got {list(edges.columns)}")
    src_ids = pd.to_numeric(edges["txId1"], errors="coerce")
    dst_ids = pd.to_numeric(edges["txId2"], errors="coerce")
    src = src_ids.map(id_to_index)
    dst = dst_ids.map(id_to_index)
    valid = src.notna() & dst.notna()
    if not valid.any():
        raise ValueError("No valid edges matched feature transaction IDs")
    edge_index = torch.tensor(
        np.stack((src.loc[valid].to_numpy(dtype=np.int64), dst.loc[valid].to_numpy(dtype=np.int64))),
        dtype=torch.long,
    )

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(labels),
        timestep=torch.from_numpy(timesteps),
        tx_id=torch.from_numpy(tx_ids_int),
    )
    known = int((data.y >= 0).sum().item())
    print(
        f"Dataset ready: nodes={data.num_nodes:,}, edges={data.edge_index.shape[1]:,}, "
        f"features={data.x.shape[1]}, known labels={known:,}"
    )
    return data
