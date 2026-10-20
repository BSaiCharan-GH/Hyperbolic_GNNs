import os
import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def load_elliptic_data():
    print("=== LOADER FUNCTION CALLED ===")

    DATA_PATH = os.path.join("data", "elliptic")

    features_path = os.path.join(DATA_PATH, "elliptic_txs_features.csv")
    classes_path = os.path.join(DATA_PATH, "elliptic_txs_classes.csv")
    edges_path = os.path.join(DATA_PATH, "elliptic_txs_edgelist.csv")

    # Load
    features_df = pd.read_csv(features_path, header=None)
    classes_df = pd.read_csv(classes_path)
    edges_df = pd.read_csv(edges_path)

    # Features
    tx_ids = features_df.iloc[:, 0].values
    features = features_df.iloc[:, 1:].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    tx_id_map = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    x = torch.tensor(features, dtype=torch.float)

    # Labels
    classes_df = classes_df.set_index("txId")

    labels = []
    for tx_id in tx_ids:
        if tx_id in classes_df.index:
            label = classes_df.loc[tx_id]["class"]
            if label == "unknown":
                labels.append(-1)
            else:
                labels.append(int(label) - 1)
        else:
            labels.append(-1)

    y = torch.tensor(labels, dtype=torch.long)

    # Edges
    edges = []
    for _, row in edges_df.iterrows():
        src, dst = row["txId1"], row["txId2"]
        if src in tx_id_map and dst in tx_id_map:
            edges.append([tx_id_map[src], tx_id_map[dst]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)

    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    return data