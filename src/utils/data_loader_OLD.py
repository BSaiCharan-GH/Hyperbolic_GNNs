import torch
from torch_geometric.data import Data

def generate_forensic_graph(num_entities=100, num_features=16):
    """
    Generates a synthetic financial network:
    - Normal entities: Follow a standard tree hierarchy.
    - Anomalies: Circular ownership links (A->B->C->A).
    """
    # 1. Generate random features (e.g., capital, registration age, risk score)
    x = torch.randn(num_entities, num_features)
    
    # 2. Create a basic hierarchy (Edge Index)
    # Most nodes connect to a "parent" (node 0)
    sources = torch.arange(1, num_entities)
    targets = torch.zeros(num_entities - 1, dtype=torch.long)
    
    # 3. Inject a Fraudulent Ring (e.g., nodes 90, 91, 92)
    # 90 -> 91, 91 -> 92, 92 -> 90
    ring_sources = torch.tensor([90, 91, 92], dtype=torch.long)
    ring_targets = torch.tensor([91, 92, 90], dtype=torch.long)
    
    edge_index = torch.cat([
        torch.stack([sources, targets], dim=0),
        torch.stack([ring_sources, ring_targets], dim=0)
    ], dim=1)
    
    # 4. Labels: 0 for normal, 1 for anomaly (the ring)
    y = torch.zeros(num_entities, dtype=torch.long)
    y[90:93] = 1 
    
    return Data(x=x, edge_index=edge_index, y=y)

if __name__ == "__main__":
    print("Testing Data Loader...")
    graph = generate_forensic_graph()
    print(f"Nodes: {graph.num_nodes}")
    print(f"Edges: {graph.num_edges}")
    print(f"Features per node: {graph.num_node_features}")
    print(f"Anomalies present: {graph.y.sum().item()}")