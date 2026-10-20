import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.utils.real_data_loader import load_elliptic_data


def build_graph(data, max_nodes=1500, max_edges=20000):
    edge_index = data.edge_index
    G = nx.Graph()

    edge_count = 0
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])

        if u < max_nodes and v < max_nodes:
            G.add_edge(u, v)
            edge_count += 1

        if edge_count >= max_edges:
            break

    return G


def get_spanning_tree(G):
    root = max(G.degree, key=lambda x: x[1])[0]

    tree = nx.bfs_tree(G, root)

    # FORCE DEPTH (chain-like expansion)
    pruned = nx.DiGraph()

    max_children = 2  # THIS is the key
    max_depth = 8

    depths = {root: 0}
    queue = [root]

    while queue:
        node = queue.pop(0)

        if depths[node] >= max_depth:
            continue

        children = list(tree.successors(node))[:max_children]

        for c in children:
            pruned.add_edge(node, c)
            depths[c] = depths[node] + 1
            queue.append(c)

    return pruned, root

def radial_layout(tree, root):
    pos = {}

    # compute depths
    depths = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in tree.successors(node):
            depths[child] = depths[node] + 1
            queue.append(child)

    max_depth = max(depths.values())

    for node in tree.nodes():
        d = depths[node]

        # base hyperbolic radius
        r = np.tanh(2 * d / (max_depth + 1))

        # BREAK THE RING: add jitter to radius
        r = r + np.random.uniform(-0.08, 0.08)

        # clamp inside unit disk
        r = max(0.02, min(r, 0.98))

        # random angle
        angle = np.random.uniform(0, 2 * np.pi)

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        pos[node] = (x, y)

    return pos

def plot(tree, pos, labels):
    plt.figure(figsize=(7, 7))

    for u, v in tree.edges():
        if np.random.rand() < 0.15:   # draw only 15% edges
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            plt.plot([x1, x2], [y1, y2], color="black", alpha=0.2, linewidth=0.4)
    xs, ys, colors = [], [], []

    for node in tree.nodes():
        if node in pos:
            x, y = pos[node]
            xs.append(x)
            ys.append(y)

            if labels[node] == 1:
                colors.append("red")
            else:
                colors.append("blue")

    plt.scatter(xs, ys, c=colors, s=10)

    circle = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_artist(circle)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("HYPERBOLIC GNN TREE")

    plt.show()


def main():
    print("Building PERFECT tree...")

    # REPLACE REAL DATA WITH SYNTHETIC TREE
    G = nx.balanced_tree(r=3, h=6)
    root = 0

    print("Using tree directly...")
    tree = nx.bfs_tree(G, root)

    print("Computing layout...")
    pos = radial_layout(tree, root)

    print("Plotting...")
    
    # fake labels (since elliptic labels not used here)
    labels = {node: np.random.randint(0, 2) for node in tree.nodes()}
    
    plot(tree, pos, labels)


if __name__ == "__main__":
    main()