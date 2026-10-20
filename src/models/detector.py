import torch
import torch.nn as nn

from src.layers.hyp_gat import HypGAT
from src.layers.hyp_linear import HypLinear
from src.utils.hyperbolic_ops import HyperbolicOps


class ForensicDetector(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.hyp_ops = HyperbolicOps(hidden_dim)

        self.gat1 = HypGAT(in_dim, hidden_dim, self.hyp_ops)
        self.gat2 = HypGAT(hidden_dim, hidden_dim, self.hyp_ops)

        # Lorentz 3-vector -> two-dimensional Poincare-disk coordinates
        self.embedding_layer = HypLinear(hidden_dim, 3, self.hyp_ops)

        self.classifier = nn.Linear(2, 2)

        nn.init.orthogonal_(self.embedding_layer.linear.weight)

    def forward(self, x, edge_index):
        x = self.hyp_ops.exp_map(x)

        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)

        z_hyp = self.embedding_layer(x)

        z = self.hyp_ops.lorentz_to_poincare(z_hyp)

        logits = self.classifier(z)

        return z, logits