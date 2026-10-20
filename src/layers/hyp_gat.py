import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class HypGAT(nn.Module):
    def __init__(self, in_channels, out_channels, hyp_ops):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels)
        self.hyp_ops = hyp_ops

    def forward(self, x, edge_index):
        x_tan = self.hyp_ops.log_map(x)

        x_tan = self.gat(x_tan, edge_index)
        x_tan = F.leaky_relu(x_tan)

        x_hyp = self.hyp_ops.exp_map(x_tan)
        return self.hyp_ops.project(x_hyp)