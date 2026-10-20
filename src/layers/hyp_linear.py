import torch
import torch.nn as nn


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, hyp_ops):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.hyp_ops = hyp_ops

    def forward(self, x):
        x_tan = self.hyp_ops.log_map(x)
        x_tan = self.linear(x_tan)
        x_hyp = self.hyp_ops.exp_map(x_tan)
        return self.hyp_ops.project(x_hyp)