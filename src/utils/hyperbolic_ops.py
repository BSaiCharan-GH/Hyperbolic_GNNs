import torch
import geoopt


class HyperbolicOps:
    def __init__(self, dim):
        self.manifold = geoopt.manifolds.Lorentz()
        self.dim = dim

    def exp_map(self, x):
        return self.manifold.expmap0(x)

    def log_map(self, x):
        return self.manifold.logmap0(x)

    def project(self, x):
        return self.manifold.projx(x)

    def lorentz_to_poincare(self, x):
        # Convert (N, d+1) → (N, d)
        return x[:, 1:] / (x[:, :1] + 1)