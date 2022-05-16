import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.distributed_utils import FullGatherLayer


__all__ = [
    'VICLoss'
]


class VICLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization Loss (https://arxiv.org/pdf/2105.04906.pdf)
    Args:
        sim_coeff: Coefficient of Invariance regularization loss term (default=25.0) - Should be equal to std_coeff!
        std_coeff: Coefficient of Variance regularization loss term (default=25.0) - Should be equal to sim_coeff!
        cov_coeff: Coefficient of Covariance regularization loss term (default=1.0)
        batch_size: Effective batch size, i.e. worker's batch size * world size
    """

    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, batch_size=256):
        super(VICLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.batch_size = batch_size

    def forward(self, x, y):
        """
        Calculate overall VICReg loss.
        Args:
            x, y: Mini-batches of embeddings (expander's output) of size N x D, N: batch size, D: embedding dim
        """

        # representation loss (distance between corresponding pairs of embeddings)
        repr_loss = F.mse_loss(x, y)

        # variance regularization loss (encourage intra-batch variance to be equal to gamma along each dim)
        # (from paper: gamma=1 and eps=1e-4)
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(0)
        y = y - y.mean(0)
        std_x = torch.sqrt(x.var(0) + 1e-4)
        std_y = torch.sqrt(y.var(0) + 1e-4)
        # gamma = torch.tensor(1.).to(x.device)
        # std_loss = 0.5 * (F.mse_loss(gamma, std_x.mean()) + F.mse_loss(gamma, std_y.mean()))
        std_loss = 0.5 * (torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y)))

        # covariance regularization loss (de-correlate the different embedding dimensions)
        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.shape[-1]) + off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])

        return self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
