import torch
import torch.nn as nn
import torch.distributed as dist
from utils.distributed_utils import _gather_from_all


class RINCE(nn.Module):
    """
    Robust InfoNCE Loss (https://arxiv.org/pdf/2201.04309.pdf)
    Args:
        q: Parameter in (0, 1] range which controls robustness against noisy views.
            For q -> 0, the loss becomes asymptotically equivalent to InfoNCE.
            Values of q in [0.1, 0.5] are recommended in practice.

        lam: Density weighting term in (0, 1] range. Reducing lam places more weight
            on the positive score, whereas lam = 0 recovers the negative-pair-free
            contrastive loss (such as that in BYOL). RINCE is relatively insensitive
            to the choice of this hyperparameter.
    """
    def __init__(self, q=0., lam=0.01):
        super(RINCE, self).__init__()
        self.q = q
        self.lam = lam

    def forward(self, scores_pos, scores_neg, q=None):
        """
        Calculate loss for current minibatch
        :param scores_pos: torch.FloatTensor of size (N, 1), N: batch size
        :param scores_neg: torch.FloatTensor of size (N, K), N: batch size, K: #negatives
        :param q: Parameter that controls robustness (useful here for annealing).
        :return: Loss (scalar)
        """
        # unnormalized distributions
        exp_pos_score = scores_pos.exp()
        exp_neg_score = scores_neg.exp()

        if q is None:
            q = self.q

        if q == 0:
            # InfoNCE Loss
            loss = - torch.div(exp_pos_score, exp_pos_score + exp_neg_score.sum(-1, keepdim=True)).log().mean()
        else:
            # Robust InfoNCE Loss
            pos = - exp_pos_score ** q / q
            neg = (self.lam * (exp_pos_score + exp_neg_score.sum(-1, keepdim=True))) ** q / q
            loss = pos.mean() + neg.mean()

        return loss


class NCECriterion(nn.Module):
    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem
        self.register_buffer('avg_exp_score', torch.tensor(-1.))
        self.distributed = dist.is_available() and dist.is_initialized()

    def compute_partition_function(self, x):
        with torch.no_grad():
            batch_mean = x.mean().unsqueeze(0)
            if self.distributed:
                batch_mean_gathered = _gather_from_all(batch_mean)
                all_batch_mean = batch_mean_gathered.mean().squeeze()
            else:
                all_batch_mean = batch_mean
        self.avg_exp_score = all_batch_mean
        return self.avg_exp_score

    def forward(self, scores_pos, scores_neg):
        K = scores_neg.size(1)
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)

        # unnormalized distributions
        exp_pos_score = scores_pos.exp()
        exp_neg_score = scores_neg.exp()

        # partition function Z: either 1) calculate it from scratch or 2) use precomputed value
        if self.avg_exp_score <= 0:
            self.compute_partition_function(exp_neg_score)

        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        Pmt = torch.div(exp_pos_score, exp_pos_score + K * self.avg_exp_score)
        lnPmt = - torch.log(Pmt).sum(-1)

        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon = torch.div(K * self.avg_exp_score, exp_neg_score + K * self.avg_exp_score)
        lnPon = - torch.log(Pon).sum(-1)

        loss = (lnPmt + lnPon).mean()
        return loss

