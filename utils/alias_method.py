import torch


class AliasMethod(object):
    """
    Alias sampling method for efficient multinomial sampling (treated as a combination of
        uniform sampling and bernoulli sampling).
    Idea:
        1) construct K alias-outcomes (with prob. 1/K each) by combining one of the original-outcomes
            (with prob. less than 1/K) with some prob. mass from another original-outcome (of higher prob.)
            (so that each alias-outcome reaches a prob. of 1/K)
        2) Then we store the information of which original-outcome contributes to which alias-outcome in arrays
            of size K.
        3) Sampling: first, we sample one of the alias-outcomes according to the uniform prob. Then, we look up
            which original-outcomes contributed to this alias and how much. Last, sampling one of these two
            original outcomes is trivial (i.e. by drawing a pseudorandom number from uniform [0,1]). This is
            repeated for N times, where N = #samples needed.

    Adapted from: https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.zeros(K, dtype=torch.long)

        # sort data into outcomes with probabilities greater and smaller than 1/K
        smaller, larger = [], []
        for k, prob in enumerate(probs):
            self.prob[k] = K * prob
            if self.prob[k] < 1.0:
                smaller.append(k)
            else:
                larger.append(k)

        # loop through and create binary mixtures that appropriately
        # allocate larger outcomes over the overall uniform mixture
        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def to(self, device):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        """
        Draw N samples from multinomial distribution
        :param N: number of samples
        :return: drawn samples
        """

        K = self.alias.size(0)
        k = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, k)
        alias = self.alias.index_select(0, k)
        b = torch.bernoulli(prob).long()  # whether a random number is greater than q
        oq = k.mul(b)
        oj = alias.mul(1 - b)

        return oq + oj
