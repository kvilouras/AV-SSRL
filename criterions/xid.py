import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.distributed_utils import _gather_from_all
from utils.alias_method import AliasMethod
from criterions.nce import RINCE


__all__ = [
    'xID'
]


class xIDSimilarityMemoryBank(nn.Module):
    """
    Handle memory bank and calculate similarity scores
    Args:
        memory_size: Size of memory bank.
        embedding_dim: Embeddings' dimensionality.
        num_negatives: Number of negative samples (default: 1024).
        hard_neg_epoch: Starting epoch for hard negative sampling
        N_hard: Number of hard negative samples to be drawn from memory bank (for each query).
        s: Number of synthetic negative samples created by mixing two hard negatives (for each query).
            Note that these synthetic negatives will be appended to the existing ones.
        s_prime: Number of synthetic negative samples created by mixing each query with a hard negative.
            Note that these synthetic negatives will be appended to the existing ones.
        momentum: momentum coefficient (for updating the memory bank).
        device: Device id.
    """

    def __init__(self, memory_size, embedding_dim, num_negatives=1024, hard_neg_epoch=10,
                 N_hard=0, s=0, s_prime=0, momentum=0.5, device=0):
        super(xIDSimilarityMemoryBank, self).__init__()
        self.num_negatives = num_negatives
        self.temperature = 0.07
        if not isinstance(momentum, (list, tuple)):
            momentum = [momentum] * 2
        self.momentum = momentum
        self.device = device

        self.multinomial = AliasMethod(torch.ones(memory_size - 1))

        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0

        self.init_memory(memory_size, embedding_dim)
        self.embedding_dim = embedding_dim

        self.hard_neg_epoch = hard_neg_epoch
        self.N_hard = N_hard
        self.s = s
        self.s_prime = s_prime

    @staticmethod
    def compute_scores(context_emb, target_emb, temperature):
        return [torch.bmm(trg, context_emb).squeeze(-1) / temperature for trg in target_emb]

    def forward(self, video_emb, audio_emb, y, epoch):
        K = int(self.num_negatives)
        batch_sz = video_emb.shape[0]

        # L2-Normalize the embeddings
        video_emb = F.normalize(video_emb, p=2, dim=1).unsqueeze(-1)
        audio_emb = F.normalize(audio_emb, p=2, dim=1).unsqueeze(-1)

        # sample from memory bank
        with torch.no_grad():
            video_pos_mem = self.view1_mem[y].unsqueeze(1)
            audio_pos_mem = self.view2_mem[y].unsqueeze(1)

            negative_idx = self.sample_negatives(y, K).to(video_emb.device)
            video_neg_mem = self.view1_mem[negative_idx].view(batch_sz, K, self.embedding_dim)
            audio_neg_mem = self.view2_mem[negative_idx].view(batch_sz, K, self.embedding_dim)

            if self.N_hard and epoch >= self.hard_neg_epoch:
                # sample synthetic hard negatives
                video_hard_ind = self.sample_hard_negatives(audio_emb, y, self.N_hard, 'video')
                audio_hard_ind = self.sample_hard_negatives(video_emb, y, self.N_hard, 'audio')
                video_hard_neg = self.hard_negative_mixing(audio_emb, y, video_hard_ind, self.s, self.s_prime, 'video')
                audio_hard_neg = self.hard_negative_mixing(video_emb, y, audio_hard_ind, self.s, self.s_prime, 'audio')
                if video_hard_neg is not None:
                    video_neg_mem = torch.cat([video_neg_mem, video_hard_neg], dim=1)
                if audio_hard_neg is not None:
                    audio_neg_mem = torch.cat([audio_neg_mem, audio_hard_neg], dim=1)

        # compute cross-modal scores (video2audio and audio2video, respectively)
        scores = dict()
        scores['v2a'] = self.compute_scores(video_emb, [audio_pos_mem, audio_neg_mem], self.temperature)
        scores['a2v'] = self.compute_scores(audio_emb, [video_pos_mem, video_neg_mem], self.temperature)

        # update memory bank
        self.update_memory(video_emb.squeeze(-1), audio_emb.squeeze(-1), y)

        return scores

    def init_memory(self, memory_size, embedding_dim):
        """
        Initialize memory bank
        :param memory_size: Memory bank size
        :param embedding_dim: Embedding size
        :return: None
        """
        self.register_buffer('view1_mem', torch.randn(memory_size, embedding_dim))
        self.register_buffer('view2_mem', torch.randn(memory_size, embedding_dim))

        self.view1_mem = F.normalize(self.view1_mem, p=2, dim=1)
        self.view1_mem = self.view1_mem.cuda(self.device)

        self.view2_mem = F.normalize(self.view2_mem, p=2, dim=1)
        self.view2_mem = self.view2_mem.cuda(self.device)

        if self.distributed:
            dist.broadcast(self.view1_mem, src=0)
            dist.broadcast(self.view2_mem, src=0)
            dist.barrier()

    def sample_negatives(self, y, K):
        """
        Draw negative samples from memory bank
        :param y: Indices of the current samples
        :param K: Number of negative samples
        :return: Indices of the negative samples to be drawn from the memory bank
        """

        batch_size = y.shape[0]
        idx = self.multinomial.draw(batch_size * K).view(batch_size, -1).to(y.device)
        idx = idx + (idx >= y.unsqueeze(1)).long()  # avoid same index
        return idx

    def sample_hard_negatives(self, emb, y, k, mem_bank='video'):
        """
        Draw hard negative samples, i.e. those that are closest to the current samples (queries)
        in the embedding space. Note that this implementation loads the entire memory bank (cloned)
        to compute L2-distances, however we get accurate results.
        :param emb: Input embeddings of size B x d, B: batch size, d: embedding's dimensionality
        :param y: Indices of the queries (i.e. input embeddings)
        :param k: Total number of hard negatives per query
        :param mem_bank: Type of memory bank (either 'video' or 'audio'). E.g. for audio queries, we set
                            this to 'video' to draw hard negatives from video memory bank (and vice versa).
        :return: Indices of the hard negatives to be drawn from memory bank (size: B x k, B: batch size)
        """

        assert mem_bank in ['video', 'audio'], 'Unknown memory bank type'

        emb = emb.squeeze(-1)
        batch_size = emb.shape[0]

        if mem_bank == 'video':
            # draw video hard negatives (i.e. wrt audio queries)
            temp_mem = self.view1_mem.clone()
        else:
            # draw audio hard negatives (i.e. wrt video queries)
            temp_mem = self.view2_mem.clone()

        temp_mem = temp_mem.unsqueeze(0).repeat(batch_size, 1, 1)
        temp_mem[torch.arange(batch_size), y] = float("-Inf")  # avoid self
        _, ind = torch.topk(
            torch.cdist(emb.unsqueeze(1), temp_mem, p=2).squeeze(1),
            k=k, largest=False)

        return ind

    def hard_negative_mixing(self, emb, y, hard_indices, s=0, s_prime=0, mem_bank='video'):
        """
        Hard negative mixing (https://arxiv.org/pdf/2010.01028.pdf). There are two options:
            - Create synthetic hard negatives that are convex combinations of two existing
                hard negatives (mixup coefficient in (0, 1) range).
            - Create synthetic harder negatives that are convex combinations of the query
                with an existing hard negative (mixup coefficient in (0, 0.5) range).
        :param emb: Input embeddings of size B x d, B: batch size, d: embedding's dimensionality
        :param y: Indices of queries
        :param hard_indices: Indices of hard negatives (per query)
        :param s: Number of synthetic hard negatives
        :param s_prime: Number of synthetic harder negatives
        :param mem_bank: Memory bank type (either 'video' or 'audio')
        :return: Synthetic negatives of size: batch_size x (s + s_prime) x embedding_dim
        """

        assert mem_bank in ['video', 'audio'], 'Unknown memory bank type'
        if s == 0 and s_prime == 0:
            return None

        batch_sz = y.shape[0]
        # Draw hard negatives (size: batch_sz x k x emb_dim)
        if mem_bank == 'video':
            hard_negs = self.view1_mem[hard_indices.reshape(-1)].view(*hard_indices.shape, self.embedding_dim)
        else:
            hard_negs = self.view2_mem[hard_indices.reshape(-1)].view(*hard_indices.shape, self.embedding_dim)

        if s != 0:
            # draw synthetic hard negatives
            coeffs = torch.FloatTensor(batch_sz, s).uniform_(1e-4, 1-1e-4).to(y.device)  # mixup coefficients in (0, 1) range
            mix1 = torch.add(
                torch.mul(
                    coeffs.unsqueeze(-1), hard_negs[:, torch.randint(0, hard_negs.shape[1], size=(s,)), :]
                ),
                torch.mul(
                    (1 - coeffs).unsqueeze(-1), hard_negs[:, torch.randint(0, hard_negs.shape[1], size=(s,)), :]
                )
            )
            mix1 = F.normalize(mix1, p=2, dim=-1)
        if s_prime != 0:
            emb = emb.squeeze(-1)
            # draw synthetic harder negatives
            coeffs = torch.FloatTensor(batch_sz, s_prime).uniform_(1e-4, 0.5 - 1e-4).to(y.device)  # mixup coeffs in (0, 0.5) range
            mix2 = torch.add(
                torch.mul(
                    coeffs.unsqueeze(-1), emb.unsqueeze(1).expand(-1, s_prime, -1)
                ),
                torch.mul(
                    (1 - coeffs).unsqueeze(-1), hard_negs[:, torch.randint(0, hard_negs.shape[1], size=(s_prime,)), :]
                )
            )
            mix2 = F.normalize(mix2, p=2, dim=-1)

        if s != 0 and s_prime != 0:
            return torch.cat([mix1, mix2], dim=1)
        elif s != 0:
            return mix1
        else:
            return mix2

    def update_memory(self, video_emb, audio_emb, y):
        """
        Update memory bank
        :param video_emb: Video embeddings
        :param audio_emb: Audio embeddings
        :param y: Indices of current samples (corresponding to their position in memory bank)
        :return: None
        """

        video_mom, audio_mom = float(self.momentum[0]), float(self.momentum[1])

        if self.distributed:
            video_emb_gathered = _gather_from_all(video_emb)
            audio_emb_gathered = _gather_from_all(audio_emb)
            y_gathered = _gather_from_all(y)
        else:
            video_emb_gathered = video_emb
            audio_emb_gathered = audio_emb
            y_gathered = y

        # update video/audio memories
        with torch.no_grad():
            l1_pos = self.view1_mem.index_select(0, y_gathered.view(-1))
            l1_pos.mul_(video_mom)
            l1_pos.add_(torch.mul(video_emb_gathered, 1 - video_mom))
            updated_l1 = F.normalize(l1_pos, p=2, dim=1)
            self.view1_mem.index_copy_(0, y_gathered, updated_l1)

            l2_pos = self.view2_mem.index_select(0, y_gathered.view(-1))
            l2_pos.mul_(audio_mom)
            l2_pos.add_(torch.mul(audio_emb_gathered, 1 - audio_mom))
            updated_l2 = F.normalize(l2_pos, p=2, dim=1)
            self.view2_mem.index_copy_(0, y_gathered, updated_l2)


class xID(nn.Module):
    def __init__(self, num_instances, q=0., lam=0.01, embedding_dim=128, num_negatives=1024,
                 hard_neg_epoch=10, N_hard=0, s=0, s_prime=0, momentum=0.9, checkpoint=None, device=0):
        super(xID, self).__init__()
        self.nce_average = xIDSimilarityMemoryBank(
            memory_size=num_instances,
            embedding_dim=embedding_dim,
            num_negatives=num_negatives,
            hard_neg_epoch=hard_neg_epoch,
            N_hard=N_hard,
            s=s,
            s_prime=s_prime,
            momentum=momentum,
            device=device
        )
        self.nce_average = self.nce_average.cuda(device)

        self.criterion = RINCE(q=q, lam=lam)

        # restore memory bank if necessary
        if checkpoint is not None:
            ckp = torch.load(checkpoint, map_location='cpu')['train_criterion']
            state_dict = self.state_dict()

            state_dict['nce_average.view1_mem'] = ckp['nce_average.view1_mem']
            state_dict['nce_average.view2_mem'] = ckp['nce_average.view2_mem']

            self.load_state_dict(state_dict, strict=True)

    def forward(self, video_emb, audio_emb, target, epoch, q=None):
        """
        Calculate final cross-modal instance discrimination loss
        :param video_emb: Input video embedding of size (N, D), N: batch size, D: embedding size
        :param audio_emb: Input audio embedding of size (N, D)
        :param target: Instance labels of size (N,)
        :param epoch: Current epoch (considered only for hard negative mixing)
        :param q: Parameter that controls robustness against noisy views (for RINCE)
        :return: Cross-modal instance discrimination loss for current mini-batch
        """

        scores = self.nce_average(video_emb, audio_emb, target, epoch)
        xID_loss = 0.
        for k in scores.keys():
            # video2audio and audio2video, respectively
            loss = self.criterion(*scores[k], q=q)
            xID_loss += loss / 2

        return xID_loss

    def set_epoch(self, epoch):
        # TODO: use this function if you want to draw hard negatives every `resample_freq` epochs (should be also set
        #       in config file (e.g. loss -> args -> resample_freq: 5).
        pass
