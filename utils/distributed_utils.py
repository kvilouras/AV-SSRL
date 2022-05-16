import torch
from torch import distributed as dist


def _gather_from_all(tensor):
    """
    Gather tensors from all gpus
    :param tensor: Tensor to be broadcast from current process
    :return: Gathered tensor
    """

    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensor, tensor)
    return torch.cat(gathered_tensor, 0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all processes and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        gathered_tensor = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensor, x)
        return tuple(gathered_tensor)

    @staticmethod
    def backward(ctx, *grads):
        all_grads = torch.stack(grads)
        dist.all_reduce(all_grads)
        return all_grads[dist.get_rank()]
