import random

import numpy as np
import PIL
import torch


class ClipToTensor(object):
    """
    Convert a list of m (H x W x C) numpy.ndarrays in range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in range [0., 1.]
    Args:
        n_channels: Number of channels (default is 3)
        normalize: Whether the output tensor should be normalized (default True)
        totensor: Convert to torch.FloatTensor (default True)
    """

    def __init__(self, n_channels=3, normalize=True, totensor=True):
        self.n_channels = n_channels
        self.normalize = normalize
        self.totensor = totensor

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            flag_pil = False
            _, _, img_c = clip[0].shape
            assert img_c == self.n_channels, f"Expected 3 channels but got {img_c}"
        elif isinstance(clip[0], PIL.Image.Image):
            flag_pil = True
        else:
            raise TypeError('Expected either numpy.ndarray or PIL.Image.Image' +
                            'but got list of {}'.format(type(clip[0])))

        if flag_pil:
            # convert to numpy
            clip = [np.array(img, copy=False) for img in clip]

        for i in range(len(clip)):
            if len(clip[i].shape) == 3:
                clip[i] = clip[i].transpose((2, 0, 1))  # convert from (H x W x C) to (C x H x W) format
            elif len(clip[i].shape) == 2:
                clip[i] = clip[i][np.newaxis, :, :]  # expand dims if the image is grayscale

        clip = np.stack(clip, axis=1)  # np.ndarray of size (C, m, H, W)

        if self.normalize:
            clip = clip / 255.

        if self.totensor:
            clip = torch.from_numpy(clip).float()

        return clip


class Normalize(object):
    """
    Normalize tensor by given mean and standard deviation
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class SpatialRandomCrop(object):
    """
    Extract a random spatial crop from a 4D (spatio-temporal) input with dimensions [Channels, Time, Height, Width]
    Args:
        size: (height, width) tuple
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, inp):
        h, w = self.size
        _, _, inp_h, inp_w = inp.shape

        if w > inp_w or h > inp_h:
            raise ValueError(f"Crop size ({h}, {w}) is larger than input size ({inp_h}, {inp_w})")

        x1 = random.randint(0, inp_w - w)
        y1 = random.randint(0, inp_h - h)

        return inp[:, :, y1:y1 + h, x1:x1 + h]

