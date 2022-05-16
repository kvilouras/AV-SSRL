import torch
import numpy as np
from librosa import filters, stft
import random
from utils.datatransforms import video_transforms, tensor_transforms, audio_transforms


class VideoPrep(object):
    """
    Preprocess video, i.e. apply augmentations, normalize and transform to torch.FloatTensor
    Args:
        crop: Crop size
        resize: Frame size after resizing
        color: Amount color jittering for (brightness, contrast, saturation, hue)
        augment: If augmentations should be applied to the input video
        min_area: Minimum crop size wrt the original image area
        msc_flag: Flag for multiscale cropping (default True).
            If set to False, it will be substituted by a resizing and a random crop transform.
        normalize: If the video should be normalized by color channel (default: True)
        totensor: If the video should be transformed to torch.FloatTensor (default: True)
        num_frames: Total number of frames (int, default: 8)
        pad_missing: Pad frames if their total number is less than `num_frames`
    """

    def __init__(self, crop=(224, 224), resize=(256, 256), color=(0.4, 0.4, 0.4, 0.2),
                 augment=True, min_area=0.08, msc_flag=True, normalize=True, totensor=True,
                 num_frames=8, pad_missing=False):
        self.num_frames = num_frames
        self.pad_missing = pad_missing
        if normalize:
            assert totensor
        if augment:
            transforms = [
                video_transforms.RandomResizedCrop(crop, scale=(min_area, 1.)),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
            if not msc_flag:
                transforms.pop(0)
                transforms = [
                    video_transforms.Resize(resize),
                    video_transforms.RandomCrop(crop),
                             ] + transforms
        else:
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.CenterCrop(crop),
            ]

        if totensor:
            transforms += [tensor_transforms.ClipToTensor()]
            if normalize:
                # Normalize using mean and std from Imagenet
                transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        if isinstance(frames[0], list):
            return torch.stack([self(f) for f in frames])
        frames = self.transform(frames)
        if self.pad_missing:
            while True:
                n_missing = self.num_frames - frames.shape[1]
                if n_missing > 0:
                    frames = torch.cat((frames, frames[:, :n_missing]), 1)
                else:
                    break
        return frames


class AudioPrep(object):
    """
    Preprocess waveform, i.e. normalize and augment it.
    Args:
        normalize: If the waveform should be normalized, default: False
        augment: If the waveform should be augmented, default: False
        tfmask: Perform time-frequency masking, default: False
        tospec: Convert waveform to log-Mel spectrogram, default: True
    """

    def __init__(self, normalize=False, augment=False, tfmask=False, tospec=True, **kwargs):
        self.normalize = normalize
        self.augment = augment
        self.tospec = tospec
        self.tfmask = tfmask
        self.kwargs = kwargs

    def __call__(self, y, srate):
        if self.normalize:
            # normalize waveform by its max norm (+ avoid division by zero)
            y = 0.95 * y / (np.linalg.norm(y, np.inf, axis=1) + 1e-9)
        if self.augment:
            # augment waveform (change its amplitude)
            y *= random.uniform(0.7, 1.0)
        if self.tospec:
            lmspec = LogMelSpectrogram(srate=srate, **self.kwargs['spec_params'])
            y = lmspec(y)
            if self.tfmask:
                tf_masking = audio_transforms.TimeFreqMasking(**self.kwargs['tfmask_params'])
                y = tf_masking(y)
            y = y.unsqueeze(0)
        else:
            y = torch.from_numpy(y)  # raw waveform!
        return y


class LogMelSpectrogram(object):
    """
    Extract log-Mel spectrogram from input raw audio
    Args:
        srate: Sampling rate
        n_fft: Number of FFT bins
        hop_length: Hop length
        n_mels: Number of mel bins
    """

    def __init__(self, srate, n_fft, hop_length, n_mels, normalize=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin, self.fmax = 0, srate / 2.0
        self.mel_basis = filters.mel(sr=srate, n_fft=n_fft, fmin=self.fmin,
                                             fmax=self.fmax, n_mels=n_mels, htk=False, norm=1)
        self.normalize = normalize
        if normalize:
            stats = np.load('datasets/assets/audio_spec_50k_norm_stats.npz')
            self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, y):
        # pad both sides of the waveform
        p = (self.n_fft - self.hop_length) // 2
        y = np.pad(y, ((0, 0), (p, p)), mode="reflect")
        y = y.reshape(-1)
        # extract spectrogram + convert to Mel scale
        spec = np.abs(stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                                   window=np.hanning(self.n_fft), center=False))
        spec = self.mel_basis @ spec
        # set lower bound to avoid -Inf in the logarithm
        spec = np.log10(spec.clip(min=1e-5))
        # z-normalization
        if self.normalize:
            spec = (spec - self.mean[:, np.newaxis]) / (self.std[:, np.newaxis] + 1e-6)
        # convert to torch.FloatTensor
        spec = torch.from_numpy(spec).float()
        return spec

