import random
import numpy as np
import itertools


class TimeFreqMasking(object):
    """
    Mask random time and/or frequency blocks of a given spectrogram.
    Set either of the ratios to 0 if you want to avoid a certain type of masking.
    Args:
        time_mask_ratio: Maximum area covered by a time mask wrt the input (default: 0.1)
        time_mask_size: Maximum time mask size in samples (default: 10)
        freq_mask_ratio: Maximum area covered by a frequency mask wrt the input (default: 0.1)
        freq_mask_size: Maximum frequency mask size in Mel bins (default: 2)
        var_masks: Whether the applied masks should be of varying length (default: False)
    """

    def __init__(self, time_mask_ratio=0.1, time_mask_size=10,
                 freq_mask_ratio=0.1, freq_mask_size=2, var_masks=False):
        assert 0 <= time_mask_ratio <= 1., "Time mask ratio should be in [0., 1.] range"
        assert 0 <= freq_mask_ratio <= 1., "Frequency mask ratio should be in [0., 1.] range"
        self.time_mask_ratio = time_mask_ratio
        self.time_mask_size = time_mask_size
        self.freq_mask_ratio = freq_mask_ratio
        self.freq_mask_size = freq_mask_size
        self.var_masks = var_masks

    def __call__(self, spectrogram):
        """
        Mask spectrogram
        :param spectrogram: Spectrogram of size (n_mels, time_samples) and type torch.FloatTensor
        :return: Masked spectrogram of same size and type
        """

        # set random mask ratio
        self.time_mask_ratio = random.uniform(0, self.time_mask_ratio)
        self.freq_mask_ratio = random.uniform(0, self.freq_mask_ratio)

        n_mels, time_samples = spectrogram.shape
        # max number of masks
        n_time_masks = int(self.time_mask_ratio * time_samples) // self.time_mask_size
        n_freq_masks = int(self.freq_mask_ratio * n_mels) // self.freq_mask_size
        if n_time_masks == 0 and n_freq_masks == 0:
            return spectrogram
        if n_time_masks != 0:
            # random starting points (for masking blocks)
            time_pts = random.sample(range(0, time_samples - self.time_mask_size), n_time_masks)
            # get full masking blocks
            if self.var_masks:
                # get masks of varying length
                var_lengths = random.choices(range(1, self.time_mask_size + 1), k=n_time_masks)
                time_pts = list(itertools.chain(*[
                    np.arange(time_pts[i], time_pts[i] + var_lengths[i]) for i in range(len(time_pts))
                ]))
            else:
                time_pts = list(itertools.chain(*[np.arange(t, t + self.time_mask_size) for t in time_pts]))
            # zero out these parts of the spectrogram
            spectrogram[:, time_pts] = spectrogram.min()  # spectrogram.min() or 0
        if n_freq_masks != 0:
            mel_pts = random.sample(range(0, n_mels - self.freq_mask_size), n_freq_masks)
            if self.var_masks:
                var_lengths = random.choices(range(1, self.freq_mask_size + 1), k=n_freq_masks)
                mel_pts = list(itertools.chain(*[
                    np.arange(mel_pts[i], mel_pts[i] + var_lengths[i]) for i in range(len(mel_pts))
                ]))
            else:
                mel_pts = list(itertools.chain(*[np.arange(f, f + self.freq_mask_size) for f in mel_pts]))
            spectrogram[mel_pts, :] = spectrogram.min()  # spectrogram.min() or 0

        return spectrogram

