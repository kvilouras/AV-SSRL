import os
import random
import numpy as np
import torch
from torch.utils import data
from utils import av_wrappers
from collections import defaultdict


def chararray(fn_list):
    """
    Convert list of filenames into numpy.chararray
    :param fn_list: Input filename list
    :return: numpy.chararray containing all filenames
    """
    charr = np.chararray(shape=(len(fn_list), ), itemsize=max([len(fn) for fn in fn_list]))
    for i in range(len(fn_list)):
        charr[i] = fn_list[i]

    return charr


class VideoDataset(data.Dataset):
    def __init__(self, return_video=True, video_root=None, video_fns=None, video_clip_duration=1.,
                 video_fps=16, video_transform=None, return_audio=True, audio_clip_duration=1.,
                 audio_srate=16000, audio_transform=None, max_offsync=0, return_labels=False,
                 labels=None, return_index=False, mode='clip', clips_per_video=1, **kwargs):
        super(VideoDataset, self).__init__()

        self.return_video = return_video
        self.video_root = video_root
        self.video_fps = video_fps
        self.video_fns = chararray(video_fns)
        self.num_samples = self.video_fns.shape[0]
        if video_transform and not isinstance(video_transform, list):
            video_transform = [video_transform]
        self.video_transform = video_transform
        self.video_clip_duration = video_clip_duration

        self.return_audio = return_audio
        self.audio_srate = audio_srate
        self.audio_transform = audio_transform
        self.audio_clip_duration = audio_clip_duration
        self.max_offsync = max_offsync

        self.mode = mode
        self.return_labels = return_labels
        if return_labels:
            self.labels = np.array(labels)
        self.clips_per_video = clips_per_video
        self.return_index = return_index

        # Experiment with VICReg: try uni-modal (video only) self-supervised learning. Here, we
        # only extract two non-overlapping clips from the same video.
        # In the following, we will refer to the 2nd clip as `audio' for variable re-usage.
        if 'video_only' in kwargs and kwargs['video_only']:
            self.video_only = True
            self.return_audio = False
        else:
            self.video_only = False

    def __getitem__(self, index):
        if self.mode == 'clip':
            try:
                sample_idx = index % self.num_samples
                container = self._load_sample(sample_idx)
                video_start, video_dur, audio_start, audio_dur = self._sample_snippet(container)
                sample = self._get_clip(sample_idx, container, video_start, audio_start, video_dur, audio_dur)
                if sample is None:
                    return self[(index + 1) % len(self)]
                return sample
            except Exception:
                return self[(index + 1) % len(self)]

        else:
            assert not self.video_only, "'Video only' variable should be set to False during 'video' mode."
            container = self._load_sample(index)
            # load entire video
            video_start, video_end, audio_start, audio_end = self._get_time_limits(container)
            if self.return_audio:
                start_time = max(video_start, audio_start) if audio_start < 0 else video_start
                end_time = min(video_end, audio_end) if audio_start < 0 else video_end
            else:
                start_time, end_time = video_start, video_end
            if end_time <= start_time:
                end_time = start_time + max(self.video_clip_duration, self.audio_clip_duration)
            video_dur = end_time - start_time
            sample = self._get_clip(index, container, start_time, start_time, video_dur, video_dur)

            # split video into overlapping chunks
            # this was originally list, not torch.Tensor
            chunks = defaultdict(torch.Tensor)  # if a key is not found, create a new entry instead of throwing KeyError
            if self.return_video:
                num_frames = sample['frames'].shape[1]
                chunk_size = int(self.video_clip_duration * self.video_fps)
                if chunk_size >= num_frames:
                    chunks['frames'] = torch.stack([sample['frames'] for _ in range(self.clips_per_video)])
                else:
                    timestamps = np.linspace(0, max(num_frames - chunk_size, 1), self.clips_per_video).astype(int)
                    chunks['frames'] = torch.stack([sample['frames'][:, s:s+chunk_size] for s in timestamps])

            if self.return_audio:
                num_frames = sample['audio'].shape[1]
                # originally, it was self.audio_fps_out instead of self.audio_srate
                chunk_size = int(self.audio_clip_duration * self.audio_srate)
                if chunk_size >= num_frames:
                    chunks['audio'] = torch.stack([sample['audio'] for _ in range(self.clips_per_video)])
                else:
                    timestamps = np.linspace(0, max(num_frames - chunk_size, 1), self.clips_per_video).astype(int)
                    chunks['audio'] = torch.stack([sample['audio'][:, s:s+chunk_size] for s in timestamps])

            if self.return_labels:
                chunks['label'] = sample['label']

            if self.return_index:
                ts = torch.from_numpy(np.linspace(start_time, end_time - self.video_clip_duration, self.clips_per_video))
                chunks['index'] = torch.stack([torch.tensor(sample['index']).repeat(self.clips_per_video), ts.float()], dim=1)

            return chunks

    def _load_sample(self, sample_idx):
        filename = os.path.join(self.video_root, self.video_fns[sample_idx].decode())
        video_cntr = av_wrappers.av_open(filename)

        return video_cntr

    def __len__(self):
        if self.mode == 'clip':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples

    def _get_time_limits(self, container):
        """
        Get video/audio start and end time (in seconds)
        :param container: Input container of type 'av.container.input.InputContainer'
        :return: Video start time, video end time, audio start time, audio end time
        """

        video_start, video_end, audio_start, audio_end = None, None, None, None
        if container:
            if self.return_video:
                video_stream = container.streams.video[0]
                video_start = video_stream.start_time * video_stream.time_base
                video_dur = video_stream.duration * video_stream.time_base
                video_end = video_start + video_dur

            if self.return_audio:
                audio_stream = container.streams.audio[0]
                audio_start = audio_stream.start_time * audio_stream.time_base
                audio_dur = audio_stream.duration * audio_stream.time_base
                audio_end = audio_start + audio_dur

        return video_start, video_end, audio_start, audio_end

    def _sample_snippet(self, container):
        """
        Sample a start/end point for both the video and the audio stream
        :param container: Input container of type 'av.container.input.InputContainer'
        :return: Sampled video start point, video clip duration, sampled audio start point,
            audio clip duration
        """

        video_start, video_end, audio_start, audio_end = self._get_time_limits(container)

        if self.return_audio:
            # audio and video stream will be slightly off-sync (different start/end points)
            min_start = max(audio_start, video_start)
            max_start = min(audio_end - self.audio_clip_duration, video_end - self.video_clip_duration)
            assert max_start > min_start

            if self.audio_clip_duration > self.video_clip_duration:
                # sample random start and end points for audio
                sample_audio_start = random.uniform(min_start, max_start)
                sample_audio_end = sample_audio_start + self.audio_clip_duration
                # sample video startpoint (its sync with audio is controlled by max_offsync parameter)
                win_min = max(sample_audio_start - self.max_offsync, video_start)
                win_max = min(sample_audio_end + self.max_offsync, video_end) - self.video_clip_duration
                sample_video_start = random.uniform(win_min, win_max)
            else:
                sample_video_start = random.uniform(min_start, max_start)
                sample_video_end = sample_video_start + self.video_clip_duration
                win_min = max(sample_video_start - self.max_offsync, audio_start)
                win_max = min(sample_video_end + self.max_offsync, audio_end) - self.audio_clip_duration
                sample_audio_start = random.uniform(win_min, win_max)

            return sample_video_start, self.video_clip_duration, sample_audio_start, self.audio_clip_duration
        else:
            # return only video stream's start point and duration
            video_dur = video_end - video_start
            if self.video_clip_duration > video_dur:
                sample_video_start = 0
                duration = video_dur
            else:
                sample_video_start = random.uniform(video_start, video_end - self.video_clip_duration)
                duration = self.video_clip_duration
                if self.video_only:
                    # sample a second clip from the same video (no overlap)
                    sample_video_start2 = video_end
                    while sample_video_start2 + duration >= video_end or sample_video_start2 <= video_start:
                        dt = random.uniform(2., 8.)  # sample distance from 1st clip (between 2 sec and 8 sec)
                        sample_video_start2 = (sample_video_start + dt) % video_end
                    return sample_video_start, duration, sample_video_start2, duration

            return sample_video_start, duration, sample_video_start, duration

    def _get_clip(self, clip_idx, container, video_start_time, audio_start_time,
                  video_clip_duration=None, audio_clip_duration=None):
        """
        Extract video/audio clip
        :param clip_idx: Index of the clip to be extracted
        :param container: Input container of type 'av.container.input.InputContainer'
        :param video_start_time: Video start point
        :param audio_start_time: Audio start point
        :param video_clip_duration: Duration of the extracted video clip
        :param audio_clip_duration: Duration of the extracted audio clip
        :return: Dictionary that contains some (or all) of the following:
            1) Video frames, 2) Audio samples, 3) Clip's label, 4) Clip's index
        """

        if video_clip_duration is None:
            video_clip_duration = self.video_clip_duration
        if audio_clip_duration is None:
            audio_clip_duration = self.audio_clip_duration

        sample = dict()
        if self.return_video:
            frames, fps, start_time = av_wrappers.av_load_video(container, self.video_fps,
                                                                video_clip_duration, video_start_time)
            if self.video_only:
                frames2, _, _ = av_wrappers.av_load_video(container, self.video_fps,
                                                          audio_clip_duration, audio_start_time)  # 2nd clip
            if self.video_transform:
                for t in self.video_transform:
                    frames = t(frames)
                    if self.video_only:
                        frames2 = t(frames2)

            sample['frames'] = frames
            if self.video_only:
                sample['audio'] = frames2  # re-use `audio' variable to minimize the required changes

            audio_start_time -= (video_start_time - start_time)  # difference due to frame extraction at different fps

        if self.return_audio:
            samples, srate = av_wrappers.av_load_audio(container, self.audio_srate,
                                                       audio_clip_duration, audio_start_time)
            if self.audio_transform:
                if isinstance(self.audio_transform, list):
                    for t in self.audio_transform:
                        samples = t(samples, srate)
                else:
                    samples = self.audio_transform(samples, srate)

            sample['audio'] = samples

        if self.return_labels:
            labels = self.labels[clip_idx]
            sample['label'] = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

        if self.return_index:
            sample['index'] = clip_idx

        return sample

