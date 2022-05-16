from pathlib import Path
import os
from datasets.video_dataset import VideoDataset
import numpy as np
import random

DATA_PATH = Path("hmdb51/videos")
ANNO_PATH = Path("hmdb51/splits")


class HMDB(VideoDataset):
    def __init__(self, subset, return_video=True, video_clip_duration=1., video_fps=25,
                 video_transform=None, return_audio=False, audio_clip_duration=1.,
                 audio_srate=16000, audio_transform=None, max_offsync=0, return_labels=False,
                 return_index=False, mode='clip', clips_per_video=20, **kwargs):

        self.root = DATA_PATH
        self.subset = subset

        classes = [ln.strip() for ln in open(DATA_PATH.parent / "classNames.txt") if not ln.strip().endswith('.txt')]
        subset, split = subset.split('-')
        subset_id = dict(train=1, test=2)[subset]  # split according to official README
        filenames, labels = [], []
        for cls in classes:
            for ln in open(ANNO_PATH / f"{cls}_test_{split}.txt"):
                fn, ss = ln.strip().split()
                # remove file if it does not exist in DATA_PATH!
                if not os.path.exists(os.path.join(DATA_PATH, fn)):
                    continue
                if int(ss) == subset_id:
                    filenames.append(fn)
                    labels.append(classes.index(cls))

        self.classes = classes

        # seen/unseen concepts split
        if 'rest_classes' in kwargs:
            if kwargs['rest_classes']:
                # keep only unseen concepts, i.e. those that belong to the rest classes
                labels_rest = [self.classes.index(cls) for cls in kwargs['rest_names']]
                # indices of samples that belong to the rest classes
                indices = [i for i in range(len(labels)) if labels[i] in labels_rest]
                filenames = [filenames[i] for i in indices]
                labels = [labels[i] for i in indices]
                # reset labels' indices to [0, len(rest_class_names)] range
                temp_dict = dict(zip([str(cls) for cls in labels_rest], list(range(len(labels_rest)))))
                labels = [temp_dict[str(cls)] for cls in labels]
                self.classes = kwargs['rest_names']
            else:
                # keep only seen concepts
                temp = list(set(self.classes) - set(kwargs['rest_names']))
                labels_seen = [self.classes.index(cls) for cls in temp]
                self.classes = temp
                indices = [i for i in range(len(labels)) if labels[i] in labels_seen]
                filenames = [filenames[i] for i in indices]
                labels = [labels[i] for i in indices]
                temp_dict = dict(zip([str(cls) for cls in labels_seen], list(range(len(labels_seen)))))
                labels = [temp_dict[str(cls)] for cls in labels]

        # few-shot learning setup
        if 'few_shot_ratio' in kwargs and kwargs['few_shot_ratio'] < 1.:
            new_labels, new_filenames = [], []
            for cls in set(labels):
                indices = np.where(np.array(labels) == cls)[0].tolist()
                num_samples = max(int(kwargs['few_shot_ratio'] * labels.count(cls)), 1)
                indices = random.sample(indices, k=num_samples)
                new_labels.extend([labels[i] for i in indices])
                new_filenames.extend([filenames[i] for i in indices])

            labels = new_labels
            filenames = new_filenames

        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)

        super(HMDB, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_clip_duration=audio_clip_duration,
            audio_srate=audio_srate,
            audio_transform=audio_transform,
            max_offsync=max_offsync,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            mode=mode,
            clips_per_video=clips_per_video
        )
