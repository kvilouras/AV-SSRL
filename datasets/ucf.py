from pathlib import Path
import os
import subprocess
from datasets.video_dataset import VideoDataset
import numpy as np
import random

DATA_PATH = Path("UCF101/data")
ANNO_PATH = Path("UCF101/ucfTrainTestlist")


class UCF(VideoDataset):
    def __init__(self, subset, return_video=True, video_clip_duration=0.5, video_fps=16,
                 video_transform=None, return_audio=False, audio_clip_duration=1.,
                 audio_srate=16000, audio_transform=None, max_offsync=0, return_labels=False,
                 return_index=False, mode='clip', clips_per_video=20, **kwargs):

        self.root = DATA_PATH
        self.subset = subset

        classes_fn = ANNO_PATH / "classInd.txt"
        self.classes = [l.strip().split()[1] for l in open(classes_fn)]

        filenames = [ln.strip().split()[0] for ln in open(ANNO_PATH / f"{subset}.txt")]
        labels = [fn.split('/')[0] for fn in filenames]
        filenames = [fn.split('/')[1] for fn in filenames]
        # check if all files exist in DATA_PATH!
        indices = [i for i in range(len(filenames)) if not os.path.exists(os.path.join(DATA_PATH, filenames[i]))]
        for itr, i in enumerate(indices):
            filenames.pop(i - itr)
            labels.pop(i - itr)
        labels = [self.classes.index(cls) for cls in labels]

        # if audio stream needs to be returned, then remove all videos with no audio!
        if return_audio:
            indices = []
            for i in range(len(filenames)):
                pr = subprocess.Popen(
                    ["bash", "ffprobe_audio_stream_exists.sh", os.path.join(DATA_PATH, filenames[i])],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
                )
                std_out, std_err = pr.communicate()
                if eval(std_out):
                    indices.append(i)
            for itr, i in enumerate(indices):
                filenames.pop(i - itr)
                labels.pop(i - itr)

        # seen/unseen concepts split
        if 'rest_classes' in kwargs:
            if kwargs['rest_classes']:
                # keep only unseen concepts, i.e. those that belong to the rest classes
                labels_rest = [self.classes.index(cls) for cls in kwargs['rest_names']]
                # indices of samples that belong to the rest classes
                indices = [i for i in range(len(labels)) if labels[i] in labels_rest]
                filenames = [filenames[i] for i in indices]
                labels = [labels[i] for i in indices]
                # reset labels' indices to [0, len(rest_class_names)]
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

        # few-shot learning setup: keep only a certain amount of data per class
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
        
        super(UCF, self).__init__(
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

