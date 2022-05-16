import csv
from pathlib import Path
from datasets.video_dataset import VideoDataset

DATA_PATH = Path("vggsound_data/")
FNS_PATH = Path("datasets/vggsound_fnames")


class VGGSoundClasses:
    def __init__(self):
        with open('vggsound.csv', 'r') as f:
            reader = csv.reader(f)
            self.classes = [r[2] for r in reader]
        # assign unique int (label) to each class
        self.d = dict([(y, x) for x, y in enumerate(sorted(set(self.classes)))])
        self.class_label = [self.d[x] for x in self.classes]

    def __getitem__(self, idx):
        return self.classes[idx]

    def __len__(self):
        return len(self.d.keys())

    def class2idx(self, class_string):
        return self.d[class_string]


class VGGSound(VideoDataset):
    def __init__(self, subset, return_video=True, video_clip_duration=1., video_fps=16,
                 video_transform=None, return_audio=True, audio_clip_duration=1.,
                 audio_srate=16000, audio_transform=None, max_offsync=0, return_labels=False,
                 return_index=False, mode='clip', clips_per_video=1, **kwargs):
        with open(FNS_PATH / f"filenames_{subset}.txt", 'r') as f:
            filenames = [line.rstrip() for line in f.readlines()]
        root = DATA_PATH / subset.split('_')[0]  # videos are either in 'train' or 'test' directory

        if return_labels:
            classes = VGGSoundClasses()
            all_fnames = []
            for fname in filenames:
                try:
                    name, start = fname.split('.')[0].split('_')
                except ValueError:
                    start = fname.split('.')[0].split('_')[-1]
                    name = fname.split('.')[0][:-len(start)-1]
                all_fnames.append((name, start))
            tempdict = dict()
            with open('vggsound.csv', 'r') as f:
                reader = csv.reader(f)
                for r in reader:
                    tempdict[(r[0], r[1])] = r[2]
            labels = []
            for fname in all_fnames:
                labels.append(classes.class2idx(tempdict[fname]))
            self.num_classes = len(classes)
        else:
            labels = None

        self.num_videos = len(filenames)

        super(VGGSound, self).__init__(
            return_video=return_video,
            video_root=root,
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
            clips_per_video=clips_per_video,
            **kwargs
        )

