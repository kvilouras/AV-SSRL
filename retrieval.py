import argparse
import time
import os
import yaml
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import utils.logger
from utils import main_utils, eval_utils


parser = argparse.ArgumentParser(description='Retrieval experiment on UCF-101')
parser.add_argument('cfg', metavar='CFG', help='Config file')
parser.add_argument('model_cfg', metavar='CFG', help='Model config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--mode', default=None, type=str,
                    help='Retrieval mode. Valid options are: a2v, v2a, v2v, a2a. '
                         'E.g., a2v means that, for a given video in test set, '
                         'we use its audio to query videos from the train set. '
                         'Therefore, a2v and v2a refer to cross-modal retrieval.'
                    )


def main():
    args = parser.parse_args()
    assert args.mode in ['a2v', 'v2a', 'v2v', 'a2a'], 'Unknown retrieval mode. Valid options: v2a, a2v, v2v, a2a'
    cfg = yaml.safe_load(open(args.cfg))
    main_worker(None, cfg['dataset']['fold'], args, cfg)


def main_worker(gpu, fold, args, cfg):
    args.distributed = False
    args.gpu = gpu

    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)
    train_loader, test_loader = build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], logger)
    model = build_model(model_cfg, logger)

    d = {'a': 'audio', 'v': 'video'}
    test_mode, train_mode = args.mode.split('2')
    test_mode, train_mode = d[test_mode], d[train_mode]

    # save training set embeddings in memory
    run_phase('train', train_mode, train_loader, model, args, cfg, logger)
    # use test set to query the training set
    top1, top5, q, q_lbls, train_lbls, train_inds = run_phase('test', test_mode, test_loader, model, args, cfg, logger)
    logger.add_line(f"R@1: {top1}")
    logger.add_line(f"R@5: {top5}")

    # visualization
    # 1) convert labels to class names
    train_cls = []
    for lbls in train_lbls:
        temp = []
        for lbl in lbls:
            cls = train_loader.dataset.classes[lbl]
            temp.append(cls)
        train_cls.append(temp)
    q_cls = []
    for lbl in q_lbls:
        cls = test_loader.dataset.classes[lbl]
        q_cls.append(cls)

    # 2) use the indices to get frames from the training set
    train_samples = []
    for inds in train_inds:
        temp = torch.tensor([], dtype=torch.float32)
        for idx in inds:
            sample = train_loader.dataset[idx]
            x = sample['frames']
            clip_idx = x.shape[0] // 2
            x = x[clip_idx, :, 0, ...]
            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
        train_samples.append(temp)

    # 3) create figure and save to eval_dir
    fig, ax = plt.subplots(len(train_samples), len(train_samples[0]) + 1, figsize=(12, 14))
    std, mean = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1), torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    for i, query in enumerate(q):
        # remove normalization
        query.mul_(std).add_(mean)
        query = query.permute((1, 2, 0))  # H x W x C
        ax[i, 0].imshow(query)
        ax[i, 0].set_title(q_cls[i], fontsize=10)
        for j, smpl in enumerate(train_samples[i], 1):
            smpl.mul_(std).add_(mean)
            smpl = smpl.permute((1, 2, 0))
            ax[i, j].imshow(smpl)
            ax[i, j].set_title(train_cls[i][j - 1], fontsize=10)
    [axi.set_axis_off() for axi in ax.ravel()]
    fig.tight_layout()
    x_c = 0.5 * (ax[0, 0].get_position().xmax + ax[0, 1].get_position().xmin)
    line = plt.Line2D((x_c, x_c), (0, 1), color="k", linewidth=1)  # separate query from results
    fig.add_artist(line)
    fig_name = os.path.join(eval_dir, f"{args.mode}.png")
    fig.savefig(fig_name, dpi=100)
    plt.close(fig)


def run_phase(phase, mode, loader, model, args, cfg, logger):
    from utils import metrics_utils
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    top1_meter = metrics_utils.AverageMeter('R@1', ':6.2f')
    top5_meter = metrics_utils.AverageMeter('R@5', ':6.2f')
    progress = utils.logger.ProgressMeter(
        len(loader), meters=[batch_time, data_time, top1_meter, top5_meter], phase=phase, epoch=0, logger=logger
    )
    model.train(False)  # inference mode

    end = time.time()
    logger.add_line('\n{} phase'.format(phase))
    train_inds, train_lbls = [], []  # keep the indices + labels of the queried training samples (for visualization)
    queries, queries_lbls = [], []  # also keep the queries from the test set
    for itr, sample in enumerate(loader):
        data_time.update(time.time() - end)
        video, audio, target = sample['frames'], sample['audio'], sample['label']
        if torch.is_floating_point(target):
            target = target.to(dtype=torch.int64)
        target = target.cuda()
        if mode == 'video':
            x_in = video.cuda()
        else:
            x_in = audio.cuda()

        with torch.no_grad():
            out, train_indices = model(x_in, y=target, phase=phase, mode=mode)
            if phase == 'test':
                top1, top5 = out
                top1_meter.update(top1[0], target.size(0))
                top5_meter.update(top5[0], target.size(0))
                # extract a video frame from the 1st sample in the batch for visualization
                clip_idx = x_in.shape[1] // 2
                qry = video[0, clip_idx, :, 0, ...].cpu()  # 3 x H x W
                trg_lbl = target[0].cpu()
                if len(queries) < 10:
                    queries.append(qry)
                    queries_lbls.append(trg_lbl)
                    train_lbls.append(model.target_labels[train_indices[0]].cpu())
                    train_inds.append(train_indices[0].cpu())
                else:
                    # random substitution
                    if random.random() < 0.05 and trg_lbl not in queries_lbls:
                        rand_idx = random.randint(0, 9)
                        queries.pop(rand_idx)
                        queries_lbls.pop(rand_idx)
                        train_lbls.pop(rand_idx)
                        train_inds.pop(rand_idx)
                        queries.append(qry)
                        queries_lbls.append(trg_lbl)
                        train_lbls.append(model.target_labels[train_indices[0]].cpu())
                        train_inds.append(train_indices[0].cpu())

        batch_time.update(time.time() - end)
        end = time.time()
        if (itr + 1) % 100 == 0 or itr == 0 or itr + 1 == len(loader):
            progress.display(itr + 1)

    logger.add_line(f"{phase} phase completed")

    if phase == 'test':
        return top1_meter.avg, top5_meter.avg, queries, queries_lbls, train_lbls, train_inds


class ModelWrapper(nn.Module):
    def __init__(self, video_model, audio_model):
        super(ModelWrapper, self).__init__()
        # freeze each backbone's weights
        video_model.train(False)
        audio_model.train(False)
        self.video_model = video_model
        self.audio_model = audio_model
        for p in self.video_model.parameters():
            p.requires_grad = False
        for p in self.audio_model.parameters():
            p.requires_grad = False

        self.register_buffer('target_embeddings', torch.tensor([], dtype=torch.float32, device=torch.device('cuda')))
        self.register_buffer('target_labels', torch.tensor([], dtype=torch.int64, device=torch.device('cuda')))

    def store_embeddings(self, x, y):
        """
        Store input embeddings + their labels in memory
        """

        self.target_embeddings = torch.cat((self.target_embeddings, x), dim=0)
        self.target_labels = torch.cat((self.target_labels, y), dim=0)

    def query(self, x, k=5):
        """
        Query the training set
        :param x: Input embeddings from test set
        :param k: Number of top retrieved samples (from train set) for input x (default: 5)
        :return: Indices of the topk embeddings retrieved from the training set
        """

        _, topk_inds = torch.topk(
            torch.cdist(x.unsqueeze(1), self.target_embeddings.unsqueeze(0), p=2).squeeze(1),
            k=k, largest=False
        )

        return topk_inds

    def forward(self, x, y=None, phase='train', mode='video'):
        batch_size, clips_per_sample = x.shape[:2]
        x = x.flatten(0, 1).contiguous()
        assert y is not None

        if mode == 'video':
            x = self.video_model(x)
        else:
            x = self.audio_model(x)
        x = x.squeeze()
        # average over all clips
        x = x.view(batch_size, clips_per_sample, -1).mean(1)

        if phase == 'train':
            self.store_embeddings(x, y)
            return None, None
        else:
            inds = self.query(x)
            pred = self.target_labels[inds].t()
            correct = pred.eq(y.view(1, -1).expand_as(pred))
            res = []
            for k in [1, 5]:
                if k == 5:
                    correct_k = correct[:k].sum(0).float().clamp(max=1.).sum(0, keepdim=True)
                else:
                    correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))

            return res, inds


def build_model(feat_cfg, logger):
    import models

    pretrained_net = models.__dict__[feat_cfg['arch']](**feat_cfg['args'])
    # load weights from checkpoint
    checkpoint_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
    ckp = torch.load(checkpoint_fn, map_location='cpu')
    pretrained_net.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
    model = ModelWrapper(pretrained_net.video_model, pretrained_net.audio_model)

    logger.add_line('=' * 30 + '   Model   ' + '=' * 30)
    logger.add_line(str(model))
    logger.add_line('=' * 30 + '   Parameters   ' + '=' * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line('=' * 30 + '   Pretrained model   ' + '=' * 30)
    logger.add_line('File: {}\nEpoch: {}'.format(checkpoint_fn, ckp['epoch']))

    model = model.cuda()  # only 1 GPU is used here!

    return model


def build_dataloaders(cfg, fold, num_workers, logger):
    logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
    train_loader = build_dataloader(cfg, cfg['train'], fold, num_workers)
    logger.add_line(str(train_loader.dataset))

    logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
    test_loader = build_dataloader(cfg, cfg['test'], fold, num_workers)
    logger.add_line(str(test_loader.dataset))
    return train_loader, test_loader


def build_dataloader(cfg, split_cfg, fold, num_workers):
    import torch.utils.data as data
    from datasets import preprocessing
    import datasets

    # Video transforms
    num_frames = int(cfg['video_clip_duration'] * cfg['video_fps'])
    if cfg['transforms'] == 'crop+color':
        msc_flag = False
    elif cfg['transforms'] == 'msc+color':
        msc_flag = True
    else:
        raise ValueError('Unknown video transform')
    video_transform = preprocessing.VideoPrep(
        crop=(cfg['crop_size'], cfg['crop_size']),
        resize=cfg['frame_size'] if 'frame_size' in cfg else (256, 256),
        augment=split_cfg['use_augmentation'],
        normalize=True,
        msc_flag=msc_flag,
        num_frames=num_frames,
        pad_missing=True,
        min_area=cfg['min_area'] if 'min_area' in cfg else 0.08,
        color=cfg['color'] if 'color' in cfg else (0.4, 0.4, 0.4, 0.2)
    )

    # Audio transforms
    audio_transform = [
        preprocessing.AudioPrep(
            normalize=True,
            augment=split_cfg['use_augmentation'],
            tospec=True,
            tfmask=split_cfg['use_augmentation'],
            spec_params=cfg['spec_params'] if 'spec_params' in cfg else dict(),
            tfmask_params=cfg['tfmask_params'] if 'tfmask_params' in cfg else dict()
        )
    ]

    dataset = datasets.UCF
    db = dataset(
        subset=split_cfg['split'].format(fold=fold),
        return_video=True,
        video_clip_duration=cfg['video_clip_duration'],
        video_fps=cfg['video_fps'],
        video_transform=video_transform,
        return_audio=True,
        audio_clip_duration=cfg['audio_clip_duration'] if 'audio_clip_duration' in cfg else 0,
        audio_srate=cfg['audio_srate'] if 'audio_srate' in cfg else 0,
        audio_transform=audio_transform,
        max_offsync=0.5 if split_cfg['use_augmentation'] else 0,
        return_labels=True,
        return_index=False,
        mode=split_cfg['mode'],
        clips_per_video=split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1,
    )

    loader = data.DataLoader(
        db,
        batch_size=max(1, cfg['batch_size'] // split_cfg['clips_per_video']),
        shuffle=split_cfg['use_shuffle'],
        drop_last=split_cfg['drop_last'] if 'drop_last' in split_cfg else True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None
    )

    return loader


if __name__ == '__main__':
    main()
