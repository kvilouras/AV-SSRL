import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import utils.logger
import os
import yaml
import shutil


def prepare_environment(args, cfg, fold):
    # first initialize distributed backend (if applicable)
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.gpu)

    model_cfg = yaml.safe_load(open(args.model_cfg))['model']
    eval_dir = '{}/{}/eval-{}/fold-{:02d}'.format(model_cfg['model_dir'], model_cfg['name'],
                                                  cfg['benchmark']['name'], fold)
    os.makedirs(eval_dir, exist_ok=True)
    yaml.safe_dump(cfg, open('{}/config.yaml'.format(eval_dir), 'w'))

    logger = utils.logger.Logger(quiet=args.quiet, log_fn='{}/eval.log'.format(eval_dir), rank=args.gpu)
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))

    logger.add_line("=" * 30 + "   Config   " + "=" * 30)

    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  ' + ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))

    print_dict(cfg)
    logger.add_line("=" * 30 + "   Model Config   " + "=" * 30)
    print_dict(model_cfg)

    return eval_dir, model_cfg, logger


def distribute_model_to_cuda(model, args, cfg):
    if torch.cuda.device_count() == 1:
        if isinstance(model, list):
            for i in range(len(model)):
                model[i] = model[i].cuda()
        else:
            model = model.cuda()
    elif args.distributed:
        torch.cuda.set_device(args.gpu)
        if isinstance(model, list):
            for i in range(len(model)):
                model[i] = model[i].cuda(args.gpu)
                model[i] = nn.parallel.DistributedDataParallel(model[i], device_ids=[args.gpu])
        else:
            model.cuda(args.gpu)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        cfg['dataset']['batch_size'] = max(cfg['dataset']['batch_size'] // args.world_size, 1)
        cfg['num_workers'] = max(cfg['num_workers'] // args.world_size, 1)
    else:
        if isinstance(model, list):
            for i in range(len(model)):
                model[i] = nn.DataParallel(model[i]).cuda()
        else:
            model = nn.DataParallel(model).cuda()

    return model


def build_dataloader(db_cfg, split_cfg, fold, num_workers, distributed):
    import torch.utils.data as data
    from datasets import preprocessing
    import datasets

    # Video transforms
    num_frames = int(db_cfg['video_clip_duration'] * db_cfg['video_fps'])
    if db_cfg['transforms'] == 'crop+color':
        msc_flag = False
    elif db_cfg['transforms'] == 'msc+color':
        msc_flag = True
    else:
        raise ValueError('Unknown video transform')
    video_transform = preprocessing.VideoPrep(
        crop=(db_cfg['crop_size'], db_cfg['crop_size']),
        resize=db_cfg['frame_size'] if 'frame_size' in db_cfg else (256, 256),
        augment=split_cfg['use_augmentation'],
        msc_flag=msc_flag,
        num_frames=num_frames,
        pad_missing=True,
        min_area=db_cfg['min_area'] if 'min_area' in db_cfg else 0.08,
        color=db_cfg['color'] if 'color' in db_cfg else (0.4, 0.4, 0.4, 0.2)
    )

    # Audio transforms (audio is not used in downstream evaluation!)
    audio_transform = [
        preprocessing.AudioPrep(
            normalize=True,
            augment=split_cfg['use_augmentation'],
            tospec=True,
            tfmask=split_cfg['use_augmentation'],
            spec_params=db_cfg['spec_params'] if 'spec_params' in db_cfg else dict(),
            tfmask_params=db_cfg['tfmask_params'] if 'tfmask_params' in db_cfg else dict()
        )
    ]

    if db_cfg['name'] == 'vggsound':
        dataset = datasets.VGGSound
    elif db_cfg['name'] == 'ucf101':
        dataset = datasets.UCF
    elif db_cfg['name'] == 'hmdb51':
        dataset = datasets.HMDB
    else:
        raise ValueError('Unknown dataset')

    if db_cfg['use_rest_classes']:
        # 1) extract proper class names
        class_names = []
        if db_cfg['name'] == 'ucf101':
            with open('datasets/rest_classes/ucf_rest_classes.txt', 'r') as f:
                for line in f.readlines():
                    class_names.append(''.join([w.capitalize() for w in line.strip().split()]))
        elif db_cfg['name'] == 'hmdb51':
            with open('datasets/rest_classes/hmdb_rest_classes.txt', 'r') as f:
                for line in f.readlines():
                    class_names.append('_'.join(line.strip().split()))
        else:
            raise ValueError('Split in seen/unseen concepts is not supported for VGGSound dataset.')
        # 2) create 2 datasets (and dataloaders), 1 for unseen and 1 for seen concepts, respectively
        db1 = dataset(
            subset=split_cfg['split'].format(fold=fold),
            return_video=True,
            video_clip_duration=db_cfg['video_clip_duration'],
            video_fps=db_cfg['video_fps'],
            video_transform=video_transform,
            return_audio=False,
            audio_clip_duration=db_cfg['audio_clip_duration'] if 'audio_clip_duration' in db_cfg else 0,
            audio_srate=db_cfg['audio_srate'] if 'audio_srate' in db_cfg else 0,
            audio_transform=audio_transform,
            max_offsync=0.5 if split_cfg['use_augmentation'] else 0,
            return_labels=True,
            return_index=False,
            mode=split_cfg['mode'],
            clips_per_video=split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1,
            few_shot_ratio=split_cfg['few_shot_ratio'] if 'few_shot_ratio' in split_cfg else 1.,
            rest_classes=True,
            rest_names=class_names
        )

        if distributed:
            sampler1 = data.distributed.DistributedSampler(db1)
        else:
            sampler1 = None

        loader1 = data.DataLoader(
            db1,
            batch_size=db_cfg['batch_size'] if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size'] //
                                                                                    split_cfg['clips_per_video']),
            shuffle=(sampler1 is None) and split_cfg['use_shuffle'],
            drop_last=split_cfg['drop_last'] if 'drop_last' in split_cfg else True,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler1
        )

        db2 = dataset(
            subset=split_cfg['split'].format(fold=fold),
            return_video=True,
            video_clip_duration=db_cfg['video_clip_duration'],
            video_fps=db_cfg['video_fps'],
            video_transform=video_transform,
            return_audio=False,
            audio_clip_duration=db_cfg['audio_clip_duration'] if 'audio_clip_duration' in db_cfg else 0,
            audio_srate=db_cfg['audio_srate'] if 'audio_srate' in db_cfg else 0,
            audio_transform=audio_transform,
            max_offsync=0.5 if split_cfg['use_augmentation'] else 0,
            return_labels=True,
            return_index=False,
            mode=split_cfg['mode'],
            clips_per_video=split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1,
            few_shot_ratio=split_cfg['few_shot_ratio'] if 'few_shot_ratio' in split_cfg else 1.,
            rest_classes=False,
            rest_names=class_names
        )

        if distributed:
            sampler2 = data.distributed.DistributedSampler(db2)
        else:
            sampler2 = None

        loader2 = data.DataLoader(
            db2,
            batch_size=db_cfg['batch_size'] if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size'] //
                                                                                    split_cfg['clips_per_video']),
            shuffle=(sampler2 is None) and split_cfg['use_shuffle'],
            drop_last=split_cfg['drop_last'] if 'drop_last' in split_cfg else True,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler2
        )

        return loader1, loader2

    else:
        db = dataset(
            subset=split_cfg['split'].format(fold=fold),
            return_video=True,
            video_clip_duration=db_cfg['video_clip_duration'],
            video_fps=db_cfg['video_fps'],
            video_transform=video_transform,
            return_audio=False,
            audio_clip_duration=db_cfg['audio_clip_duration'] if 'audio_clip_duration' in db_cfg else 0,
            audio_srate=db_cfg['audio_srate'] if 'audio_srate' in db_cfg else 0,
            audio_transform=audio_transform,
            max_offsync=0.5 if split_cfg['use_augmentation'] else 0,
            return_labels=True,
            return_index=False,
            mode=split_cfg['mode'],
            clips_per_video=split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1,
            few_shot_ratio=split_cfg['few_shot_ratio'] if 'few_shot_ratio' in split_cfg else 1.
        )

        if distributed:
            sampler = data.distributed.DistributedSampler(db)
        else:
            sampler = None

        loader = data.DataLoader(
            db,
            batch_size=db_cfg['batch_size'] if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size'] //
                                                                                    split_cfg['clips_per_video']),
            shuffle=(sampler is None) and split_cfg['use_shuffle'],
            drop_last=split_cfg['drop_last'] if 'drop_last' in split_cfg else True,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler
        )

        return loader


def build_dataloaders(cfg, fold, num_workers, distributed, logger):
    if cfg['use_rest_classes']:
        train_dl_rest, train_dl_seen = build_dataloader(cfg, cfg['train'], fold, num_workers, distributed)
        logger.add_line("=" * 30 + "   Train DB (rest)   " + "=" * 30)
        logger.add_line(str(train_dl_rest.dataset))
        logger.add_line("=" * 30 + "   Train DB (seen)   " + "=" * 30)
        logger.add_line(str(train_dl_seen.dataset))

        test_dl_rest, test_dl_seen = build_dataloader(cfg, cfg['test'], fold, num_workers, distributed)
        logger.add_line("=" * 30 + "   Test DB (rest)   " + "=" * 30)
        logger.add_line(str(test_dl_rest.dataset))
        logger.add_line("=" * 30 + "   Test DB (seen)   " + "=" * 30)
        logger.add_line(str(test_dl_seen.dataset))

        dense_dl_rest, dense_dl_seen = build_dataloader(cfg, cfg['test_dense'], fold, num_workers, distributed)
        logger.add_line("=" * 30 + "   Dense DB (rest)  " + "=" * 30)
        logger.add_line(str(dense_dl_rest.dataset))
        logger.add_line("=" * 30 + "   Dense DB (seen)  " + "=" * 30)
        logger.add_line(str(dense_dl_seen.dataset))

        return train_dl_rest, train_dl_seen, test_dl_rest, test_dl_seen, dense_dl_rest, dense_dl_seen

    else:
        logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
        train_loader = build_dataloader(cfg, cfg['train'], fold, num_workers, distributed)
        logger.add_line(str(train_loader.dataset))

        logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
        test_loader = build_dataloader(cfg, cfg['test'], fold, num_workers, distributed)
        logger.add_line(str(test_loader.dataset))

        logger.add_line("=" * 30 + "   Dense DB   " + "=" * 30)
        dense_loader = build_dataloader(cfg, cfg['test_dense'], fold, num_workers, distributed)
        logger.add_line(str(dense_loader.dataset))

        return train_loader, test_loader, dense_loader


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.
        self.rank = rank

    def save(self, model, optimizer, scheduler, epoch, eval_metric=0.):
        if self.rank is not None and self.rank != 0:
            return None
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        self.save_checkpoint(state=dict(
            epoch=epoch + 1, state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()
        ), is_best=is_best, model_dir=self.checkpoint_dir)

    @staticmethod
    def save_checkpoint(state, is_best, model_dir='.', filename=None):
        if filename is None:
            filename = '{}/checkpoint.pth.tar'.format(model_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '{}/model_best.pth.tar'.format(model_dir))

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, scheduler, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        model.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])

        return start_epoch


class ClassificationWrapper(nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_name, feat_dim,
                 pooling_op=None, use_dropout=False, dropout=0.5):
        super(ClassificationWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.feat_name = feat_name
        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if pooling_op is not None:
            self.pooling = eval('torch.nn.' + pooling_op)
        else:
            self.pooling = None
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, *inputs):
        emb = self.feature_extractor(*inputs, return_embs=True)[self.feat_name]
        if self.pooling is not None:
            emb = self.pooling(emb)
        emb = emb.view(inputs[0].shape[0], -1)
        if self.dropout is not None:
            emb = self.dropout(emb)

        return self.classifier(emb)  # return logits


class Classifier(nn.Module):
    def __init__(self, n_classes, feat_name, feat_dim, pooling,
                 l2_norm=False, use_bn=True, use_dropout=False):
        super(Classifier, self).__init__()
        self.feat_name = feat_name
        self.l2_norm = l2_norm
        self.pooling = eval('torch.nn.' + pooling) if pooling is not None else None
        self.bn = nn.BatchNorm1d(feat_dim) if use_bn else None
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            if self.dropout is not None:
                x = self.dropout(x)
            if self.l2_norm:
                x = F.normalize(x, p=2, dim=-1)
            if self.pooling is not None and len(x.shape) > 2:
                x = self.pooling(x)
            x = x.view(x.shape[0], -1).contiguous().detach()
        if self.bn is not None:
            x = self.bn(x)

        return self.classifier(x)


class MOSTModel(nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_names, feat_dims, pooling_ops,
                 l2_norm=None, use_bn=False, use_dropout=False):
        super(MOSTModel, self).__init__()
        assert len(feat_dims) == len(pooling_ops) == len(feat_names)
        n_outputs = len(feat_dims)
        self.feat_names = feat_names
        self.feat_dims = feat_dims
        self.pooling_ops = pooling_ops
        if l2_norm is None:
            l2_norm = [False] * len(feat_names)
        if not isinstance(l2_norm, list):
            l2_norm = [l2_norm] * len(feat_names)
        self.l2_norm = l2_norm
        if not isinstance(use_bn, list):
            use_bn = [use_bn] * len(feat_names)

        feature_extractor.train(False)
        self.feature_extractor = feature_extractor

        self.classifiers = nn.ModuleList([Classifier(n_classes, feat_names[i], feat_dims[i], pooling_ops[i], l2_norm[i],
                                                     use_bn[i], use_dropout) for i in range(n_outputs)])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False  # avoid computing unnecessary gradients

    def forward(self, *x):
        with torch.no_grad():
            embs = self.feature_extractor(*x, return_embs=self.feat_names)
            embs = {ft: embs[ft] for ft in self.feat_names}

        for classifier, ft in zip(self.classifiers, self.feat_names):
            embs[ft] = classifier(embs[ft])

        return embs


class LinearModel(nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_dim):
        super(LinearModel, self).__init__()
        feature_extractor.train(False)  # in eval mode
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        if isinstance(feat_dim, (list, tuple)):
            feat_dim = feat_dim[-1]
        self.classifiers = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            emb = self.feature_extractor(x, return_embs=False)

        emb = self.classifiers(emb.squeeze())

        return {'pool': emb}


class MOSTCheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.best_metric = 0.

    def save(self, model, optimizer, epoch, eval_metric=0.):
        if self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        try:
            state_dict = model.classifiers.state_dict()
        except AttributeError:
            state_dict = model.module.classifiers.state_dict()

        self.save_checkpoint(state=dict(epoch=epoch, state_dict=state_dict, optimizer=optimizer.state_dict()),
                             is_best=is_best, model_dir=self.checkpoint_dir)

    @staticmethod
    def save_checkpoint(state, is_best, model_dir='.', filename=None):
        if filename is None:
            filename = '{}/checkpoint.pth.tar'.format(model_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '{}/model_best.pth.tar'.format(model_dir))

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        if last:
            return self.last_checkpoint_fn()
        elif best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        try:
            model.classifiers.load_state_dict(ckp['state_dict'])
        except AttributeError:
            model.module.classifiers.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        return start_epoch


def build_model(feat_cfg, eval_cfg, eval_dir, args, logger, **kwargs):
    import models
    from utils import main_utils
    pretrained_net = models.__dict__[feat_cfg['arch']](**feat_cfg['args'])

    # load from checkpoint
    if not args.random_weights:
        checkpoint_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
        ckp = torch.load(checkpoint_fn, map_location='cpu')
        pretrained_net.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})

    # wrap with linear-head classifiers
    if args.use_rest_classes:
        # seen/unseen split: 2 models
        model_rest = LinearModel(feature_extractor=pretrained_net.video_model, n_classes=kwargs['n_cls_rest'],
                                 feat_dim=eval_cfg['model']['args']['feat_dims'])
        model_seen = LinearModel(feature_extractor=pretrained_net.video_model, n_classes=kwargs['n_cls_seen'],
                                 feat_dim=eval_cfg['model']['args']['feat_dims'])
        os.makedirs(eval_dir + '/rest', exist_ok=True)
        ckp_manager_rest = MOSTCheckpointManager(eval_dir + '/rest', rank=args.gpu)
        os.makedirs(eval_dir + '/seen', exist_ok=True)
        ckp_manager_seen = MOSTCheckpointManager(eval_dir + '/seen', rank=args.gpu)

        logger.add_line('=' * 30 + '   Model (rest)  ' + '=' * 30)
        logger.add_line(str(model_rest))
        logger.add_line('=' * 30 + '   Parameters (rest)   ' + '=' * 30)
        logger.add_line(main_utils.parameter_description(model_rest))
        logger.add_line('=' * 30 + '   Model (seen)  ' + '=' * 30)
        logger.add_line(str(model_seen))
        logger.add_line('=' * 30 + '   Parameters (seen)   ' + '=' * 30)
        logger.add_line(main_utils.parameter_description(model_seen))
        logger.add_line('=' * 30 + '   Pretrained model   ' + '=' * 30)
        if not args.random_weights:
            logger.add_line('File: {}\nEpoch: {}'.format(checkpoint_fn, ckp['epoch']))
        else:
            logger.add_line('No checkpoint loaded for feature extraction. '
                            'Starting from a randomly initialized network.')
        # distribute
        model_rest, model_seen = distribute_model_to_cuda([model_rest, model_seen], args, eval_cfg)

        return model_rest, model_seen, ckp_manager_rest, ckp_manager_seen
    else:
        if eval_cfg['model']['name'] == 'ClassificationWrapper':
            model = ClassificationWrapper(feature_extractor=pretrained_net.video_model, **eval_cfg['model']['args'])
            ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
        elif eval_cfg['model']['name'] == 'MOSTWrapper':
            model = MOSTModel(feature_extractor=pretrained_net.video_model, **eval_cfg['model']['args'])
            ckp_manager = MOSTCheckpointManager(eval_dir, rank=args.gpu)
        else:
            raise ValueError('Unknown name (expected: ClassificationWrapper, MOSTWrapper)')

        # log model description
        logger.add_line('=' * 30 + '   Model   ' + '=' * 30)
        logger.add_line(str(model))
        logger.add_line('=' * 30 + '   Parameters   ' + '=' * 30)
        logger.add_line(main_utils.parameter_description(model))
        logger.add_line('=' * 30 + '   Pretrained model   ' + '=' * 30)
        if not args.random_weights:
            logger.add_line('File: {}\nEpoch: {}'.format(checkpoint_fn, ckp['epoch']))
        else:
            logger.add_line('No checkpoint loaded for feature extraction. '
                            'Starting from a randomly initialized network.')

        # distribute
        model = distribute_model_to_cuda(model, args, eval_cfg)

        return model, ckp_manager


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            outs += [self.model(x[i:i + self.batch_size])]

        return torch.cat(outs, 0)


class BatchWrapper2:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        from collections import defaultdict
        outs = defaultdict(list)
        for i in range(0, x.shape[0], self.batch_size):
            out_dict = self.model(x[i: i + self.batch_size])
            for k in out_dict:
                outs[k] += [out_dict[k]]

        for k in outs:
            outs[k] = torch.cat(outs[k], 0)

        return outs
