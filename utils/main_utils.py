import os
import shutil
import torch
import torch.distributed as dist
import datetime
from utils.logger import Logger


def initialize_distributed_backend(args, ngpus_per_node):
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            if 'SLURM_NODEID' in list(os.environ.keys()):
                args.rank = int(os.environ['SLURM_NODEID'])
            else:
                args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank
            # needs to be the global rank among all processes
            args.rank = args.gpu + args.rank * ngpus_per_node
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.rank == -1:
        args.rank = 0

    return args


def prep_environment(args, cfg):
    # prepare loggers (must be configured after initialize_distributed_backend())
    model_dir = '{}/{}'.format(cfg['model']['model_dir'], cfg['model']['name'])
    if args.rank == 0:
        prep_output_folder(model_dir, False)
    log_fn = '{}/train.log'.format(model_dir)
    logger = Logger(quiet=args.quiet, log_fn=log_fn, rank=args.rank)

    logger.add_line(str(datetime.datetime.now()))
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
    logger.add_line("=" * 30 + "   Args   " + "=" * 30)
    for k in args.__dict__:
        logger.add_line('{:30} {}'.format(k, args.__dict__[k]))

    return logger, model_dir


def build_model(cfg, logger=None):
    import models
    assert cfg['arch'] in models.__dict__, 'Unknown model architecture'
    model = models.__dict__[cfg['arch']](**cfg['args'])

    if logger is not None:
        if isinstance(model, (list, tuple)):
            logger.add_line("=" * 30 + "   Model   " + "=" * 30)
            for m in model:
                logger.add_line(str(m))
            logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
            for m in model:
                logger.add_line(parameter_description(m))
        else:
            logger.add_line("=" * 30 + "   Model   " + "=" * 30)
            logger.add_line(str(model))
            logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
            logger.add_line(parameter_description(model))

    return model


def prep_output_folder(model_dir, evaluate):
    if evaluate:
        assert os.path.isdir(model_dir)
    else:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)


def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(p.numel())
        )

    return desc


def distribute_model_to_cuda(models, args, batch_size, num_workers, ngpus_per_node):
    if ngpus_per_node == 0:
        return models, args, batch_size, num_workers
    squeeze = False
    if not isinstance(models, list):
        models = [models]
        squeeze = True

    for i in range(len(models)):
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope. Otherwise, it will use all
            # available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                models[i].cuda(args.gpu)
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu])
            else:
                models[i].cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i])
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            models[i] = models[i].cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            models[i] = torch.nn.DataParallel(models[i]).cuda()

    if squeeze:
        models = models[0]

    if args.distributed and args.gpu is not None:
        # When using a single GPU per process and per DistributedDataParallel,
        # we need to divide the batch size ourselves based on the total number of available GPUs
        batch_size = int(batch_size / ngpus_per_node)
        num_workers = int((num_workers + ngpus_per_node - 1) / ngpus_per_node)

    return models, args, batch_size, num_workers


def build_dataloaders(cfg, num_workers, distributed, logger):
    train_loader = build_dataloader(cfg, cfg['train'], num_workers, distributed)
    logger.add_line("\n" + "=" * 30 + "   Train data   " + "=" * 30)
    logger.add_line(str(train_loader.dataset))
    return train_loader


def build_dataloader(db_cfg, split_cfg, num_workers, distributed):
    import torch.utils.data as data
    import torch.utils.data.distributed
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
    )

    # Audio transforms
    audio_transform = [
        preprocessing.AudioPrep(
            normalize=True,
            augment=split_cfg['use_augmentation'],
            tospec=True,
            tfmask=split_cfg['use_augmentation'],
            spec_params=db_cfg['spec_params'],
            tfmask_params=db_cfg['tfmask_params'] if 'tfmask_params' in db_cfg else dict()
        )
    ]

    if db_cfg['name'] == 'vggsound':
        dataset = datasets.VGGSound
    else:
        raise ValueError('Unknown dataset')

    clips_per_video = split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1

    db = dataset(
        subset=split_cfg['split'],
        return_video=True,
        video_clip_duration=db_cfg['video_clip_duration'],
        video_fps=db_cfg['video_fps'],
        video_transform=video_transform,
        return_audio=True,
        audio_clip_duration=db_cfg['audio_clip_duration'],
        audio_srate=db_cfg['audio_srate'],
        audio_transform=audio_transform,
        max_offsync=0.5 if split_cfg['use_augmentation'] else 0,
        return_labels=False if 'return_labels' not in split_cfg else split_cfg['return_labels'],
        return_index=True,
        mode='clip',
        clips_per_video=clips_per_video,
        video_only=split_cfg['video_only'] if 'video_only' in split_cfg else False
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size'],
        shuffle=(sampler is None),
        drop_last=split_cfg['drop_last'],
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )

    return loader


def build_criterion(cfg, logger=None):
    import criterions
    criterion = criterions.__dict__[cfg['name']](**cfg['args'])
    if logger is not None:
        logger.add_line(str(criterion))

    return criterion


def build_optimizer(params, cfg, logger=None):
    if cfg['name'] == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg['lr']['base_lr'],
            betas=cfg['betas'] if 'betas' in cfg else [0.9, 0.999],
            weight_decay=cfg['weight_decay']
        )
    else:
        raise ValueError('Unknown optimizer')

    # decay lr by gamma once the number of epochs reaches one of the milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr']['milestones'],
                                                     gamma=cfg['lr']['gamma'])
    return optimizer, scheduler


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.best_metric = 0.

    def save(self, epoch, filename=None, eval_metric=0., **kwargs):
        if self.rank != 0:
            return

        is_best = False
        if eval_metric > self.best_metric:
            is_best = True
            self.best_metric = eval_metric

        state = dict(epoch=epoch)
        for k in kwargs:
            state[k] = kwargs[k].state_dict()

        if filename is None:
            self.save_checkpoint(state=state, is_best=is_best, model_dir=self.checkpoint_dir)
        else:
            self.save_checkpoint(state=state, is_best=False, filename='{}/{}'.format(self.checkpoint_dir, filename))

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

    def restore(self, fn=None, restore_last=False, restore_best=False, **kwargs):
        checkpoint_fn = fn if fn is not None else self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})  # map tensors from GPU 0 to CPU
        start_epoch = ckp['epoch']
        for k in kwargs:
            if k == 'train_criterion':
                kwargs[k].load_state_dict(ckp[k], strict=False)
            else:
                kwargs[k].load_state_dict(ckp[k])

        return start_epoch


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.top1_max = 0

    def __call__(self, top1_acc):
        if top1_acc < self.top1_max:
            self.counter += 1
        else:
            self.counter = 0
            self.top1_max = top1_acc

        return True if self.patience <= self.counter else False

