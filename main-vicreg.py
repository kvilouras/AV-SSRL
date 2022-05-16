import argparse
import os
import random
import time
import yaml
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp

import utils.logger
from utils import main_utils

parser = argparse.ArgumentParser(description="Self-supervised pretraining on VGGSound subset (VICReg)")
parser.add_argument('cfg', help='Model directory')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--world-size', default=-1, type=int,
                    help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='Node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='URL used to set up distributed training')
parser.add_argument('--dist-backend',default='nccl', type=str,
                    help='Distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='Seed to initialize training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training '
                         'to launch N processes per node, which has '
                         'N GPUs. It is the fastest way to use PyTorch '
                         'for either single-node or multi-node data-parallel training')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Seeded training is enabled. This will slow down training.')

    if args.gpu is not None:
        warnings.warn('A specific GPU is chosen. This will completely disable data parallelism!')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # adjust world_size (we have N GPUs per node)
        args.world_size = ngpus_per_node * args.world_size
        # launch distributed processes
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu

    # setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node)
    logger, model_dir = main_utils.prep_environment(args, cfg)

    # define model
    model = main_utils.build_model(cfg['model'], logger)
    model, args, cfg['dataset']['batch_size'], cfg['num_workers'] = distribute_to_cuda(
        model, args, cfg['dataset']['batch_size'], cfg['num_workers'], ngpus_per_node
    )

    # define dataloaders
    train_loader = main_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], args.distributed, logger)

    # define criterion
    # device = args.gpu if args.gpu is not None else 0
    cfg['loss']['args']['batch_size'] = cfg['dataset']['batch_size'] * args.world_size  # effective batch size
    train_criterion = main_utils.build_criterion(cfg['loss'], logger)

    # define optimizer
    optimizer, scheduler = main_utils.build_optimizer(
        params=list(model.parameters()) + list(train_criterion.parameters()),
        cfg=cfg['optimizer'],
        logger=logger
    )

    # checkpoint manager
    ckp_manager = main_utils.CheckpointManager(model_dir, rank=args.rank)

    # optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume']:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(restore_last=True, model=model, optimizer=optimizer,
                                              train_criterion=train_criterion)
            scheduler.step(start_epoch)
            logger.add_line("Checkpoint loaded '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))
        else:
            logger.add_line("No checkpoint found at '{}'".format(ckp_manager.last_checkpoint_fn()))

    cudnn.benchmark = True

    ######### TRAIN #########
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, end_epoch):
        if epoch in cfg['optimizer']['lr']['milestones']:
            ckp_manager.save(epoch, model=model, optimizer=optimizer, train_criterion=train_criterion,
                             filename='checkpoint-ep{}.pth.tar'.format(epoch))
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        scheduler.step(epoch)

        # train for 1 epoch
        logger.add_line('=' * 30 + ' Epoch {} '.format(epoch) + '=' * 30)
        logger.add_line('LR: {}'.format(scheduler.get_last_lr()))
        run_phase('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, scaler)
        if epoch % test_freq == 0 or epoch == end_epoch - 1:
            ckp_manager.save(epoch + 1, model=model, optimizer=optimizer, train_criterion=train_criterion)


def run_phase(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger, scaler):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.3e')
    progress = utils.logger.ProgressMeter(len(loader), [batch_time, data_time, loss_meter],
                                          phase=phase, epoch=epoch, logger=logger)

    # switch to train mode
    model.train(phase == 'train')

    end = time.time()
    device = args.gpu if args.gpu is not None else 0
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # prepare batch
        video, audio = sample['frames'], sample['audio']
        video = video.cuda(device, non_blocking=True)
        audio = audio.cuda(device, non_blocking=True)

        # compute video/audio embeddings
        if phase == 'train':
            with torch.cuda.amp.autocast():
                video_emb, audio_emb = model(video, audio)
        else:
            with torch.no_grad():
                video_emb, audio_emb = model(video, audio)

        # compute loss
        with torch.cuda.amp.autocast():
            loss = criterion(video_emb, audio_emb)
        loss_meter.update(loss.item(), video.size(0))

        # compute gradients + do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print
        if (i + 1) % cfg['print_freq'] == 0 or i == 0 or i + 1 == len(loader):
            progress.display(i + 1)

    # sync metrics across all GPUs and print final averages
    if args.distributed:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size)


def distribute_to_cuda(models, args, batch_size, num_workers, ngpus_per_node):
    if ngpus_per_node == 0:
        return models, args, batch_size, num_workers
    squeeze = False
    if not isinstance(models, list):
        models = [models]
        squeeze = True
    for i in range(len(models)):
        if args.distributed:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                models[i].cuda(args.gpu)
                # compute batch norm statistics across all GPUs
                models[i] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models[i])
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu])
            else:
                models[i].cuda()
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i])
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            models[i] = models[i].cuda(args.gpu)
        else:
            models[i] = torch.nn.DataParallel(models[i]).cuda()
    if squeeze:
        models = models[0]
    if args.distributed and args.gpu is not None:
        batch_size = int(batch_size / ngpus_per_node)
        num_workers = int((num_workers + ngpus_per_node - 1) / ngpus_per_node)

    return models, args, batch_size, num_workers


if __name__ == '__main__':
    main()
