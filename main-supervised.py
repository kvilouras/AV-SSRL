import argparse
import os
import random
import time
import warnings
import yaml

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp

import utils.logger
from utils import main_utils

parser = argparse.ArgumentParser(description='Supervised pretraining on VGGSound')

parser.add_argument('cfg', help='Model directory')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--world_size', default=-1, type=int,
                    help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='Node rank for distributed training')
parser.add_argument('--dist-url', default="env://", type=str,
                    help='URL used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
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
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Seeded training is enabled. This will turn on '
                      'the CUDNN deterministic setting, which can '
                      'slow down training considerably! Restarting '
                      'from checkpoints might lead to unexpected behavior'
                      'as well!')

    if args.gpu is not None:
        warnings.warn('A specific GPU is chosen. This will completely disable data parallelism!')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have N GPUs per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        # simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu

    # setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node)
    logger, model_dir = main_utils.prep_environment(args, cfg)

    # define model
    model = main_utils.build_model(cfg['model'], logger)
    model, args, cfg['dataset']['batch_size'], cfg['num_workers'] = main_utils.distribute_model_to_cuda(
        model, args, cfg['dataset']['batch_size'], cfg['num_workers'], ngpus_per_node)
    # define dataloaders
    train_loader, test_loader = build_dataloaders(cfg['dataset'], cfg['num_workers'], args.distributed, logger)

    # define criterion
    device = args.gpu if args.gpu is not None else 0
    # train_criterion = main_utils.build_criterion(cfg['loss'], logger)
    train_criterion = torch.nn.CrossEntropyLoss()

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
            logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))
        else:
            logger.add_line("No checkpoint found at '{}'".format(ckp_manager.last_checkpoint_fn()))

    # enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    # works well if the network's input sizes do not vary.
    cudnn.benchmark = True

    ########## TRAIN ##########
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
    early_stop = main_utils.EarlyStopping(patience=2)  # early stopping with patience = 2 epochs
    for epoch in range(start_epoch, end_epoch):
        if epoch in cfg['optimizer']['lr']['milestones']:
            ckp_manager.save(epoch, model=model, train_criterion=train_criterion, optimizer=optimizer,
                             filename='checkpoint-ep{}.pth.tar'.format(epoch))
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        scheduler.step(epoch)

        # Train for one epoch
        logger.add_line('=' * 30 + ' Epoch {} '.format(epoch) + '=' * 30)
        logger.add_line('LR: {}'.format(scheduler.get_last_lr()))
        run_phase('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger)
        logger.add_line('=' * 30 + ' TESTING - Epoch {} '.format(epoch) + '=' * 30)
        top1 = run_phase('test', test_loader, model, optimizer, train_criterion, epoch, args, cfg, logger)
        # check if training needs to be stopped early
        flag = early_stop(top1)
        if flag:
            break
        if epoch % test_freq == 0 or epoch == end_epoch - 1 and early_stop.counter == 0:
            ckp_manager.save(epoch + 1, model=model, optimizer=optimizer, train_criterion=train_criterion)


def run_phase(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.3e')
    top1_meter = metrics_utils.AverageMeter('Acc@1', ':6.2f')
    progress = utils.logger.ProgressMeter(len(loader), [batch_time, data_time, loss_meter, top1_meter],
                                          phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    model.train(phase == 'train')

    softmax = torch.nn.Softmax(dim=1)

    end = time.time()
    device = args.gpu if args.gpu is not None else 0
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # prepare batch
        video, audio, labels = sample['frames'], sample['audio'], sample['label']
        video = video.cuda(device, non_blocking=True)
        audio = audio.cuda(device, non_blocking=True)
        labels = labels.cuda(device, non_blocking=True)

        # compute video/audio embeddings
        if phase == 'train':
            logits = model(video, audio)
        else:
            with torch.no_grad():
                logits = model(video, audio)

        confidence = softmax(logits)
        loss = criterion(logits, labels)

        # update meters
        with torch.no_grad():
            acc1 = metrics_utils.accuracy(confidence, labels, topk=(1,))[0]
            loss_meter.update(loss.item(), labels.size(0))
            top1_meter.update(acc1[0], labels.size(0))

        # compute gradients + do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    return top1_meter.avg


def build_dataloaders(cfg, num_workers, distributed, logger):
    train_loader = main_utils.build_dataloader(cfg, cfg['train'], num_workers, distributed)
    logger.add_line("\n" + "=" * 30 + "   Train data   " + "=" * 30)
    logger.add_line(str(train_loader.dataset))

    test_loader = main_utils.build_dataloader(cfg, cfg['test'], num_workers, distributed)
    logger.add_line("\n" + "=" * 30 + "   Test data   " + "=" * 30)
    logger.add_line(str(test_loader.dataset))

    return train_loader, test_loader


if __name__ == '__main__':
    main()

