import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import random
import utils.logger
from utils import main_utils, eval_utils


parser = argparse.ArgumentParser(description='Downstream evaluation (linear classification)')
parser.add_argument('cfg', metavar='CFG', help='Config file')
parser.add_argument('model_cfg', metavar='CFG', help='Model config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--random-weights', action='store_true',
                    help='Keep randomly initialized weights (instead of loading from '
                         'a checkpoint) for the feature extraction network. '
                         'Useful for experiments. Consider setting a manual seed '
                         'for reproducibility.')
parser.add_argument('--few-shot-ratio', default=1.0, type=float,
                    help='Ratio of labeled data used for training the linear classifier. '
                         'Must be in (0, 1.] range. Useful for data-efficiency '
                         'related experiments.')
parser.add_argument('--use-rest-classes', action='store_true',
                    help='This option enables the split between seen and unseen (rest) '
                         'classes (concepts) during pretraining. If specified, one linear '
                         'classifier will be trained on each type of concepts separately, '
                         'resulting in 2 trained classifiers in total. The program will '
                         'return final results on each type of concepts.')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 1
    if args.random_weights:
        cfg['random_weights'] = True
    assert 0 < args.few_shot_ratio <= 1.
    if args.few_shot_ratio < 1.:
        # set seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
    cfg['dataset']['train']['few_shot_ratio'] = args.few_shot_ratio
    cfg['dataset']['use_rest_classes'] = args.use_rest_classes

    ngpus = torch.cuda.device_count()
    if 'num_folds' in cfg['dataset']:
        for fold in range(1, cfg['dataset']['num_folds'] + 1):
            if args.distributed:
                mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, fold, args, cfg))
            else:
                main_worker(None, ngpus, fold, args, cfg)
    elif 'fold' in cfg['dataset']:
        if args.distributed:
            mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg['dataset']['fold'], args, cfg))
        else:
            main_worker(None, ngpus, cfg['dataset']['fold'], args, cfg)
    else:
        raise ValueError('Fold is not specified in config file. You can either select a specific '
                         'fold or set the total number of folds.')


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'

    # prepare folder and logger
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # dataloaders + check if the split in seen/unseen concepts is applicable here
    if args.use_rest_classes:
        train_dl_rest, train_dl_seen, test_dl_rest, test_dl_seen, dense_dl_rest, dense_dl_seen = \
            eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    else:
        train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
            cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    # model(s)
    if args.use_rest_classes:
        model_rest, model_seen, ckp_manager_rest, ckp_manager_seen = eval_utils.build_model(
            model_cfg, cfg, eval_dir, args, logger,
            n_cls_rest=train_dl_rest.dataset.num_classes, n_cls_seen=train_dl_seen.dataset.num_classes)
    else:
        model, ckp_manager = eval_utils.build_model(model_cfg, cfg, eval_dir, args, logger)

    # optimizer(s) + scheduler(s)
    if args.use_rest_classes:
        opt_rest, sched_rest = main_utils.build_optimizer(model_rest.parameters(), cfg['optimizer'], logger)
        opt_seen, sched_seen = main_utils.build_optimizer(model_seen.parameters(), cfg['optimizer'], logger)
    else:
        optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    ########## TRAIN ##########
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if args.use_rest_classes:
        if (cfg['resume'] or args.test_only) and ckp_manager_rest.checkpoint_exists(last=True) and \
                ckp_manager_seen.checkpoint_exists(last=True):
            start_epoch = ckp_manager_rest.restore(model_rest, opt_rest, sched_rest, restore_last=True)
            _ = ckp_manager_seen.restore(model_seen, opt_seen, sched_seen, restore_last=True)
            logger.add_line("Loaded both checkpoints '{}' and '{}' (epoch {})".format(
                ckp_manager_rest.last_checkpoint_fn(), ckp_manager_seen.last_checkpoint_fn(), start_epoch
            ))
    else:
        if (cfg['resume'] or args.test_only) and ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True)
            logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))

    if not cfg['test_only']:
        logger.add_line('=' * 30 + '   Training   ' + '=' * 30)
        for epoch in range(start_epoch, end_epoch):
            if args.use_rest_classes:
                sched_rest.step(epoch=epoch)
                sched_seen.step(epoch=epoch)
            else:
                scheduler.step(epoch=epoch)
            if args.distributed:
                if args.use_rest_classes:
                    train_dl_rest.sampler.set_epoch(epoch)
                    train_dl_seen.sampler.set_epoch(epoch)
                    test_dl_rest.sampler.set_epoch(epoch)
                    test_dl_seen.sampler.set_epoch(epoch)
                else:
                    train_loader.sampler.set_epoch(epoch)
                    test_loader.sampler.set_epoch(epoch)

            logger.add_line('=' * 30 + ' Epoch {} '.format(epoch) + '=' * 30)
            if args.use_rest_classes:
                logger.add_line('LR: {}'.format(sched_rest.get_last_lr()))
                logger.add_line('-' * 30 + '  Training on rest classes  ' + '-' * 30)
                run_phase('train', train_dl_rest, model_rest, opt_rest, epoch, args, cfg, logger)
                logger.add_line('-' * 30 + '  Testing on rest classes  ' + '-' * 30)
                run_phase('test', test_dl_rest, model_rest, None, epoch, args, cfg, logger)
                logger.add_line('-' * 30 + '  Training on seen classes  ' + '-' * 30)
                run_phase('train', train_dl_seen, model_seen, opt_seen, epoch, args, cfg, logger)
                logger.add_line('-' * 30 + '  Testing on seen classes  ' + '-' * 30)
                run_phase('test', test_dl_seen, model_seen, None, epoch, args, cfg, logger)
                ckp_manager_rest.save(model_rest, opt_rest, sched_rest, epoch)
                ckp_manager_seen.save(model_seen, opt_seen, sched_seen, epoch)
            else:
                logger.add_line('LR: {}'.format(scheduler.get_last_lr()))
                run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
                run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
                ckp_manager.save(model, optimizer, scheduler, epoch)

    ########## EVALUATION ##########
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    cfg['dataset']['test']['clips_per_video'] = 25
    if args.use_rest_classes:
        train_dl_rest, train_dl_seen, test_dl_rest, test_dl_seen, dense_dl_rest, dense_dl_seen = \
            eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
        top1_dense_rest, top5_dense_rest = run_phase('test_dense', dense_dl_rest, model_rest, None,
                                                     end_epoch, args, cfg, logger)
        top1_rest, top5_rest = run_phase('test', test_dl_rest, model_rest, None, end_epoch, args, cfg, logger)
        top1_dense_seen, top5_dense_seen = run_phase('test_dense', dense_dl_seen, model_seen, None,
                                                     end_epoch, args, cfg, logger)
        top1_seen, top5_seen = run_phase('test', test_dl_seen, model_seen, None, end_epoch, args, cfg, logger)
        logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
        for ft in top1_rest:
            logger.add_line('')
            logger.add_line('[{}] Clip@1 (rest): {:6.2f}'.format(ft, top1_rest[ft]))
            logger.add_line('[{}] Clip@5 (rest): {:6.2f}'.format(ft, top5_rest[ft]))
            logger.add_line('[{}] Video@1 (rest): {:6.2f}'.format(ft, top1_dense_rest[ft]))
            logger.add_line('[{}] Video@5 (rest): {:6.2f}'.format(ft, top5_dense_rest[ft]))
            logger.add_line('-' * 50)
            logger.add_line('[{}] Clip@1 (seen): {:6.2f}'.format(ft, top1_seen[ft]))
            logger.add_line('[{}] Clip@5 (seen): {:6.2f}'.format(ft, top5_seen[ft]))
            logger.add_line('[{}] Video@1 (seen): {:6.2f}'.format(ft, top1_dense_seen[ft]))
            logger.add_line('[{}] Video@5 (seen): {:6.2f}'.format(ft, top5_dense_seen[ft]))
    else:
        train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
            cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
        top1_dense, top5_dense = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)
        top1, top5 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)

        logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
        for ft in top1:
            logger.add_line('')
            logger.add_line('[{}] Clip@1: {:6.2f}'.format(ft, top1[ft]))
            logger.add_line('[{}] Clip@5: {:6.2f}'.format(ft, top5[ft]))
            logger.add_line('[{}] Video@1: {:6.2f}'.format(ft, top1_dense[ft]))
            logger.add_line('[{}] Video@5: {:6.2f}'.format(ft, top5_dense[ft]))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    feature_names = cfg['model']['args']['feat_names'] if 'feat_names' in cfg['model']['args'] else ['pool']
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', 100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', 100)
    loss_meters = {ft: metrics_utils.AverageMeter('Loss', ':.4e', 0) for ft in feature_names}
    top1_meters = {ft: metrics_utils.AverageMeter('Acc@1', ':6.2f', 0) for ft in feature_names}
    top5_meters = {ft: metrics_utils.AverageMeter('Acc@5', ':6.2f', 0) for ft in feature_names}
    progress = {'timers': utils.logger.ProgressMeter(
        len(loader), meters=[batch_time, data_time], phase=phase, epoch=epoch, logger=logger)}
    progress.update({ft: utils.logger.ProgressMeter(
        len(loader), meters=[loss_meters[ft], top1_meters[ft], top5_meters[ft]], phase=phase, epoch=epoch, logger=logger
    ) for ft in feature_names})

    # switch to train/test mode
    model.train(phase == 'train')

    if phase in ['test_dense', 'test']:
        model = eval_utils.BatchWrapper2(model, cfg['dataset']['batch_size'])

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    end = time.time()
    for itr, sample in enumerate(loader):
        data_time.update(time.time() - end)

        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
        if torch.cuda.device_count() == 1 and args.gpu is None:
            video = video.cuda()

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[:2]
            video = video.flatten(0, 1).contiguous()

        # outputs
        if phase == 'train':
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)

        # compute loss + measure accuracy
        total_loss = 0.
        for ft in feature_names:
            if phase == 'test_dense':
                confidence = softmax(logits[ft]).view(batch_size, clips_per_sample, -1).mean(1)
                # tile targets appropriately before calculating loss
                loss = criterion(logits[ft], target.unsqueeze(1).repeat(1, clips_per_sample).view(-1))
            else:
                confidence = softmax(logits[ft])
                loss = criterion(logits[ft], target)

            total_loss += loss

            with torch.no_grad():
                acc1, acc5 = metrics_utils.accuracy(confidence, target, topk=(1, 5))
                loss_meters[ft].update(loss.item(), target.size(0))
                top1_meters[ft].update(acc1[0].item(), target.size(0))
                top5_meters[ft].update(acc5[0].item(), target.size(0))

        # compute gradient + do SGD step (if necessary)
        if phase == 'train':
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (itr + 1) % 100 == 0 or itr == 0 or itr + 1 == len(loader):
            for ft in progress:
                progress[ft].display(itr + 1)

    if args.distributed:
        for ft in progress:
            progress[ft].synchronize_meters(args.gpu)
            progress[ft].display(len(loader) * args.world_size)

    return {ft: top1_meters[ft].avg for ft in feature_names}, {ft: top5_meters[ft].avg for ft in feature_names}


if __name__ == '__main__':
    main()

