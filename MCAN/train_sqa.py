# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for RelViT. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import argparse
import os
from re import L
import yaml
import numpy as np
import random
import time
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from tensorboardX import SummaryWriter
from easydict import EasyDict
from tqdm import tqdm

from load_data import DataSet 
import models
from models.mcan import MCANCustomized
import utils


def compute_acc_sqa(final_y_true, final_y_pred):
    count = len(final_y_true)
    correct = (final_y_true == final_y_pred).sum()
    return (correct / count).item()

def gather_score_label(dist, loader, current_idx, y_true, y_pred, ddp=False):
    if ddp:
        # all_gather and compute Acc
        current_idx_list = [torch.zeros(1).long().cuda(non_blocking=True) for _ in range(args.world_size)]
        y_true_list = [torch.zeros((len(loader.dataset), 1)).long().cuda(non_blocking=True) for _ in range(args.world_size)]
        y_pred_list = [torch.zeros((len(loader.dataset), 1)).long().cuda(non_blocking=True) for _ in range(args.world_size)]
        dist.all_gather(current_idx_list, current_idx)
        dist.all_gather(y_true_list, y_true)
        dist.all_gather(y_pred_list, y_pred)

        final_y_true = np.empty((0, 1))
        final_y_pred = np.empty((0, 1))
        for idx, yt, ys in zip(current_idx_list, y_true_list, y_pred_list):
            idx = idx.item()
            yt = yt.detach().cpu().numpy()
            ys = ys.detach().cpu().numpy()
            final_y_true = np.vstack((final_y_true, yt[:idx]))
            final_y_pred = np.vstack((final_y_pred, ys[:idx]))
    else:
        final_y_true = y_true.detach().cpu().numpy()[:current_idx.item()]
        final_y_pred = y_pred.detach().cpu().numpy()[:current_idx.item()]
    return final_y_true, final_y_pred


def main(config):
    args.gpu = ''#[i for i in range(torch.cuda.device_count())]
    args.train_gpu = [i for i in range(torch.cuda.device_count())]
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus - 1):
        args.gpu += '{},'.format(i)
    args.gpu += '{}'.format(num_gpus - 1)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    utils.set_gpu(args.gpu)
    args.config = config

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.sync_bn = True
        if args.dist_url[-2:] == '{}':
            port = utils.find_free_port()
            args.dist_url = args.dist_url.format(port)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    config = args.config
    svname = args.svname
    if svname is None:
        config_name, _ = os.path.splitext(os.path.basename(args.config_file))
        svname = 'hico'
        svname += '_' + config['model']
        if config['model_args'].get('encoder'):
            svname += '-' + config['model_args']['encoder']
        svname = os.path.join(config_name, config['train_dataset'], svname)
    if not args.test_only:
        svname += '-seed' + str(args.seed)
    if args.tag is not None:
        svname += '_' + args.tag

    sub_dir_name = 'default'
    if args.opts:
        sub_dir_name = args.opts[0]
        split = '#'
        for opt in args.opts[1:]:
            sub_dir_name += split + opt
            split = '#' if split == '_' else '_'
    svname = os.path.join(svname, sub_dir_name)

    if utils.is_main_process() and not args.test_only:
        save_path = os.path.join(args.save_dir, svname)
        utils.ensure_path(save_path, remove=False)
        utils.set_log_path(save_path)
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
        args.writer = writer

        yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

        logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)
    else:
        save_path = None
        writer = None
        args.writer = writer
        logger = None

    #### Dataset ####

    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    dataset_flags = EasyDict()
    dataset_flags.anno_dir = config["anno_dir"]
    dataset_flags.img_dir = config["img_dir"]
    dataset_flags.USE_GLOVE = config["USE_GLOVE"]
    dataset_flags.MAX_TOKEN = config["MAX_TOKEN"]

    train_dataset = DataSet(dataset_flags, 'train')
    pretrained_emb = train_dataset.pretrained_emb
    token_size = train_dataset.token_size 
    if utils.is_main_process():
        utils.log('train dataset: {} samples'.format(len(train_dataset)))
    if args.distributed:
        args.batch_size = int(ep_per_batch / args.world_size)
        args.batch_size_test = int(ep_per_batch / args.world_size)
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    else:
        args.batch_size = ep_per_batch
        args.batch_size_test = ep_per_batch
        args.workers = args.workers

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    print(args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    # testing
    test_dataset = DataSet(dataset_flags, 'val')
    if utils.is_main_process():
        utils.log('test dataset: {} samples'.format(len(test_dataset)))
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, # args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    ########

    #### Model and optimizer ####

    if config.get('load'):
        print('loading pretrained model: ', config['load'])
        model = models.load(torch.load(config['load']))
    else:
        model = MCANCustomized(pretrained_emb, token_size, **config['model_args'])  

        if config.get('load_encoder'):
            pretrain = config.get('encoder_pretrain').lower()
            if pretrain != 'scratch':
                pretrain_model_path = config['load_encoder'].format(pretrain)
                state_dict = torch.load(pretrain_model_path, map_location='cpu')
                missing_keys, unexpected_keys = model.encoder.encoder.load_state_dict(state_dict, strict=False)
                for key in missing_keys:
                    assert key.startswith('g_mlp.') \
                        or key.startswith('proj') \
                        or key.startswith('trans') \
                        or key.startswith('roi_processor') \
                        or key.startswith('roi_dim_processor') \
                        or key.startswith('classifier'), key
                for key in unexpected_keys:
                    assert key.startswith('fc.')
                if utils.is_main_process():
                    utils.log('==> Successfully loaded {} for the enocder.'.format(pretrain_model_path))

        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if utils.is_main_process():
            utils.log(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    else:
        model = torch.nn.DataParallel(model.cuda())

    if utils.is_main_process() and not args.test_only:
        utils.log('num params: {}'.format(utils.compute_n_params(model)))
        utils.log('Results will be saved to {}'.format(save_path))

    max_steps = min(len(train_loader), config['train_batches']) * config['max_epoch']
    optimizer, lr_scheduler, update_lr_every_epoch = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], max_steps, **config['optimizer_args']
    )
    assert lr_scheduler is not None
    args.update_lr_every_epoch = update_lr_every_epoch

    if args.test_only:
        filename = args.test_model
        assert os.path.exists(filename)
        ckpt = torch.load(filename, map_location='cpu')
        start_epoch = ckpt['epoch']

        sd_before_load = deepcopy(model.state_dict())
        model.load_state_dict(ckpt['state_dict'])
        sd_after_load = deepcopy(model.state_dict())
        same_keys = [k for k in sd_before_load if torch.equal(sd_before_load[k], sd_after_load[k])]
        new_keys = []
        for key in same_keys:
            new_keys.append(key)
        print('-------------------- Loaded weights --------------------')
        print(f'Weights unloaded:{new_keys}')
        print('----------------------------')
        
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        best_test_result = ckpt['best_test_result']
        if utils.is_main_process():
            utils.log('==> Sucessfully resumed from a checkpoint {}'.format(filename))
    else:
        ckpt = None
        start_epoch = 0
        best_test_result = 0.0

    ######## Training & Validation

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = best_test_result
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    if args.test_only:
        ret = test(test_loader, model, args)
        if ret is None:
            return 0
        else:
            loss_test, acc_test = ret
            if utils.is_main_process():
                print('Testing result: ', acc_test)
            return 0

    for epoch in range(start_epoch, max_epoch):

        sd_model = model.state_dict()

        epoch_log = epoch + 1
        finish = False
        while finish is False:
            model.load_state_dict(sd_model)
            if args.distributed:
                train_sampler.set_epoch(epoch)
            ret = train(train_loader, model, optimizer, lr_scheduler, epoch_log, writer, args)
            # autoresume
            if ret is None:
                return 0
            # NaN
            elif ret == -1:
                utils.synchronize()
                if utils.is_main_process():
                    utils.log('NaN detected, redo the current epoch...')
                continue
            else:
                finish = True
                loss_train, acc_train = ret
        import gc; gc.collect()
        torch.cuda.empty_cache()
        if args.update_lr_every_epoch:
            lr_scheduler.step()
        if utils.is_main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('Acc_train', acc_train, epoch_log)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch_log)

        if (epoch_log % config['save_epoch'] == 0 or epoch_log == config['max_epoch']) and utils.is_main_process():
            filename = os.path.join(save_path, 'train.pth')
            utils.log('==> Saving checkpoint to: ' + filename)
            ckpt = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'tea_k_state_dict': model.state_dict(),
                'moco': None,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_test_result': best_test_result,
            }
            torch.save(ckpt, filename)
        acc_test = 0
        if epoch_log % config['eval_epoch'] == 0:
            ret = test(test_loader, model, args)
            if ret is None:
                return 0
            else:
                loss_test, acc_test = ret
            import gc; gc.collect()
            torch.cuda.empty_cache()
            if acc_test > best_test_result:
                best_test_result = acc_test
            if utils.is_main_process():
                utils.log('test result: loss {:.4f}, Acc: {:.4f}.'.format(loss_test, acc_test))
                writer.add_scalar('loss_test', loss_test, epoch_log)
                writer.add_scalar('Acc_test', acc_test, epoch_log)

        if utils.is_main_process():
            utils.log('Best test results so far:')
            utils.log(best_test_result)

        if acc_test > max_va and utils.is_main_process():
            max_va = acc_test
            filename = os.path.join(save_path, 'best_model.pth')
            ckpt = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'tea_k_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_test_result': best_test_result,
            }
            torch.save(ckpt, filename)
        if utils.is_main_process():
            writer.flush()

    if utils.is_main_process():
        logger.close()

def train(train_loader, model, optimizer, lr_scheduler, epoch, writer, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    # Acc
    current_idx = torch.zeros(1).long()
    y_true = torch.zeros((len(train_loader.dataset), 1)).long()
    y_pred = torch.zeros((len(train_loader.dataset), 1)).long()

    config = args.config

    # train
    model.train()
    model.module.embedding.requires_grad_(requires_grad=False)
    for n, p in model.named_parameters():
        if p.requires_grad and "embedding" in n:
            print(n)
    if utils.is_main_process():
        args.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    end = time.time()
    max_iter = config['max_epoch'] * len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= config['train_batches']:
            break

        feed_list = []
        ims = batch[0].cuda(non_blocking=True)
        feed_list.append(ims)

        ques_ix_iter = batch[1].cuda(non_blocking=True)
        feed_list.append(ques_ix_iter)
        ans_iter = batch[2].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(*feed_list)
            loss = F.cross_entropy(logits, ans_iter.squeeze(-1), reduction='sum')
            
            y_pred[current_idx.item():current_idx.item()+logits.size(0)] = logits.detach().argmax(-1).unsqueeze(-1).cpu()
            y_true[current_idx.item():current_idx.item()+logits.size(0)] = ans_iter.detach().cpu()
            current_idx += logits.size(0)
        if torch.isnan(loss).all().item():
            utils.synchronize()
            return -1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        optimizer.step()
        lrs = lr_scheduler.get_last_lr()
        if not args.update_lr_every_epoch:
            lr_scheduler.step()

        n = logits.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = ans_iter.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            n = count.item()
            loss = loss / n

        loss_meter.update(loss.item(), logits.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + batch_idx + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (batch_idx + 1) % config['print_freq'] == 0 and utils.is_main_process():
            utils.log(
                '{} Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'Loss {loss_meter.val:.4f} '
                'lr {lr:.6f}'.format(
                    'Train',
                    epoch, config['max_epoch'], batch_idx + 1, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    lr=lrs[0]
                )
            )
    final_y_true, final_y_pred = gather_score_label(dist, train_loader, current_idx.cuda(), y_true.cuda(), y_pred.cuda(), args.multiprocessing_distributed)
    acc = compute_acc_sqa(final_y_true, final_y_pred)
    if utils.is_main_process():
        utils.log('{} result at epoch [{}/{}]: loss {:.4f}, Acc {:.4f}.'.format('Train', epoch, config['max_epoch'], loss_meter.avg, acc))
    return loss_meter.avg, acc

def test(test_loader, model, args):
    # eval
    model.eval()

    loss_meter = utils.AverageMeter()
    # Acc
    current_idx = torch.zeros(1).long()
    y_true = torch.zeros((len(test_loader.dataset), 1)).long()
    y_pred = torch.zeros((len(test_loader.dataset), 1)).long()

    np.random.seed(0)
    count_what = 0
    count_is = 0
    count_can = 0
    count_how = 0
    count_which = 0
    count_other = 0
    right_what = 0
    right_is = 0
    right_how = 0
    right_can = 0
    right_which = 0
    right_other = 0
    for batch in tqdm(test_loader):
        feed_list = []
        ims = batch[0].cuda(non_blocking=True)
        feed_list.append(ims)
        ques_ix_iter = batch[1].cuda(non_blocking=True)
        feed_list.append(ques_ix_iter)
        ans_iter = batch[2].cuda(non_blocking=True)
        q_type = batch[3]
        if q_type.item() == 0:
            count_what += 1
        elif q_type.item() == 1:
            count_is += 1
        elif q_type.item() == 2:
            count_how += 1
        elif q_type.item() == 3:
            count_can += 1
        elif q_type.item() == 4:
            count_which += 1
        elif q_type.item() == 5:
            count_other += 1
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(*feed_list)
                loss = F.cross_entropy(logits, ans_iter.squeeze(-1), reduction='sum')
                y_pred[current_idx.item():current_idx.item()+logits.size(0)] = logits.detach().argmax(-1).unsqueeze(-1).cpu()
                y_true[current_idx.item():current_idx.item()+logits.size(0)] = ans_iter.detach().cpu()
                if logits.detach().argmax(-1).unsqueeze(-1).cpu().item() == ans_iter.detach().cpu().item():
                    if q_type.item() == 0:
                        right_what += 1
                    elif q_type.item() == 1:
                        right_is += 1
                    elif q_type.item() == 2:
                        right_how += 1
                    elif q_type.item() == 3:
                        right_can += 1
                    elif q_type.item() == 4:
                        right_which += 1
                    else:
                        right_other += 1
                current_idx += logits.size(0)

        n = logits.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = logits.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        loss_meter.update(loss.item(), logits.size(0))
    final_y_true, final_y_pred = gather_score_label(dist, test_loader, current_idx.cuda(), y_true.cuda(), y_pred.cuda(), args.multiprocessing_distributed)
    acc = compute_acc_sqa(final_y_true, final_y_pred)
    print("what acc:", right_what / count_what)
    print("is acc:", right_is / count_is)
    print("how acc:", right_how / count_how)
    print("can acc:", right_can / count_can)
    print("which acc:", right_which / count_which)
    print("other acc:", right_other / count_other)
    return loss_meter.avg, acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file')
    parser.add_argument('--svname', default=None)
    parser.add_argument('--save_dir', default='./save_dist')
    parser.add_argument('--tag', default=None)
    # parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_model', default=None)

    # distributed training
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.multiprocessing_distributed = True

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    if args.opts is not None:
        config = utils.override_cfg_from_list(config, args.opts)
    print('config:')
    print(config)
    main(config)
