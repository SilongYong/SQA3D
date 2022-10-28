# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Bongard-HOI library
# which was released under the NVIDIA Source Code Licence.
#
# Source:
# https://github.com/NVlabs/Bongard-HOI
#
# The license for the original version of this file can be
# found in https://github.com/NVlabs/Bongard-HOI/blob/master/LICENSE
# The modifications to this file are subject to the same NVIDIA Source Code Licence.
# ---------------------------------------------------------------

import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR

from typing import Any
import sys
from . import relvit

import glob
import PIL.Image
from torchvision import transforms
import torch.distributed as dist
import torch.nn as nn
import functools
from copy import deepcopy

_log_path = None

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""

import multiprocessing
from torch import Tensor
from typing import Optional, Iterable, Any, List, Union, Tuple


def div(numerator: Tensor, denom: Union[Tensor, int, float]) -> Tensor:
    """Handle division by zero"""
    if type(denom) in [int, float]:
        if denom == 0:
            return torch.zeros_like(numerator)
        else:
            return numerator / denom
    elif type(denom) is Tensor:
        zero_idx = torch.nonzero(denom == 0).squeeze(1)
        denom[zero_idx] += 1e-8
        return numerator / denom
    else:
        raise TypeError("Unsupported data type ", type(denom))


def hico_logit_conversion_to_hoi(logits_verb, logits_object, corr):
    # logits_verb: (B, 117)
    # logits_object: (B, 80)

    # FIXME: how to compute the joint prob properly
    with torch.no_grad():
        B = logits_object.size(0)
        logits_hoi = torch.zeros(B, 600).to(logits_verb)
        for ind, c in enumerate(corr):
            obj, verb = c[1], c[2]
            logits_hoi[:, ind] = torch.sigmoid(logits_verb[:, verb]) * torch.sigmoid(logits_object[:, obj])
            # logits_hoi[:, ind] = logits_verb[:, verb] + logits_object[:, obj]
        return logits_hoi


class AveragePrecisionMeter:
    """
    Meter to compute average precision
    Arguments:
        num_gt(iterable): Number of ground truth instances for each class. When left
            as None, all positives are assumed to have been included in the collected
            results. As a result, full recall is guaranteed when the lowest scoring
            example is accounted for.
        algorithm(str, optional): AP evaluation algorithm
            '11P': 11-point interpolation algorithm prior to voc2010
            'INT': Interpolation algorithm with all points used in voc2010
            'AUC': Precisely as the area under precision-recall curve
        chunksize(int, optional): The approximate size the given iterable will be split
            into for each worker. Use -1 to make the argument adaptive to iterable size
            and number of workers
        precision(int, optional): Precision used for float-point operations. Choose
            amongst 64, 32 and 16. Default is 64
        output(tensor[N, K], optinoal): Network outputs with N examples and K classes
        labels(tensor[N, K], optinoal): Binary labels
    Usage:

    (1) Evalute AP using provided output scores and labels
        >>> # Given output(tensor[N, K]) and labels(tensor[N, K])
        >>> meter = pocket.utils.AveragePrecisionMeter(output=output, labels=labels)
        >>> ap = meter.eval(); map_ = ap.mean()
    (2) Collect results on the fly and evaluate AP
        >>> meter = pocket.utils.AveragePrecisionMeter()
        >>> # Compute output(tensor[N, K]) during forward pass
        >>> meter.append(output, labels)
        >>> ap = meter.eval(); map_ = ap.mean()
        >>> # If you are to start new evaluation and want to reset the meter
        >>> meter.reset()
    """
    def __init__(self, num_gt: Optional[Iterable] = None,
            algorithm: str = 'AUC', chunksize: int = -1,
            precision: int = 64,
            output: Optional[Tensor] = None,
            labels: Optional[Tensor] = None) -> None:
        self._dtype = eval('torch.float' + str(precision))
        self.num_gt = torch.as_tensor(num_gt, dtype=self._dtype) \
            if num_gt is not None else None
        self.algorithm = algorithm
        self._chunksize = chunksize

        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = torch.tensor([], dtype=self._dtype)
            self._labels = torch.tensor([], dtype=self._dtype)
        elif is_none == (False, False):
            self._output = output.detach().cpu().to(self._dtype)
            self._labels = labels.detach().cpu().to(self._dtype)
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [torch.tensor([], dtype=self._dtype)]
        self._labels_temp = [torch.tensor([], dtype=self._dtype)]

    @staticmethod
    def compute_per_class_ap_as_auc(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break
            d_x = rec[idx] - rec[idx - 1]
            # Skip when negative example is registered
            if d_x == 0:
                continue
            ap +=  prec[idx] * rec[idx] if idx == 0 \
                else 0.5 * (prec[idx] + prec[idx - 1]) * d_x
        return ap

    @staticmethod
    def compute_per_class_ap_with_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break
            d_x = rec[idx] - rec[idx - 1]
            # Skip when negative example is registered
            if d_x == 0:
                continue
            # Compute interpolated precision
            max_ = prec[idx:].max()
            ap +=  max_ * rec[idx] if idx == 0 \
                else 0.5 * (max_ + torch.max(prec[idx - 1], max_)) * d_x
        return ap

    @staticmethod
    def compute_per_class_ap_with_11_point_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        dtype = rec.dtype
        ap = 0
        for t in torch.linspace(0, 1, 11, dtype=dtype):
            inds = torch.nonzero(rec >= t).squeeze()
            if inds.numel():
                ap += (prec[inds].max() / 11)
        return ap

    @classmethod
    def compute_ap(cls, output: Tensor, labels: Tensor,
            num_gt: Optional[Tensor] = None,
            algorithm: str = 'AUC',
            chunksize: int = -1) -> Tensor:
        """
        Compute average precision under the classification setting. Scores of all
        classes are retained for each sample.
        Arguments:
            output(Tensor[N, K])
            labels(Tensor[N, K])
            num_gt(Tensor[K]): Number of ground truth instances for each class
            algorithm(str): AP evaluation algorithm
            chunksize(int, optional): The approximate size the given iterable will be split
                into for each worker. Use -1 to make the argument adaptive to iterable size
                and number of workers
        Returns:
            ap(Tensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, labels,
            num_gt=num_gt)
        ap = torch.zeros(output.shape[1], dtype=prec.dtype)
        # Use the logic from pool._map_async to compute chunksize
        # https://github.com/python/cpython/blob/master/Lib/multiprocessing/pool.py
        # NOTE: Inappropriate chunksize will cause [Errno 24]Too many open files
        # Make changes with caution
        if chunksize == -1:
            chunksize, extra = divmod(
                    output.shape[1],
                    multiprocessing.cpu_count() * 4)
            if extra:
                chunksize += 1

        if algorithm == 'INT':
            algorithm_handle = cls.compute_per_class_ap_with_interpolation
        elif algorithm == '11P':
            algorithm_handle = cls.compute_per_class_ap_with_11_point_interpolation
        elif algorithm == 'AUC':
            algorithm_handle = cls.compute_per_class_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

        # with multiprocessing.get_context('spawn').Pool() as pool:
            # for idx, result in enumerate(pool.imap(
            #     func=algorithm_handle,
            #     # NOTE: Use transpose instead of T for compatibility
            #     iterable=zip(prec.transpose(0,1), rec.transpose(0,1)),
            #     chunksize=chunksize
            # )):
            #     ap[idx] = algorithm_handle(prec[idx], rec[idx])
        prec = prec.transpose(0,1)
        rec = rec.transpose(0,1)
        for idx in range(len(prec)):
            ap[idx] = algorithm_handle((prec[idx], rec[idx]))

        return ap

    @staticmethod
    def compute_precision_and_recall(output: Tensor, labels: Tensor,
            num_gt: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
            output(Tensor[N, K])
            labels(Tensor[N, K])
            num_gt(Tensor[K])
        Returns:
            prec(Tensor[N, K])
            rec(Tensor[N, K])
        """
        order = output.argsort(0, descending=True)
        tp = labels[
            order,
            torch.ones_like(order) * torch.arange(output.shape[1])
        ]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        rec = div(tp, labels.sum(0)) if num_gt is None \
            else div(tp, num_gt)

        return prec, rec

    def append(self, output: Tensor, labels: Tensor) -> None:
        """
        Add new results to the meter
        Arguments:
            output(tensor[N, K]): Network output with N examples and K classes
            labels(tensor[N, K]): Binary labels
        """
        if isinstance(output, torch.Tensor) and isinstance(labels, torch.Tensor):
            assert output.shape == labels.shape, \
                "Output scores do not match the dimension of labelss"
            self._output_temp.append(output.detach().cpu().to(self._dtype))
            self._labels_temp.append(labels.detach().cpu().to(self._dtype))
        else:
            raise TypeError("Arguments should both be torch.Tensor")

    def reset(self, keep_old: bool = False) -> None:
        """
        Clear saved statistics
        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        if not keep_old:
            self._output = torch.tensor([], dtype=self._dtype)
            self._labels = torch.tensor([], dtype=self._dtype)
        self._output_temp = [torch.tensor([], dtype=self._dtype)]
        self._labels_temp = [torch.tensor([], dtype=self._dtype)]

    def eval(self) -> Tensor:
        """
        Evaluate the average precision based on collected statistics
        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        self._output = torch.cat([
            self._output,
            torch.cat(self._output_temp, 0)
        ], 0)
        self._labels = torch.cat([
            self._labels,
            torch.cat(self._labels_temp, 0)
        ], 0)
        self.reset(keep_old=True)

        # Sanity check
        if self.num_gt is not None:
            self.num_gt = self.num_gt.to(dtype=self._labels.dtype)
            faulty_cls = torch.nonzero(self._labels.sum(0) > self.num_gt).squeeze(1)
            if len(faulty_cls):
                raise AssertionError("Class {}: ".format(faulty_cls.tolist())+
                    "Number of true positives larger than that of ground truth")
        if len(self._output) and len(self._labels):
            return self.compute_ap(self._output, self._labels, num_gt=self.num_gt,
                algorithm=self.algorithm, chunksize=self._chunksize)
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for all class.")
            return torch.zeros(self._output.shape[1], dtype=self._dtype)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                       or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def make_optimizer(params, name, max_steps, lr, weight_decay=None, milestones=None, scheduler='step', use_sam=False, sam_rho=0.005, eps=1e-8, **kwargs):
    if weight_decay is None:
        weight_decay = 0.
    if use_sam:
        optimizer = SAM(params, AdamW, rho=sam_rho, lr=lr, weight_decay=weight_decay, eps=1e-08)
    else:
        if name == 'sgd':
            optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
        elif name == 'adam':
            optimizer = Adam(params, lr, weight_decay=weight_decay)
        elif name == 'adamw':
            optimizer = AdamW(
                params, float(lr), betas=(0.9, 0.999), eps=float(eps),
                weight_decay=weight_decay
            )

    update_lr_every_epoch = True
    if scheduler == 'step':
        if milestones:
            lr_scheduler = MultiStepLR(optimizer, milestones)
        else:
            lr_scheduler = None
    elif scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			lr,
			max_steps + 100,
        	pct_start=0.05,
			cycle_momentum=False,
			anneal_strategy='linear',
			final_div_factor=10000
		)
        update_lr_every_epoch = False
    elif scheduler == 'warmup_cosine':
        import pl_bolts
        lr_scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, kwargs['warmup_epochs'], kwargs['max_epochs'], warmup_start_lr=kwargs['warmup_start_lr'], eta_min=0.0, last_epoch=-1)
    return optimizer, lr_scheduler, update_lr_every_epoch

def set_lr(optimizer, lr):
    s = optimizer.state_dict()
    s['param_groups'][0]['lr'] = lr
    optimizer.load_state_dict(s)

def get_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

class ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
            )

    def forward(self, x):
        x = self.proj(x)
        return x

MCAN_GQA_PARAMS = {
    'FRCN_FEAT_SIZE': (100, 2048),
    'GRID_FEAT_SIZE': (49, 2048),
    'BBOX_FEAT_SIZE': (100, 5),
    'BBOXFEAT_EMB_SIZE': 2048,
    'HIDDEN_SIZE': 512,
    'FLAT_MLP_SIZE': 512,
    'FLAT_GLIMPSES': 1,
    'FLAT_OUT_SIZE': 1024,
    'DROPOUT_R': 0.1,
    'LAYER': 6,
    'FF_SIZE': 2048,
    'MULTI_HEAD': 8,
    'WORD_EMBED_SIZE': 300,
    'TOKEN_SIZE': 2933,
    'WORD_EMBED_SIZE': 300,
    'ANSWER_SIZE': 1843,
    'MAX_TOKEN_LENGTH': 29,
    'USE_BBOX_FEAT': True,
    'USE_AUX_FEAT': True,
}

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()

def anytype2bool_dict(s):
    # check str
    if not isinstance(s, str):
        return s
    else:
        # try int
        try:
            ret = int(s)
        except:
            # try bool
            if s.lower() in ('true', 'false'):
                ret = s.lower() == 'true'
            # try float
            else:
                try:
                    ret = float(s)
                except:
                    ret = s
        return ret

def parse_string_to_dict(field_name, value):
    fields = field_name.split('.')
    for fd in fields[::-1]:
        res = {fd: anytype2bool_dict(value)}
        value = res
    return res

def merge_to_dicts(a, b):
    if isinstance(b, dict) and isinstance(a, dict):
        a_and_b = set(a.keys()) & set(b.keys())
        every_key = set(a.keys()) | set(b.keys())
        return {k: merge_to_dicts(a[k], b[k]) if k in a_and_b else
                   deepcopy(a[k] if k in a else b[k]) for k in every_key}
    return deepcopy(type(a)(b))

def override_cfg_from_list(cfg, opts):
    assert len(opts) % 2 == 0, 'Paired input must be provided to override config, opts: {}'.format(opts)
    for ix in range(0, len(opts), 2):
        opts_dict = parse_string_to_dict(opts[ix], opts[ix + 1])
        cfg = merge_to_dicts(cfg, opts_dict)
    return cfg

# ----------------------------------------------------------------------------

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def get_world_size() -> int:
	if not dist.is_available():
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size()


def get_rank() -> int:
	if not dist.is_available():
		return 0
	if not dist.is_initialized():
		return 0
	return dist.get_rank()


def get_local_rank() -> int:
	"""
	Returns:
		The rank of the current process within the local (per-machine) process group.
	"""
	if not dist.is_available():
		return 0
	if not dist.is_initialized():
		return 0
	assert _LOCAL_PROCESS_GROUP is not None
	return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
	"""
	Returns:
		The size of the per-machine process group,
		i.e. the number of processes per machine.
	"""
	if not dist.is_available():
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
	return get_rank() == 0


def synchronize():
	"""
	Helper function to synchronize (barrier) among all processes when
	using distributed training
	"""
	if not dist.is_available():
		return
	if not dist.is_initialized():
		return
	world_size = dist.get_world_size()
	if world_size == 1:
		return
	dist.barrier()
