# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for RelViT. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import logging
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def comp_sample_prob(size, pos, K, use_log=False):
    # case 1: size < K
    # then pos-1 > pos-2 > ... > 0
    if size < K:
        rank = np.arange(pos)+1
    # case 2: size == K
    # then (pos-1)%K > (pos-2)%K > ... > pos
    else:
        rank = np.roll(np.arange(size), pos)+1
    if use_log:
        rank = np.log(10*rank)
    assert len(rank) == size
    return torch.from_numpy(rank).float()

def dequeue_with_concept(feat, concept, queue_dict, queue_grid_dict, queue_ptr_dict, sample_uniform=True):
    # concept: (B, num_concepts)
    # If there is no enough concepts in the queue or the current sample comes without any concepts,
    # just return the corresponding feat.

    # (2*B, C), (2*B, N, C), (2*B, N, C)
    t_cls_out, t_region_out, t_fea = feat
    N = t_region_out.size(1)

    K = queue_dict[0].buffer.size(0)

    B = concept.size(0)
    with torch.no_grad():
        # sanitize samples without any concept
        concept = concept.repeat(2, 1)
        mask = (concept.sum(dim=-1) == 0)
        concept[mask] += 1
        concept_sample = torch.distributions.Categorical(concept).sample()

    ret1, ret2, ret3 = [], [], []
    for ind, (c, m) in enumerate(zip(concept_sample, mask)):
        cur_pos = queue_ptr_dict[c.item()].buffer[0].item()
        size = queue_ptr_dict[c.item()].buffer[1].item()

        if size != 0 and m == 0:
            if sample_uniform:
                # Equal prob
                pos = torch.distributions.Categorical(torch.ones(size)).sample().item()
            else:
                # "FIFO" prob
                prob = comp_sample_prob(size, cur_pos, K, use_log=True)
                pos = torch.distributions.Categorical(prob).sample().item()

            ret1.append(queue_dict[c.item()].buffer[pos])
            ret2.append(queue_grid_dict[c.item()].buffer[pos, :N])
            ret3.append(queue_grid_dict[c.item()].buffer[pos, N:])
        else:
            ret1.append(feat[0][ind])
            ret2.append(feat[1][ind])
            ret3.append(feat[2][ind])
    ret1 = torch.stack(ret1).to(t_region_out)
    ret2 = torch.stack(ret2).to(t_region_out)
    ret3 = torch.stack(ret3).to(t_region_out)
    assert ret1.shape == feat[0].shape
    assert ret2.shape == feat[1].shape
    assert ret3.shape == feat[2].shape
    return (ret1.contiguous(), ret2.contiguous(), ret3.contiguous())

def enqueue_with_concept(feat, concept, queue_dict, queue_grid_dict, queue_ptr_dict):
    # feat: (2*B, C)
    # concept: (B, num_concepts)
    # We only work on the first B instances and skip those without any concepts.

    # (2*B, C), (2*B, N, C), (2*B, N, C)
    t_cls_out, t_region_out, t_fea = feat
    N = t_region_out.size(1)

    K = queue_dict[0].buffer.size(0)

    with torch.no_grad():
        # sanitize samples without any concept
        mask = (concept.sum(dim=-1) == 0)
        concept[mask] += 1
        concept_sample = torch.distributions.Categorical(concept).sample()

    for ind, (c, m) in enumerate(zip(concept_sample, mask)):
        if m == 0:
            # write pos and size
            pos = queue_ptr_dict[c.item()].buffer[0]
            size = queue_ptr_dict[c.item()].buffer[1]
            pos += 1
            size += 1
            # write pos should loop back to 0
            pos %= K
            queue_ptr_dict[c.item()].buffer[0] = pos
            # size should be clamped to K
            size = torch.clamp(size, 0, K)
            queue_ptr_dict[c.item()].buffer[1] = size

            queue_dict[c.item()].buffer[pos.item()] = feat[0][ind].detach()
            queue_grid_dict[c.item()].buffer[pos.item(), :N] = feat[1][ind].detach()
            queue_grid_dict[c.item()].buffer[pos.item(), N:] = feat[2][ind].detach()

def RCL(attn_v_tea, attn_v_stu, concept, queue_dict, queue_grid_dict, queue_ptr_dict, center, center_grid, tau=0.04, local_only=1, tau_stu=0.1, sample_uniform=True):
    # only: 1 -- local; 2 -- global

    # FIXME: we should dequeue before enqueuing to avoid using the recently added samples.
    target = dequeue_with_concept(attn_v_tea, concept, queue_dict, queue_grid_dict, queue_ptr_dict, sample_uniform)
    enqueue_with_concept(attn_v_tea, concept, queue_dict, queue_grid_dict, queue_ptr_dict)

    return token_level_esvit(target, attn_v_stu, center, center_grid, tau, local_only, tau_stu)

def token_level_esvit(attn_v_tea, attn_v_stu, center, center_grid, tau=0.04, local_only=True, tau_stu=0.1):
    # only: 1 -- local; 2 -- global
    # FIXME: temporarily disable multi-crop
    ncrops = 2
    # (2*B, C), (2*B, N, C), (2*B, N, C)
    s_cls_out, s_region_out, s_fea = attn_v_stu
    t_cls_out, t_region_out, t_fea = attn_v_tea
    B, N = s_region_out.size(0), s_region_out.size(1)
    B = B // 2
    s_region_out = torch.cat([
        s_region_out[:B].reshape(B*N, -1),
        s_region_out[B:].reshape(B*N, -1)]
    ).contiguous()
    s_fea = torch.cat([
        s_fea[:B].reshape(B*N, -1),
        s_fea[B:].reshape(B*N, -1)]
    ).contiguous()
    t_region_out = torch.cat([
        t_region_out[:B].reshape(B*N, -1),
        t_region_out[B:].reshape(B*N, -1)]
    ).contiguous()
    t_fea = torch.cat([
        t_fea[:B].reshape(B*N, -1),
        t_fea[B:].reshape(B*N, -1)]
    ).contiguous()
    s_npatch = [N, N]
    t_npatch = [N, N]

    # teacher centering and sharpening
    temp = tau
    t_cls = F.softmax((t_cls_out - center) / temp, dim=-1)
    t_cls = t_cls.detach().chunk(2)

    t_region = F.softmax((t_region_out - center_grid) / temp, dim=-1)
    t_region = t_region.detach().chunk(2)
    t_fea = t_fea.chunk(2)


    N = t_npatch[0] # num of patches in the first view
    B = t_region[0].shape[0]//N # batch size,

    # student sharpening
    s_cls = s_cls_out / tau
    s_cls = s_cls.chunk(ncrops)

    s_region = s_region_out / tau_stu
    s_split_size = [s_npatch[0]] * 2 + [s_npatch[1]] * (ncrops -2)

    s_split_size_bs = [i * B for i in s_split_size]

    s_region = torch.split(s_region, s_split_size_bs, dim=0)
    s_fea = torch.split(s_fea, s_split_size_bs, dim=0)

    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(t_cls):
        for v in range(len(s_cls)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue

            # view level prediction loss
            loss = 0.5 * torch.sum(-q * F.log_softmax(s_cls[v], dim=-1), dim=-1)
            if local_only == 1:
                loss *= 0

            # region level prediction loss
            s_region_cur, s_fea_cur = s_region[v].view(B, s_split_size[v], -1).contiguous(), s_fea[v].view(B, s_split_size[v], -1).contiguous()  # B x T_s x K, B x T_s x P
            t_region_cur, t_fea_cur = t_region[iq].view(B, N, -1).contiguous(), t_fea[iq].view(B, N, -1).contiguous()  # B x T_t x K, B x T_t x P,

            # similarity matrix between two sets of region features
            region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1) , F.normalize(t_fea_cur, p=2, dim=-1).permute(0, 2, 1).contiguous()) # B x T_s x T_t
            region_sim_ind = region_sim_matrix.max(dim=2)[1] # B x T_s; collect the argmax index in teacher for a given student feature

            t_indexed_region = torch.gather( t_region_cur, 1, region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.size(2)) ) # B x T_s x K (index matrix: B, T_s, 1)

            loss_grid = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1]).mean(-1)   # B x T_s x K --> B

            if local_only == 2:
                loss += 0. * loss_grid
            else:
                loss += 0.5 * loss_grid

            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms

    return total_loss, t_cls_out.contiguous(), t_region_out.contiguous()

class Buffer(nn.Module):
    def __init__(self, tensor_cls, *args, **kwargs):
        super(Buffer, self).__init__()
        self.register_buffer('buffer', tensor_cls(*args, **kwargs))

class MoCo(nn.Module):
    """
        Build a MoCo model with: a query encoder, a key encoder, and a queue
        https://arxiv.org/abs/1911.05722
        """
    def __init__(self, encoder_tea_k, encoder_stu_q, K=65536, m=0.999, feat_dim=None, cl_loss='single-view', num_concepts=600, relvit_mode=1, num_tokens=49):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m

        # create the encoders
        # feature embedding size is the output fc dimension
        self.encoder_q = [encoder_stu_q]
        self.encoder_k = [encoder_tea_k]
        dim = self.encoder_q[0].out_dim if feat_dim is None else feat_dim

        for param_q, param_k in zip(self.encoder_q[0].parameters(), self.encoder_k[0].parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.relvit_mode = relvit_mode
        self.center_momentum = 0.9
        self.register_buffer("center", torch.zeros(1, dim))
        self.register_buffer("center_grid", torch.zeros(1, dim))
        if self.relvit_mode:
            # FIXME: https://github.com/microsoft/esvit/blob/main/main_esvit.py#L606
            self.register_buffer("center_rcl", torch.zeros(1, dim))
            self.register_buffer("center_grid_rcl", torch.zeros(1, dim))

        if self.relvit_mode:
            # assert self.K <= 10
            self.queue_dict = nn.ModuleList([
                Buffer(torch.rand, self.K, dim) for i in range(num_concepts)
            ])
            # FIXME: magic number; first projected and unprojected
            # 49x2 works with pvtv2b2 and swin_small
            # 196x2 works with vit_small_16
            self.queue_grid_dict = nn.ModuleList([
                Buffer(torch.rand, self.K, num_tokens*2, dim) for i in range(num_concepts)
            ])
            # current writing pos and size
            self.queue_ptr_dict = nn.ModuleList([
                Buffer(torch.zeros, 2, dtype=torch.long) for i in range(num_concepts)
            ])

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q[0].parameters(), self.encoder_k[0].parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, batch_size  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, attn_v_k, attn_v_q, B, config, args, concepts=None):
        def cal_batch(teacher_output, teacher_grid_output):
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            if args.multiprocessing_distributed:
                dist.all_reduce(batch_center)
                world_size = dist.get_world_size()
            else:
                world_size = 1
            batch_center = batch_center / (len(teacher_output) * world_size)

            # region level center update
            batch_grid_center = torch.sum(teacher_grid_output, dim=0, keepdim=True)
            if args.multiprocessing_distributed:
                dist.all_reduce(batch_grid_center)
                world_size = dist.get_world_size()
            else:
                world_size = 1
            batch_grid_center = batch_grid_center / (len(teacher_grid_output) * world_size)
            return batch_center, batch_grid_center

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        loss, teacher_output, teacher_grid_output = token_level_esvit(
            attn_v_k,
            attn_v_q,
            self.center,
            self.center_grid,
            tau=config['relvit_loss_tau'],
            local_only=config['relvit_local_only'])
        with torch.no_grad():
            # ema update for esvit
            batch_center, batch_grid_center = cal_batch(teacher_output, teacher_grid_output)
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
            self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)

        if self.relvit_mode:
            loss_rcl, teacher_output_rcl, teacher_grid_output_rcl = RCL(
                attn_v_k,
                attn_v_q,
                concepts,
                self.queue_dict,
                self.queue_grid_dict,
                self.queue_ptr_dict,
                self.center_rcl,
                self.center_grid_rcl,
                tau=config['relvit_loss_tau'],
                local_only=config['relvit_local_only'],
                sample_uniform=config['relvit_sample_uniform'])

            if self.relvit_mode == 2:
                loss = loss * 0 + loss_rcl
            elif self.relvit_mode == 1:
                loss += loss_rcl
            else:
                loss = loss
            with torch.no_grad():
                # ema update for RCL
                batch_center_rcl, batch_grid_center_rcl = cal_batch(teacher_output_rcl, teacher_grid_output_rcl)
                self.center_rcl = self.center_rcl * self.center_momentum + batch_center_rcl * (1 - self.center_momentum)
                self.center_grid_rcl = self.center_grid_rcl * self.center_momentum + batch_grid_center_rcl * (1 - self.center_momentum)

        return loss