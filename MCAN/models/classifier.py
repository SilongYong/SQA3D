# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for RelViT. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('classifier')
class Classifier(nn.Module):

    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x, boxes=None, info_nce=False):
        if info_nce:
            x, attn_v = self.encoder(x, boxes, info_nce=info_nce)
            x = self.classifier(x)
            return x, attn_v
        else:
            x = self.encoder(x, boxes)
            x = self.classifier(x)
            return x


@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)

# Assume input is [B, C, H, W]
@register('max-pooling-classifier')
class MaxPoolingClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, n_classes)
        )

    def forward(self, x):
        # x: [B, C, W, H]
        B, C = x.size(0), x.size(1)
        x = x.reshape(B, C, -1).max(-1)[0]
        # x = x.reshape(B, C, -1)[:, :, 0]
        return self.proj(x)


# Assume input is [B, C, H, W]
@register('max-pooling-classifier-twoheads')
class MaxPoolingClassifierTwoHeads(nn.Module):

    def __init__(self, in_dim, n_classes1, n_classes2, **kwargs):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, n_classes1)
        )
        self.proj2 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, n_classes2)
        )

    def forward(self, x):
        # x: [B, C, W, H]
        B, C = x.size(0), x.size(1)
        x = x.reshape(B, C, -1).max(-1)[0]
        # x = x.reshape(B, C, -1)[:, :, 0]
        return (self.proj1(x), self.proj2(x))


@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)
