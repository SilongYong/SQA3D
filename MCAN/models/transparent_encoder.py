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

import torch
import torch.nn as nn
import torch.nn.functional as F

# from detectron2.modeling.poolers import ROIPooler

import models
import utils
from .models import register

'''
@register('transparent_bbox_encoder')
class TransparentBBoxNetworkEncoder(nn.Module):
    def __init__(self, encoder, **kwargs):
        raise NotImplementedError('Currently we have to make sure each image only produce one token.')
        super(TransparentBBoxNetworkEncoder, self).__init__()

        # image encoder
        encoder = models.make(encoder)
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.proj = nn.Conv2d(encoder.out_dim, encoder.out_dim // 2, kernel_size=1)

        # ROI Pooler
        self.roi_pooler = ROIPooler(
           output_size=7,
           scales=(1/32,),
           sampling_ratio=0,
           pooler_type='ROIAlignV2',
        )
        self.roi_processor = nn.Sequential(
            nn.Conv2d(encoder.out_dim // 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*7*7, 512),
            nn.ReLU()
        )
        self.roi_processor_ln = nn.LayerNorm(512)

        # bbox coord encoding
        self.roi_processor_box = nn.Linear(4, 128)
        self.roi_processor_box_ln = nn.LayerNorm(128)
        rn_in_planes = 512 + 128

        self.out_dim = rn_in_planes

    def forward(self, im, boxes, boxes_dim=None):
        # assert im.shape[0] == len(boxes), 'im: {} vs boxes: {}'.format(im.shape[0], len(boxes))
        img_shape = im.shape
        im = im.view(-1, *img_shape[-3:])
        num_im = im.size(0)
        # assert im.shape[0] == boxes_dim.shape[0], '{} vs {}'.format(im.shape, boxes_dim.shape)
        if boxes_dim is not None:
            boxes_dim_shape = boxes_dim.shape
            boxes_dim = boxes_dim.view(-1, *boxes_dim_shape[-1:])

        # BxCxHxW
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.proj(x)

        # RoI pooling/align
        # x_shape = list(img_shape[:-3]) + list(x.shape[-3:])
        # x = x.view(x_shape)
        all_boxes = []
        for boxes_i in boxes:
            all_boxes.extend(boxes_i)
        num_boxes = [boxes_i.tensor.shape[0] for boxes_i in all_boxes]

        # roi_feats = roi_align(
        #    x, all_boxes,
        #    output_size=(7, 7),
        #    spatial_scale=1/32.,
        #    sampling_ratio=0,
        #    aligned=True
        # )
        roi_feats = self.roi_pooler([x], all_boxes)
        roi_feats = self.roi_processor(roi_feats)
        roi_feats = self.roi_processor_ln(roi_feats)
        # Add bbox pos features
        bbox_tensor = torch.cat([box.tensor for box in all_boxes]).to(roi_feats)
        # bbox coord normalization
        bbox_tensor[:, 0] = bbox_tensor[:, 0] / im.shape[3]
        bbox_tensor[:, 1] = bbox_tensor[:, 1] / im.shape[2]
        bbox_tensor[:, 2] = bbox_tensor[:, 2] / im.shape[3]
        bbox_tensor[:, 3] = bbox_tensor[:, 3] / im.shape[2]
        bbox_tensor = bbox_tensor * 2 - 1
        roi_box_feats = self.roi_processor_box_ln(self.roi_processor_box(bbox_tensor))
        roi_feats = torch.cat([roi_feats, roi_box_feats], dim=-1)

        # TODO: This assumes all the images have the same number of bboxes.
        feat = roi_feats.reshape(num_im, -1, roi_feats.size(-1))
        return feat
'''
# CNN backbone then superpixels as patches
@register('transparent_superpixel_encoder')
class TransparentSuperpixelEncoder(nn.Module):
    def __init__(self, encoder, **kwargs):
        super().__init__()

        # image encoder
        encoder = models.make(encoder)
        self.encoder = encoder
        self.out_dim = encoder.out_dim

    def forward(self, im, boxes=None, boxes_dim=None, info_nce=False):
        img_shape = im.shape
        im = im.view(-1, *img_shape[-3:])
        num_im = im.size(0)

        # BxCxHxW
        if info_nce:
            feats, attn_v = self.encoder(im, info_nce)
            return feats, attn_v
        else:
            feats = self.encoder(im)
            return feats
