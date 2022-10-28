# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the mcan-vqa library
# which was released under the Apache Licence.
#
# Source:
# https://github.com/MILVLG/mcan-vqa
#
# The license for the original version of this file can be
# found in https://github.com/MILVLG/mcan-vqa/blob/master/LICENSE
# The modifications to this file are subject to the same Apache Licence.
# ---------------------------------------------------------------

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from .models import register
MCAN_GQA_PARAMS = {
    'FRCN_FEAT_SIZE': (100, 2048),
    'GRID_FEAT_SIZE': (49, 2048),
    'BBOX_FEAT_SIZE': (100, 5),
    'BBOXFEAT_EMB_SIZE': 2048,
    'HIDDEN_SIZE': 512,  # former 512
    'FLAT_MLP_SIZE': 512,
    'FLAT_GLIMPSES': 1,
    'FLAT_OUT_SIZE': 1024,
    'DROPOUT_R': 0.1,
    'LAYER': 6,
    'FF_SIZE': 2048,
    'MULTI_HEAD': 8,
    'WORD_EMBED_SIZE': 300,
    # 'TOKEN_SIZE': 2933,
    'WORD_EMBED_SIZE': 300,
    'ANSWER_SIZE': 707,
    'MAX_TOKEN_LENGTH': 100,
    'USE_BBOX_FEAT': True,
    'USE_AUX_FEAT': True,
    'LANG_SIZE': 512
    }

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(MCAN_GQA_PARAMS['HIDDEN_SIZE'], MCAN_GQA_PARAMS['HIDDEN_SIZE'])
        self.linear_k = nn.Linear(MCAN_GQA_PARAMS['HIDDEN_SIZE'], MCAN_GQA_PARAMS['HIDDEN_SIZE'])
        self.linear_q = nn.Linear(MCAN_GQA_PARAMS['HIDDEN_SIZE'], MCAN_GQA_PARAMS['HIDDEN_SIZE'])
        self.linear_merge = nn.Linear(MCAN_GQA_PARAMS['HIDDEN_SIZE'], MCAN_GQA_PARAMS['HIDDEN_SIZE'])

        self.dropout = nn.Dropout(MCAN_GQA_PARAMS['DROPOUT_R'])

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            MCAN_GQA_PARAMS['MULTI_HEAD'],
            int(MCAN_GQA_PARAMS['HIDDEN_SIZE'] / MCAN_GQA_PARAMS['MULTI_HEAD'])
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            MCAN_GQA_PARAMS['MULTI_HEAD'],
            int(MCAN_GQA_PARAMS['HIDDEN_SIZE'] / MCAN_GQA_PARAMS['MULTI_HEAD'])
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            MCAN_GQA_PARAMS['MULTI_HEAD'],
            int(MCAN_GQA_PARAMS['HIDDEN_SIZE'] / MCAN_GQA_PARAMS['MULTI_HEAD'])
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            MCAN_GQA_PARAMS['HIDDEN_SIZE']
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=MCAN_GQA_PARAMS['HIDDEN_SIZE'],
            mid_size=MCAN_GQA_PARAMS['FF_SIZE'],
            out_size=MCAN_GQA_PARAMS['HIDDEN_SIZE'],
            dropout_r=MCAN_GQA_PARAMS['DROPOUT_R'],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()

        self.mhatt = MHAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(MCAN_GQA_PARAMS['DROPOUT_R'])
        self.norm1 = LayerNorm(MCAN_GQA_PARAMS['HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(MCAN_GQA_PARAMS['DROPOUT_R'])
        self.norm2 = LayerNorm(MCAN_GQA_PARAMS['HIDDEN_SIZE'])

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt()
        self.mhatt2 = MHAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(MCAN_GQA_PARAMS['DROPOUT_R'])
        self.norm1 = LayerNorm(MCAN_GQA_PARAMS['HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(MCAN_GQA_PARAMS['DROPOUT_R'])
        self.norm2 = LayerNorm(MCAN_GQA_PARAMS['HIDDEN_SIZE'])

        self.dropout3 = nn.Dropout(MCAN_GQA_PARAMS['DROPOUT_R'])
        self.norm3 = LayerNorm(MCAN_GQA_PARAMS['HIDDEN_SIZE'])

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA() for _ in range(MCAN_GQA_PARAMS['LAYER'])])
        self.dec_list = nn.ModuleList([SGA() for _ in range(MCAN_GQA_PARAMS['LAYER'])])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x


def feat_filter(frcn_feat, grid_feat, bbox_feat):
    feat_dict = {}

    feat_dict['FRCN_FEAT'] = frcn_feat
    feat_dict['GRID_FEAT'] = grid_feat
    feat_dict['BBOX_FEAT'] = bbox_feat

    return feat_dict


class BaseAdapter(nn.Module):
    def __init__(self):
        super(BaseAdapter, self).__init__()
        self.gqa_init()

    def gqa_init(self):
        raise NotImplementedError()

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(frcn_feat, grid_feat, bbox_feat)

        return self.gqa_forward(feat_dict)

    def gqa_forward(self, feat_dict):
        raise NotImplementedError()


class Adapter(BaseAdapter):
    def __init__(self):
        super(Adapter, self).__init__()

    def bbox_proc(self, bbox):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])
        # return torch.cat((bbox, area), -1)
        ##### FIXME: possibly buggy
        return torch.cat((bbox, area.unsqueeze(2)), -1)
        #####

    def gqa_init(self):
        imgfeat_linear_size = utils.MCAN_GQA_PARAMS['FRCN_FEAT_SIZE'][1]
        if utils.MCAN_GQA_PARAMS['USE_BBOX_FEAT']:
            self.bbox_linear = nn.Linear(5, utils.MCAN_GQA_PARAMS['BBOXFEAT_EMB_SIZE'])
            imgfeat_linear_size += utils.MCAN_GQA_PARAMS['BBOXFEAT_EMB_SIZE']
        self.frcn_linear = nn.Linear(imgfeat_linear_size, utils.MCAN_GQA_PARAMS['HIDDEN_SIZE'])

        if utils.MCAN_GQA_PARAMS['USE_AUX_FEAT']:
            self.grid_linear = nn.Linear(utils.MCAN_GQA_PARAMS['GRID_FEAT_SIZE'][1], utils.MCAN_GQA_PARAMS['HIDDEN_SIZE'])


    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if utils.MCAN_GQA_PARAMS['USE_BBOX_FEAT']:
            ##### FIXME: possibly buggy
            # bbox_feat = self.bbox_proc(bbox_feat)
            #####
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        if utils.MCAN_GQA_PARAMS['USE_AUX_FEAT']:
            grid_feat_mask = make_mask(grid_feat)
            img_feat_mask = torch.cat((img_feat_mask, grid_feat_mask), dim=-1)
            grid_feat = self.grid_linear(grid_feat)
            img_feat = torch.cat((img_feat, grid_feat), dim=1)

        return img_feat, img_feat_mask


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=MCAN_GQA_PARAMS['HIDDEN_SIZE'],
            mid_size=MCAN_GQA_PARAMS['FLAT_MLP_SIZE'],
            out_size=MCAN_GQA_PARAMS['FLAT_GLIMPSES'],
            dropout_r=MCAN_GQA_PARAMS['DROPOUT_R'],
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            MCAN_GQA_PARAMS['HIDDEN_SIZE'] * MCAN_GQA_PARAMS['FLAT_GLIMPSES'],
            MCAN_GQA_PARAMS['FLAT_OUT_SIZE']
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(MCAN_GQA_PARAMS['FLAT_GLIMPSES']):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


@register('mcan')
class MCAN(nn.Module):
    def __init__(self, **kwargs):
        super(MCAN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=utils.MCAN_GQA_PARAMS['TOKEN_SIZE'],
            embedding_dim=utils.MCAN_GQA_PARAMS['WORD_EMBED_SIZE']
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(np.load(kwargs.get('word_emb_path'))))

        self.lstm = nn.LSTM(
            input_size=utils.MCAN_GQA_PARAMS['WORD_EMBED_SIZE'],
            hidden_size=utils.MCAN_GQA_PARAMS['HIDDEN_SIZE'],
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter()

        self.backbone = MCA_ED()

        # Flatten to vector
        self.attflat_img = AttFlat()
        self.attflat_lang = AttFlat()

        # Classification layers
        self.proj_norm = LayerNorm(utils.MCAN_GQA_PARAMS['FLAT_OUT_SIZE'])
        self.proj = nn.Linear(utils.MCAN_GQA_PARAMS['FLAT_OUT_SIZE'], utils.MCAN_GQA_PARAMS['ANSWER_SIZE'])


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, info_nce=False, pretrain=False):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        # lang_feat: [B, MAX_TOKEN_LENGTH, HIDDEN_SIZE]
        # img_feat: [B, FRCN_FEAT_SIZE[0]+GRID_FEAT_SIZE[0], HIDDEN_SIZE]
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


@register('mcan-customized')
class MCANCustomized(nn.Module):
    def __init__(self, pretrained_emb, token_size, **kwargs):
        super(MCANCustomized, self).__init__()

        self.encoder = models.make(kwargs.get('encoder'), **kwargs.get('encoder_args'))
        self.connector = nn.Linear(self.encoder.out_dim, MCAN_GQA_PARAMS['HIDDEN_SIZE'])

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=MCAN_GQA_PARAMS['WORD_EMBED_SIZE']
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        # self.embedding.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=MCAN_GQA_PARAMS['WORD_EMBED_SIZE'],
            hidden_size=MCAN_GQA_PARAMS['HIDDEN_SIZE'],
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        # self.lang_feat_linear = nn.Sequential(
        #     nn.Linear(MCAN_GQA_PARAMS["LANG_SIZE"], MCAN_GQA_PARAMS["HIDDEN_SIZE"]),
        #     nn.GELU()
        #         )
        self.backbone = MCA_ED()

        # Flatten to vector
        self.attflat_img = AttFlat()
        self.attflat_lang = AttFlat()

        # Classification layers
        self.proj_norm = LayerNorm(MCAN_GQA_PARAMS['FLAT_OUT_SIZE'])
        self.proj = nn.Linear(MCAN_GQA_PARAMS['FLAT_OUT_SIZE'], MCAN_GQA_PARAMS['ANSWER_SIZE'])


    def forward(self, ims, ques_ix, info_nce=False, pretrain=False):
        if pretrain:
            B = ims.size(0)
            logits = torch.zeros(B, MCAN_GQA_PARAMS['ANSWER_SIZE']).to(ims)
            if info_nce:
                img_feat, attn_v = self.encoder(ims, info_nce=info_nce)
                return logits, attn_v
            else:
                return logits
        else:
            if info_nce:
                img_feat, attn_v = self.encoder(ims, info_nce=info_nce)
            else:
                img_feat = self.encoder(ims)
        B, C = img_feat.size(0), img_feat.size(1)
        img_feat = self.connector(img_feat.reshape(B, C, -1).permute(0, 2, 1))
        img_feat_mask = make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        self.lstm.flatten_parameters()
        lang_feat, _ = self.lstm(lang_feat)
        # lang_feat = self.lang_feat_linear(lang_feat)
        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        if info_nce:
            return proj_feat, attn_v
        else:
            return proj_feat
