# Copyright (c) 2019 Yuhao Cui
#
# This source code is licensed under the MIT license 
# [see https://github.com/MILVLG/mcan-vqa/blob/master/LICENSE for details]

''' 
MCAN module: represents the relationship between question words and objects

Modified from: https://github.com/MILVLG/mcan-vqa/blob/master/core/model/mca.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, in_size, out_size, pdrop=0., use_gelu=True):
        super(FC, self).__init__()
        self.pdrop = pdrop
        self.use_gelu = use_gelu

        self.linear = nn.Linear(in_size, out_size)

        if use_gelu:
            #self.relu = nn.Relu(inplace=True)
            self.gelu = nn.GELU()

        if pdrop > 0:
            self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        x = self.linear(x)

        if self.use_gelu:
            #x = self.relu(x)
            x = self.gelu(x)

        if self.pdrop > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, pdrop=0., use_gelu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, pdrop=pdrop, use_gelu=use_gelu)
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
# ---- Flatten the sequence ----
# ------------------------------
class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size=512, flat_glimpses=1, flat_out_size=1024, pdrop=0.1):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            pdrop=pdrop,
            use_gelu=True
        )
        self.flat_glimpses = flat_glimpses

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted, att

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------
class MHAtt(nn.Module):
    def __init__(self, hidden_size, num_heads=8, pdrop=0.1):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_hidden_size = int(hidden_size / num_heads)

        self.dropout = nn.Dropout(pdrop)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_heads,
            self.head_hidden_size
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_heads,
            self.head_hidden_size
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_heads,
            self.head_hidden_size
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    # 0 where the element is, 1 where the element is not
    # ([[0, 0, 0], 
    #  [0, 0, 1], 
    #  [0, 1, 1]]).bool() True, False
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
    def __init__(self, hidden_size, pdrop=0.1):
        super(FFN, self).__init__()

        ff_size = int(hidden_size * 4)

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            pdrop=pdrop,
            use_gelu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------
class SA(nn.Module):
    def __init__(self, hidden_size, num_heads=8, pdrop=0.1):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hidden_size, num_heads, pdrop)
        self.ffn = FFN(hidden_size, pdrop)

        self.dropout1 = nn.Dropout(pdrop)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(pdrop)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------
class SGA(nn.Module):
    def __init__(self, hidden_size, num_heads=8, pdrop=0.1):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(hidden_size, num_heads, pdrop)
        self.mhatt2 = MHAtt(hidden_size, num_heads, pdrop)
        self.ffn = FFN(hidden_size, pdrop)

        self.dropout1 = nn.Dropout(pdrop)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(pdrop)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(pdrop)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------
class MCAN_ED(nn.Module):
    def __init__(self, hidden_size, num_heads=8, num_layers=6, pdrop=0.1):
        super(MCAN_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(hidden_size, num_heads, pdrop) for _ in range(num_layers)])
        self.dec_list = nn.ModuleList([SGA(hidden_size, num_heads, pdrop) for _ in range(num_layers)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
