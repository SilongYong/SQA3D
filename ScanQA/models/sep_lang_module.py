import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.qa_helper import *
    

class LangModule(nn.Module):
    def __init__(self, num_object_class, use_lang_classifier=True, use_bidir=False, num_layers=1,
        emb_size=300, hidden_size=256, pdrop=0.1):
        super().__init__() 

        self.num_object_class = num_object_class
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.num_layers = num_layers               

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=use_bidir,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.word_drop = nn.Dropout(pdrop)

        lang_size = hidden_size * 2 if use_bidir else hidden_size

        # Language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Dropout(p=pdrop),
                nn.Linear(lang_size, num_object_class),
            )

    def make_mask(self, feature):
        """
        return a mask that is True for zero values and False for other values.
        """
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0) #.unsqueeze(-1) #.unsqueeze(2)        


    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        s_embs = data_dict["s_feat"]
        q_embs = data_dict["q_feat"]

        # dropout word embeddings
        s_embs = self.word_drop(s_embs)
        q_embs = self.word_drop(q_embs)

        s_feat = pack_padded_sequence(s_embs, data_dict["s_len"].cpu(), batch_first=True, enforce_sorted=False)
        q_feat = pack_padded_sequence(q_embs, data_dict["q_len"].cpu(), batch_first=True, enforce_sorted=False)

        # encode description
        packed_s, (_, _) = self.lstm(s_feat)
        packed_q, (_, _) = self.lstm(q_feat)
        
        s_output, _ = pad_packed_sequence(packed_s, batch_first=True)
        q_output, _ = pad_packed_sequence(packed_q, batch_first=True)
        
        data_dict["s_out"] = s_output
        data_dict["q_out"] = q_output
        data_dict["s_mask"] = self.make_mask(s_output)
        data_dict["q_mask"] = self.make_mask(q_output)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])
        return data_dict
