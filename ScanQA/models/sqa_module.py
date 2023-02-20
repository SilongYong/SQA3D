import torch
import torch.nn as nn
from models.mcan_sqa_module import MCAN_ED, AttFlat, LayerNorm, SA, SGA
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.sep_lang_module import LangModule

class ScanQA(nn.Module):
    def __init__(self, num_answers,
        # proposal
        num_object_class, input_feature_dim,
        num_heading_bin, num_size_cluster, mean_size_arr,
        num_proposal=256, vote_factor=1, sampling="vote_fps", seed_feat_dim=256, proposal_size=128,
        pointnet_width=1,
        pointnet_depth=2,
        vote_radius=0.3,
        vote_nsample=16,
        # qa
        answer_pdrop=0.3,
        mcan_num_layers=2,
        mcan_num_heads=8,
        mcan_pdrop=0.1,
        mcan_flat_mlp_size=512,
        mcan_flat_glimpses=1,
        mcan_flat_out_size=1024,
        # lang
        lang_use_bidir=False,
        lang_num_layers=1,
        lang_emb_size=300,
        lang_pdrop=0.1,
        # common
        hidden_size=128,
        # option
        use_object_mask=False,
        use_aux_situation=False,
        use_answer=False,
        num_pos=7,
        wo3d=False,
        Lam=0.005,
    ):
        super().__init__()

        # Option
        self.use_object_mask = use_object_mask
        self.use_aux_situation = use_aux_situation
        self.use_answer = use_answer
        self.wo3d = wo3d
        lang_size = hidden_size * (1 + lang_use_bidir)
        # Language encoding
        self.lang_net = LangModule(num_object_class, use_lang_classifier=False,
                                    use_bidir=lang_use_bidir, num_layers=lang_num_layers,
                                    emb_size=lang_emb_size, hidden_size=hidden_size, pdrop=lang_pdrop)

        # Ojbect detection
        self.detection_backbone = Pointnet2Backbone(input_feature_dim=input_feature_dim,
                                                width=pointnet_width, depth=pointnet_depth,
                                                seed_feat_dim=seed_feat_dim)
        # Hough voting
        self.voting_net = VotingModule(vote_factor, seed_feat_dim)

        # Vote aggregation and object proposal
        self.proposal_net = ProposalModule(num_object_class, num_heading_bin, num_size_cluster, mean_size_arr,
                                        num_proposal, sampling, seed_feat_dim=seed_feat_dim, proposal_size=proposal_size,
                                        radius=vote_radius, nsample=vote_nsample)

        # Feature projection
        self.lang_feat_linear = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.GELU()
        )

        self.s_feat_linear = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.GELU()
        )
        self.q_feat_linear = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.GELU()
        )
        self.object_feat_linear = nn.Sequential(
            nn.Linear(proposal_size, hidden_size),
            nn.GELU()
        )

        self.enc_list_s = nn.ModuleList([SA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        self.enc_list_q = nn.ModuleList([SA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        self.dec_list = nn.ModuleList([SGA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        self.dec_list_2 = nn.ModuleList([SGA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        # --------------------------------------------

        # Esitimate confidence
        self.object_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
        )

        # Language classifier
        self.aux_reg = nn.Sequential(
                nn.Linear(2*mcan_flat_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_pos)
        )

        # QA head
        self.attflat_visual = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_s = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_q = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_lstm_q = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        if self.wo3d:
            self.answer_cls = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(answer_pdrop),
                    nn.Linear(hidden_size, num_answers)
            )
        else:
            self.answer_cls = nn.Sequential(
                    nn.Linear(3*mcan_flat_out_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(answer_pdrop),
                    nn.Linear(hidden_size, num_answers)
            )
        self.Lam = Lam

    def forward(self, data_dict):
        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################

        # --------- LANGUAGE ENCODING ---------
        data_dict = self.lang_net(data_dict)

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.detection_backbone(data_dict)

        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"] # batch_size, seed_feature_dim, num_seed, (16, 256, 1024)
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz

        data_dict["seed_features"] = features
        xyz, features = self.voting_net(xyz, features) # batch_size, vote_feature_dim, num_seed * vote_factor, (16, 256, 1024)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal_net(xyz, features, data_dict)

        #######################################
        #                                     #
        #             QA BACKBONE             #
        #                                     #
        #######################################

        # unpack outputs from question encoding branch
        s_feat = data_dict["s_out"]
        s_mask = data_dict["s_mask"]
        q_feat = data_dict["q_out"]
        q_mask = data_dict["q_mask"]

        # unpack outputs from detection branch
        if not self.wo3d:
            object_feat = data_dict['aggregated_vote_features'] # batch_size, num_proposal, proposal_size (128)
        if self.use_object_mask:
            object_mask = ~data_dict["bbox_mask"].bool().detach() #  # batch, num_proposals
        else:
            object_mask = None
        if s_mask.dim() == 2:
            s_mask = s_mask.unsqueeze(1).unsqueeze(2)
        if q_mask.dim() == 2:
            q_mask = q_mask.unsqueeze(1).unsqueeze(2)
        if not self.wo3d:
            if object_mask.dim() == 2:
                object_mask = object_mask.unsqueeze(1).unsqueeze(2)

        # --------- QA BACKBONE ---------
        # Pre-process Lanauge & Image Feature

        s_feat = self.lang_feat_linear(s_feat)
        q_feat = self.lang_feat_linear(q_feat)
        if not self.wo3d:
            object_feat = self.object_feat_linear(object_feat) # batch_size, num_proposal, hidden_size

        # QA Backbone (Fusion network) original double enc and dec

        for enc in self.enc_list_s:
            s_feat = enc(s_feat, s_mask)
        for enc in self.enc_list_q:
            q_feat = enc(q_feat, q_mask)
        if not self.wo3d:
            for dec in self.dec_list:
                object_feat = dec(object_feat, s_feat, object_mask, s_mask)
            for dec in self.dec_list_2:
                object_feat = dec(object_feat, q_feat, object_mask, q_mask)

        # object_feat: batch_size, num_proposal, hidden_size
        # lang_feat: batch_size, num_words, hidden_size

        #######################################
        #                                     #
        #          PROPOSAL MATCHING          #
        #                                     #
        #######################################

        s_feat, data_dict["satt"] = self.attflat_s(
                s_feat,
                s_mask
        )

        q_feat, data_dict["qatt"] = self.attflat_q(
                q_feat,
                q_mask,
        )
        if not self.wo3d:
            object_feat, data_dict["oatt"] = self.attflat_visual(
                    object_feat,
                    object_mask
            )

        if not self.wo3d:
            fuse_feat = torch.cat((s_feat, q_feat, object_feat), dim=1)
        else:
            fuse_feat = torch.cat((s_feat, q_feat), dim=1)
        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################
        if self.use_aux_situation:
            assert self.wo3d is False
            temp = torch.cat((s_feat, object_feat), dim=1)
            data_dict["aux_scores"] = self.aux_reg(temp)
        #######################################
        #                                     #
        #          QUESTION ANSERING          #
        #                                     #
        #######################################
        if self.use_answer:
            if self.wo3d:
                data_dict["answer_scores"] = self.answer_cls(fuse_feat) # batch_size, num_answers
            else:
                data_dict["answer_scores"] = self.answer_cls(fuse_feat)

        return data_dict
