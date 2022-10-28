""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from data.scannet.model_util_scannet import ScannetDatasetConfig
from utils.box_util import get_3d_box_batch
# constants
DC = ScannetDatasetConfig()

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
import lib.pointnet2.pointnet2_utils
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, proposal_size=128, radius=0.3, nsample=16):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.votenet_hidden_size = proposal_size

        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.num_proposal,
            radius=radius, # 0.3 (scanrefer, votenet), 5 (scan2cap)
            nsample=nsample, # 16 (scanrefer, votenet), 512 (scan2cap)
            mlp=[self.seed_feat_dim, proposal_size, proposal_size, proposal_size],
            use_xyz=True,
            normalize_xyz=True
        )
            
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)

        self.proposal = nn.Sequential(
            nn.Conv1d(proposal_size,proposal_size,1, bias=False),
            nn.BatchNorm1d(proposal_size),
            nn.ReLU(),
            nn.Conv1d(proposal_size,proposal_size,1, bias=False),
            nn.BatchNorm1d(proposal_size),
            nn.ReLU(),
            nn.Conv1d(proposal_size,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        )

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # Farthest point sampling (FPS) on votes
        # feturea: batch, 256, 1024
        xyz, features, fps_inds = self.vote_aggregation(xyz, features) #  batch, votenet_mlp_size (128), 256
        
        sample_inds = fps_inds
        data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features)
        # net: batch, ???, num_proposals (32, 97, 256)
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_pred_box(self, data_dict):
        # predicted bbox
        pred_center = data_dict["center"].detach().cpu().numpy() # (B,K,3)
        pred_heading_class = torch.argmax(data_dict["heading_scores"], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(data_dict["size_scores"], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

        batch_size, num_proposals, _ = pred_center.shape
        pred_bboxes = []
        for i in range(batch_size):
            # convert the bbox parameters to bbox corners
            pred_obb_batch = DC.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                        pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            pred_bboxes.append(torch.from_numpy(pred_bbox_batch).cuda().unsqueeze(0))
        pred_bboxes = torch.cat(pred_bboxes, dim=0) # batch_size, num_proposals, 8, 3
        return pred_bboxes

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        #net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, num_proposal, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2]

        base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)

        heading_scores = net_transposed[:,:,5:5+num_heading_bin]
        heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
        
        size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        
        sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        data_dict['heading_scores'] = heading_scores # B x num_proposal x num_heading_bin
        data_dict['heading_residuals_normalized'] = heading_residuals_normalized # B x num_proposal x num_heading_bin (should be -1 to 1)
        data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # B x num_proposal x num_heading_bin
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores # B x num_proposal x 10

        # processed box info
        data_dict["bbox_corner"] = self.decode_pred_box(data_dict) # batch_size, num_proposals, 8, 3 (bounding box corner coordinates)
        data_dict["bbox_feature"] = data_dict["aggregated_vote_features"]
        data_dict["bbox_mask"] = objectness_scores.argmax(-1)
        data_dict['bbox_sems'] = sem_cls_scores.argmax(-1) # # B x num_proposal
        #data_dict['sem_cls'] = sem_cls_scores.argmax(-1)

        return data_dict

