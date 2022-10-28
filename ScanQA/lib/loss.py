import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothl1_loss(error, delta=1.0):
    diff = torch.abs(error)
    loss = torch.where(diff < delta, 0.5 * diff * diff / delta, diff - 0.5 * delta)
    return loss


def l1_loss(error):
    loss = torch.abs(error)
    return loss

class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = torch.softmax(inputs + 1e-8, dim=1)

        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss        