"""
THIS SCRIPT CONTAINS ALL THE TRIPLET LOSS FUNCTIONS USED TO TRAIN THE NETWORK MODELS
Training scripts will call these classes to use the losses
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from config import PARAMS
#
#
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')


def get_loss(lf):
    if lf == 'triplet loss':
        criterion = TripletLoss()
    elif lf == 'lifted embedding':
        criterion = LiftedEmbeddingLoss()
    elif lf == 'lazy triplet':
        criterion = LazyTripletLoss()
    elif lf == 'semi hard':
        criterion = SemiHardLoss()
    elif lf == 'batch hard':
        criterion = BatchHardLoss()
    elif lf == 'circle loss':
        criterion = CircleLoss()
    elif lf == 'angular loss':
        criterion = AngularLoss()
    else:
        criterion = -1
    return criterion


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, margin):
        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)
        losses = torch.relu(distance_positive - distance_negative + margin)

        return losses.mean()


class LiftedEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(LiftedEmbeddingLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, margin):
        distance_anc_pos = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_anc_neg = F.pairwise_distance(anchor, negative, keepdim=True)
        distance_pos_neg = F.pairwise_distance(positive, negative, keepdim=True)
        loss = torch.relu(distance_anc_pos + torch.log(torch.exp(margin-distance_anc_neg) + torch.exp(margin-distance_pos_neg)))
        return loss.mean()


class LazyTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(LazyTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, margin):
        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)
        losses = torch.relu(distance_positive - distance_negative + margin)

        return losses.max()

class CircleLoss(nn.Module):
    def __init__(self, margin=1, gamma=1):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive, negative, margin):
        delta_p = 1 - margin
        delta_n = margin
        op = 1 + margin
        on = - margin

        pos_cos_similarity = (torch.diagonal(torch.matmul(anchor, positive.transpose(0, 1)))).unsqueeze(1)
        neg_cos_similarity = (torch.diagonal(torch.matmul(anchor, negative.transpose(0, 1)))).unsqueeze(1)

        alpha_p = torch.abs(op - pos_cos_similarity)
        alpha_n = torch.abs(neg_cos_similarity - on)
        similarity_positive = torch.exp(-self.gamma * alpha_p * (pos_cos_similarity - delta_p))
        similarity_negative = torch.exp(self.gamma * alpha_n * (neg_cos_similarity - delta_n))
        losses = torch.log(1 + torch.sum(similarity_positive) * torch.sum(similarity_negative))
        return losses.mean()


class BatchHardLoss(nn.Module):
    def __init__(self, margin=0.25):
        super(BatchHardLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, margin):
        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)
        hardest_positive = torch.max(distance_positive)
        hardest_negative = torch.min(distance_negative)
        loss = torch.relu(hardest_positive - hardest_negative + margin)

        return loss


class SemiHardLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(SemiHardLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, margin):
        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)
        mean_positive = torch.mean(distance_positive)
        hardest_negative = torch.min(distance_negative)
        loss = torch.relu(mean_positive - hardest_negative + margin)

        return loss


class AngularLoss(nn.Module):
    def __init__(self, alpha=45):
        super(AngularLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative, alpha):
        alpha = 2*np.pi*alpha/360
        f = 4*(np.tan(alpha)) ** 2 * (torch.diagonal(torch.matmul(anchor+positive, negative.transpose(0, 1)))).unsqueeze(1) \
            - 2 * (1+(np.tan(alpha)) ** 2) * (torch.diagonal(torch.matmul(anchor, positive.transpose(0, 1)))).unsqueeze(1)
        f = f.reshape(-1, 1)
        loss = torch.relu(torch.logsumexp(f, dim=0)[0])
        return loss
