from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import t2np

def one_hot(index, classes):

    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).to(index.device)
        mask = Variable(mask, volatile=index.volatile).to(index.device)
    return mask.scatter_(1, index, ones)

def one_hot_embedding(labels, num_classes):

    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = .25
        self.gamma = 2
        self.e = torch.eye(num_classes+1, requires_grad=False).to('cuda')


    def _ohe(self, y):
        # ohe = torch.zeros((y.shape[0], self.num_classes), requires_grad=False).to('cuda')
        # print("Y SHAPE", y.shape)
        # print('OHE SHAPE', ohe.shape)
        # ohe.scatter_(1, y.reshape(-1, 1), 1)

        # keke = self.e.to('cpu')
        # keky = y.to('cpu')
        # print(keke[keky].sum(0))
        # q=input()
        return self.e[y]



    def focal_loss(self, x, y):
        y = y.float()
        p = F.sigmoid(x)

        fg = self.alpha * y * ((1 - p) ** self.gamma) * p.log()
        bg = (1 - self.alpha) * (1 - y) * (p ** self.gamma) * (1 - p).log()

        loss = - (fg.sum() + bg.sum())
        return loss


    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets ,verbose):
        batch_size, n_anchors = cls_targets.size()
        fg_idx = cls_targets > 0
        n_fg = fg_idx.float().sum()

        fg_mask = fg_idx.unsqueeze(2).expand_as(loc_preds)

        fg_loc_preds = loc_preds[fg_mask]
        fg_loc_targets = loc_targets[fg_mask]
        loc_loss = 0.4 * F.smooth_l1_loss(fg_loc_preds, fg_loc_targets, reduction='none').sum()

        fbg_idx = cls_targets != -1

        print('cls_targets shape', cls_targets.shape)
        fbg_mask = fbg_idx.expand_as(cls_targets)
        print('fgb_mask shape', fbg_mask.shape)
        fgb_cls_preds = cls_preds[fbg_mask].squeeze()
        fgb_cls_targets = cls_targets[fbg_mask]

        print('fgb_cls_preds', fgb_cls_preds.shape)
        print('fgb_cls_targets', fgb_cls_targets.shape)

        ohe_targets = self._ohe(fgb_cls_targets)
        # print(ohe_targets[:10])
        print(ohe_targets.shape)
        print(fgb_cls_preds.shape)

        cls_loss = self.focal_loss(fgb_cls_preds, ohe_targets)

        if verbose:
            np_loc_loss = t2np(loc_loss)
            np_cls_loss = t2np(cls_loss)
            np_n_fg = t2np(n_fg)
            # np_n_fg = batch_size
            print('loc_loss: %.5f | cls_loss: %.4f' % (np_loc_loss / np_n_fg, np_cls_loss / np_n_fg), end=' | ')

        return (loc_loss + cls_loss) / n_fg
        # print('mask', fg_idx.unsqueeze(2).expand_as(loc_preds))
        # loc_loss = F.smooth_l1_loss(size_average=False)

        # mask = pos.unsqueeze(2).expand_as(loc_preds)
        # masked_loc_preds = loc_preds[mask].view(-1,4)
        # masked_loc_targets = loc_targets[mask].view(-1,4)
        # loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets,size_average=False)
        # pos_neg = cls_targets > -1
        # mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        # masked_cls_preds = cls_preds[mask].view(-1, self.num_classes+1)
        # cls_loss =self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        # if verbose:
        #     print('loc_loss: %.5f | cls_loss: %.4f' %((loc_loss.to('cpu').detach().numpy().mean()/t2np(num_pos)), cls_loss.to('cpu').detach().numpy().mean()/t2np(num_pos)), end=' | ')
        #
        # loss =(loc_loss+cls_loss)/(num_pos)
        # return loss
