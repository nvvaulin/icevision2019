import argparse
import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import preprocessing.transforms as transforms
from encoder import DataEncoder
from loss import FocalLoss
from retinanet import RetinaNet
from preprocessing.datasets import VocLikeDataset, IceDataset
from utils import t2np

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler

from apex import amp

# amp_handle = amp.init(enabled=True)

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--exp', required=True, help='experiment name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

#sys.path.insert(0, os.path.join('exps', 'voc'))
import config as cfg

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')
start_epoch = 0
lr = cfg.lr

print('Preparing data..')

train_transform_list = [
    transforms.RandomHorizontalFlip(),
    transforms.AdditiveGaussian(),
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std),
]
if cfg.scale is not None:
    train_transform_list.insert(0,transforms.Scale(cfg.scale))
train_transform = transforms.Compose(train_transform_list)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std)
])

trainset = IceDataset(test=False, encoder=DataEncoder(), transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                                      num_workers=cfg.num_workers, collate_fn=trainset.collate_fn)
valset = IceDataset(test=False, encoder=DataEncoder(), transform=val_transform, val=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=10, collate_fn=valset.collate_fn)

print('Building model...')
net = RetinaNet(backbone=cfg.backbone, num_classes=99)
net.cuda()
cudnn.benchmark = True

if args.resume:
    print('Resuming from checkpoint..')
    # checkpoint = torch.load(os.path.join('ckpts', args.exp, '29_ckpt.pth'), map_location='cuda')
    # checkpoint = torch.load(os.path.join('ckpts', 'efnet4', '29_ckpt.pth'), map_location='cuda')
    model_state  = net.state_dict()
    checkpoint = torch.load('/media/grisha/hdd1/icevision/efnet_detector_wo_cls.pth', map_location='cuda')
    model_state.update(checkpoint)
    net.load_state_dict(model_state)
    # net.load_state_dict(checkpoint)
    # start_epoch = checkpoint['epoch']
    # lr = cfg.lr




criterion = FocalLoss(99)

# optimizer = optim.Adam(net.parameters(), lr=cfg.lr,weight_decay=cfg.weight_decay)
optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
cosine_lr = CosineAnnealingWarmRestarts(optimizer, T_0=len(trainloader), T_mult=2)
lr_s = GradualWarmupScheduler(optimizer, multiplier=40, total_epoch=2000, after_scheduler=cosine_lr)

# optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
# lr_s = CosineAnnealingWarmRestarts(optimizer, T_0=5*len(trainloader), T_mult=1)


net, optimizer = amp.initialize(net, optimizer, opt_level="O2", keep_batchnorm_fp32=True)


def train(epoch):
    from time import time

    print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0

    tik = time()
    tok = time()

    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
#         if True:
#             print(cls_targets.unique())
#             print(inputs.shape)
        # print(inputs)
        inputs = inputs.to('cuda')
        loc_targets = loc_targets.to('cuda')
        cls_targets = cls_targets.to('cuda')
        # print(cls_targets[:10])

        optimizer.zero_grad()

        loc_preds, cls_preds = net(inputs)
        # print(cls_preds)

        # pos = (cls_targets > 0).detach()
        # num_pos = t2np(pos).astype(np.int).sum()
        # if num_pos==0:
        #     print('zero num positive')
        #     continue

        # print('loc_preds', loc_preds.shape)
        # print('cls_preds', cls_preds.shape)
        # print('loc_targets', loc_targets.shape)
        # print('cls_targets', cls_targets.shape)
        # print('loc_preds', loc_preds.shape)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets, batch_idx%20==0)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
#         nn.utils.clip_grad_norm(net.parameters(), max_norm=1.0)
        optimizer.step()
        lr_s.step()
        train_loss += t2np(loss).mean()

        if batch_idx%20==0:
            tok = time()
            print('batch: %d/%d | time: %.2f min | train_loss: %.3f | avg_loss: %.4f | lr: %.6f ' % (batch_idx, len(trainloader), (tok-tik)/60, loss.detach().to('cpu').numpy().mean(), train_loss/(batch_idx+1), lr_s.get_lr()[0]))
            import sys
            sys.stdout.flush()
            tik = time()

def val(epoch):
    net.eval()
    val_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
        inputs = inputs.to('cuda')
        loc_targets = loc_targets.to('cuda')
        cls_targets = cls_targets.to('cuda')

        loc_preds, cls_preds = net(inputs)

            # pos = (cls_targets > 0)

        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets,batch_idx%1==0)

        loss, inputs, loc_targets, cls_targets = t2np(loss), t2np(inputs), t2np(loc_targets), t2np(cls_targets)


        val_loss += loss.mean()
        if batch_idx%1==0:
            print('val_loss: %.4f | avg_loss: %.4f' % (loss.mean(), val_loss/(batch_idx+1)))
            import sys
            sys.stdout.flush()
    print('val_loss: %.4f | avg_loss: %.4f' % (loss.mean(), val_loss/(batch_idx+1)))

    save_checkpoint(val_loss, epoch, len(valloader))





def save_checkpoint(loss, epoch, n):
    global best_loss
    loss /= n
    if loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': loss,
            'epoch': epoch,
            'lr': lr
        }
        ckpt_path = os.path.join('ckpts', args.exp)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(state, os.path.join(ckpt_path, str(epoch)+'_ckpt.pth'))
        best_loss = loss

for epoch in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
    # if epoch in cfg.lr_decay_epochs:
    #     lr *= 0.1
    #     print('learning rate decay to: ', lr)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    train(epoch)
    if cfg.eval_while_training and epoch % cfg.eval_every == 0:
        with torch.no_grad():
            val(epoch)
