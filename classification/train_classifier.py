import torch
from torch import nn
from torch.optim import SGD
import pickle
from apex import amp
from glob import glob
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import FuckingDataset, N_CLASSES
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import sys
import numpy as np
import os
from time import time
from argparse import ArgumentParser

from model import SignModel
from utils import t2np
from data import get_preprocessing
from losses import LabelSmoothingLoss


def parse_args():
    parser = ArgumentParser('sign cls trainig')
    parser.add_argument('--train-data', dest="train_data", required=True)
    parser.add_argument('--val-data', dest="val_data", required=True)
    parser.add_argument('--model', dest="path_to_models", required=True)
    parser.add_argument('--init_lr', default=0.01, type=float)
    parser.add_argument('--n_cycles', default=3, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--amp', default='O1', type=str)
    parser.add_argument('--n_workers', default=10, type=int)
    return parser.parse_args()


def train_epoch(model, criterion, optim, lr_scheduler, loader, meta):
    model.train()
    tik = time()
    for batch_idx, data in enumerate(loader):
        x = data['image']
        y = data['classes']
        x, y = x.to('cuda'), y.to('cuda')

        optim.zero_grad()
        y_hat = model(x)
        probs = torch.functional.F.sigmoid(y_hat)
        loss = criterion(y_hat, y)

        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()

        optim.step()
        lr_scheduler.step()

        prediction = probs.flatten() > 0.5
        gt = y.flatten() > 0.5

        accuracy = (t2np(prediction) == t2np(gt)).mean()
        meta['iteration'].append(meta['iteration'][-1]+1)
        meta['lr'].append(lr_scheduler.get_lr()[0])
        meta['train']['loss'].append(t2np(loss).mean())
        meta['train']['accuracy'].append(accuracy)

        if batch_idx % 100 == 0:
            tok = time()
            print('batch {:#5}/{:^5} | loss {:^.3f} | accuracy {:^.3f} | lr {:^.6f} | time {:^.2f}s'.format(
                batch_idx, len(loader), meta['train']['loss'][-1], meta['train']['accuracy'][-1], meta['lr'][-1], (tok - tik)))
            sys.stdout.flush()
            tik = time()


def val(model, criterion, loader, meta, amp):
    model.eval()
    accuracy = []
    mean_loss = 0
    for batch_idx, data in enumerate(loader):
        x = data['image']
        y = data['classes']
        x = x.to('cuda')
        if amp != '01':
            x = x.half()
        y = y.to('cuda')

        y_hat = model(x)
        probs = torch.sigmoid(y_hat)
        loss = criterion(y_hat, y)

        prediction = probs.flatten() > 0.5
        gt = y.flatten() > 0.5
        accuracy.extend(t2np(prediction) == t2np(gt))
        mean_loss += t2np(loss).mean()

    accuracy = np.mean(accuracy)
    mean_loss = np.mean(mean_loss / len(loader))

    meta['val']['loss'].append(mean_loss)
    meta['val']['accuracy'].append(accuracy)
    meta['val']['iteration'].append(meta['iteration'][-1])

    print('\nval loss {:^.3f} | val accuracy {:^.3f}'.format(meta['val']['loss'][-1], meta['val']['accuracy'][-1]))


if __name__ == '__main__':
    args = parse_args()

    train_transform = get_preprocessing(train=True)
    trainset = FuckingDataset(args.train_data, transforms=train_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    val_transform = get_preprocessing(train=False)
    valset = FuckingDataset(args.val_data, transforms=val_transform)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    model = SignModel(n_classes=N_CLASSES).to('cuda')
    optim = SGD(params=model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=True)
    lr_scheduler = CosineAnnealingWarmRestarts(optim, T_0=len(trainloader), T_mult=2)

    if args.amp in ('02', '03'):
        keep_batchnorm_fp32 = True
    else:
        keep_batchnorm_fp32 = None
    model, optim = amp.initialize(model, optim, opt_level=args.amp, keep_batchnorm_fp32=keep_batchnorm_fp32, verbosity=1)

    criterion = nn.modules.loss.BCEWithLogitsLoss(reduction='mean')

    n_epochs = sum(2**i for i in range(args.n_cycles))

    meta = {
        'train': {
            'loss': [],
            'accuracy': [],
        },
        'val': {
            'iteration': [],
            'loss': [],
            'accuracy': [],
        },
        'iteration': [0],
        'lr': []
    }

    best_acc = 0
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch+1, n_epochs))
        print('Training...')
        train_epoch(model, criterion, optim, lr_scheduler, trainloader, meta)
        print('Validating...')
        val(model, criterion, valloader, meta, args.amp)
        if meta['val']['accuracy'][-1] > best_acc:
            best_acc = meta['val']['accuracy'][-1]
            state = model.state_dict(),

            ckpt_path = args.path_to_models
            if not os.path.isdir(ckpt_path):
                os.makedirs(ckpt_path)
            torch.save(state, os.path.join(ckpt_path, str(epoch)+'_ckpt.pth'))

    meta['iteration'] = meta['iteration'][1:]
    with open(os.path.join(ckpt_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
