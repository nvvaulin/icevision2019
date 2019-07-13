import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pickle
from apex import amp
from glob import glob
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
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
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--init_lr', default=0.01, type=float)
    parser.add_argument('--n_cycles', default=3, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--amp', default='O2', type=str)
    parser.add_argument('--n_workers', default=10, type=int)
    parser.add_argument('--train_folder', default='classifications_data/train', type=str)
    parser.add_argument('--val_folder', default='classification_data/val', type=str)
    parser.add_argument('--label_smoothing', default=0, type=float)
    return parser.parse_args()


def train_epoch(model, criterion, optim, lr_scheduler, loader, meta):
    model.train()
    tik = time()
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to('cuda'), y.type(torch.LongTensor).to('cuda')

        optim.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)

        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()

        optim.step()
        lr_scheduler.step()

        accuracy = (t2np(y_hat.argmax(-1).flatten()) == t2np(y.flatten())).mean()
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
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to('cuda')
        if amp != '01':
            x = x.half()
        y = y.type(torch.LongTensor).to('cuda')

        y_hat = model(x)
        loss = criterion(y_hat, y)

        accuracy.extend(t2np(y_hat.argmax(-1).flatten()) == t2np(y.flatten()))
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
    trainset = ImageFolder(args.train_folder, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    val_transform = get_preprocessing(train=False)
    valset = ImageFolder(args.val_folder, transform=val_transform)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    model = SignModel(n_classes=len(trainset.classes)).to('cuda')
    optim = SGD(params=model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=True)
    lr_scheduler = CosineAnnealingWarmRestarts(optim, T_0=len(trainloader), T_mult=2)

    if args.amp in ('02', '03'):
        keep_batchnorm_fp32 = True
    else:
        keep_batchnorm_fp32 = None
    model, optim = amp.initialize(model, optim, opt_level=args.amp, keep_batchnorm_fp32=keep_batchnorm_fp32, verbosity=1)

    if args.label_smoothing == 0:
        criterion = nn.modules.loss.CrossEntropyLoss(reduction='mean')
    else:
        criterion = LabelSmoothingLoss(args.label_smoothing, len(valset.classes))

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

            ckpt_path = os.path.join('class_ckpts', args.experiment)
            if not os.path.isdir(ckpt_path):
                os.makedirs(ckpt_path)
            torch.save(state, os.path.join(ckpt_path, str(epoch)+'_ckpt.pth'))

    meta['iteration'] = meta['iteration'][1:]
    with open(os.path.join(ckpt_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
