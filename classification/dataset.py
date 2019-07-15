import torch.utils.data
import numpy as np
import cv2
from PIL import Image
import torch
import json
import imageio

REMAP = '/media/storage/vyurchenko/projects/ice/rep/classification/remapping.json'
N_CLASSES=231

N_TRAIN = 142838
N_VAL = 4278


class FuckingDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset, transforms):
        with open(path_to_dataset) as f:
            self.data = json.load(f)
        with open(REMAP) as f:
            self.remapping = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data[index]['image'].endswith('pnm'):
            im = imageio.imread(self.data[index]['image'])
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)
        else:
            im = cv2.imread(self.data[index]['image'])[:,:,::-1]

        x0, y0 = self.data[index]['x0'], self.data[index]['y0']
        x1, y1 = self.data[index]['x1'], self.data[index]['y1']

        patch = im[y0:y1, x0:x1].copy()
        cl = self.data[index]['class'].split('.')
        cl = ['.'.join(cl[:i]) for i in range(1, len(cl)+1)]
        cl = [self.remapping[elem] for elem in cl]
        gt = np.zeros(N_CLASSES, np.float32)
        gt[cl] = 1.0
        patch = Image.fromarray(patch)

        patch = self.transforms(patch)
        gt = torch.Tensor(gt)
        return {'image': patch, "classes": gt,
                'is_temporary': self.data[index]['is_temporary']}
