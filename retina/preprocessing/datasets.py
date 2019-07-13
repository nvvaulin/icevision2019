import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import config as cfg
from torch.utils.data.dataset import Dataset

from preprocessing.annotations import AnnotationDir
import preprocessing.transforms as transforms

import config as cfg
from glob import glob
import pandas as pd
import pickle
import cv2
import imagesize
from preprocessing.bbox import BoundingBox
import numpy as np

from joblib import Parallel, delayed

test_folders = ['2018-02-16_1515_left', '2018-03-16_1424_left', '2018-03-23_1352_right']

DIR = '/media/grisha/hdd1/icevision/'
from tqdm import tqdm

TASKS = {}

def update_task(path, tasks, label_map):
    t = pd.read_csv(path, sep='\t', dtype=str)
    if t.values.shape[0] == 0:
        return
    dataset_name = path.split('/')[5]
    start = path.find('annotations') + len('annotations') + 1
    # print(path, start, path[start:])
    task = os.path.join(dataset_name, os.path.splitext(path[start:])[0])
    # print(task)
    boxes = []
    labels = []
    ext = t.ext[0] if 'ext' in t else 'jpg'
    impath = os.path.join(DIR, dataset_name, 'images', path[start:].replace('.tsv', '.' + ext))

#    img = cv2.imread(impath)
#    height, width = img.shape[:2]
    for _, row in t.iterrows():
        if ((float(row.xbr) - float(row.xtl)) >=cfg.width) or ((float(row.ybr) - float(row.ytl))) >= cfg.height:
            continue
        class_ = row['class']
        if class_ not in label_map:
            label = 97
        else:
            label = label_map[class_]
        box = BoundingBox(
            max(0, float(row.xtl)),
            max(0, float(row.ytl)),
            min(10000, float(row.xbr)),
            min(10000, float(row.ybr)),
            -1,
            -1,
            label)
        boxes.append(box)
        labels.append(label)

    if len(boxes) == 0:
        return
    impath = os.path.join(DIR, dataset_name, 'images', path[start:].replace('.tsv', '.' + ext))
    if not os.path.exists(impath):
        print(impath)
        1/0
    tasks[task] = {
        'impath': impath,
        'boxes': boxes,
        'labels': [int(l) for l in labels],
    }



class IceDataset(Dataset):
    def __init__(self, encoder, transform=None, test=False, val=False):
#         self.class_map = {c: i for i, c in enumerate(['2.1', '2.4', '3.1', '3.24', '3.27', '4.1', '4.2', '5.19', '5.20', '8.22'], 1)}
#         self.class_map = defaultdict(lambda: 0, self.class_map)
#         self.categories = {v:k for k, v in self.class_map.items

        self.labels_paths = glob(os.path.join(DIR, '**/annotations/**/*.tsv'), recursive=True)
        self.labels_paths = [x for x in self.labels_paths if x.split('/')[5] in ('RTSD', 'ice')]
        # print(self.labels_paths)
        if val:
            self.labels_paths = [x for x in self.labels_paths if x.split('/')[-2] in test_folders]
        else:
            self.labels_paths = [x for x in self.labels_paths if x.split('/')[-2] not in test_folders]
        print(len(self.labels_paths))
        self.tasks = []

        with open(os.path.join(DIR, 'classes_map.pkl'), 'rb') as f:
            self.label_map = pickle.load(f)
        # print(self.label_map)
        self.val = val
        paths = []
        self.test = test
        self.encoder = encoder
        self.task_dict = {}
        self.transform = transform
        np.random.shuffle(self.labels_paths)
        print("FINDING TABLES")
        from time import time
        from functools import partial
        tik = time()
        # for path in tqdm(self.labels_paths[:3000]):
        # update_task(tasks=self.task_dict, path=path)

        fname = 'train.tasks' if not val else 'val.tasks'
        if not os.path.exists(fname):
            Parallel(n_jobs=8, require='sharedmem')(delayed(partial(update_task, tasks=self.task_dict, label_map=self.label_map))(path) for path in tqdm(self.labels_paths[:]))
            with open(fname, 'wb') as f:
                pickle.dump(self.task_dict, f)
        else:
            print('FOUND CACHE', fname)
            with open(fname, 'rb') as f:
                self.task_dict = pickle.load(f)
        tok = time()
        print('elapsed {:.1f}'.format((tok-tik)/60), 'minutes')
        #for path in tqdm(self.labels_paths):
            #print(val, path.split('/')[-2])
         #   pass
        self.tasks = list(self.task_dict.keys())

#         width_count = 0
#         height_count = 0
#         total_count = 0
#         for t, d in self.task_dict.items():
#             for b in d['boxes']:
#                 if (b.right - b.left) >= 512:
#                     width_count += 1
#                 if (b.bottom-b.top) >= 512:
#                     height_count+=1
#                 total_count += 1
#         print('\n\n\n')
#         print(width_count / total_count)
#         print(height_count / total_count)
#         print('\n\n\n')

        self.pil_transform = transforms.Compose([
            transforms.ResizeKeepAR(cfg.width, cfg.height),
            transforms.RandomCrop(cfg.width, cfg.height, 5, 100, 0.3),
        ])


        print("FOUND", len(self.tasks), " TABLES FOR {}".format("TRAIN" if not val else "TEST"))
        import sys
        sys.stdout.flush()


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.task_dict[self.tasks[idx]]
        image = Image.fromarray(cv2.imread(task['impath'])[...,::-1])
        boxes = task['boxes']
        labels = task['labels']

        width, height = image.size

        new_boxes = []
        for box in boxes:
            box = BoundingBox(
            max(0, float(box.left)),
            max(0, float(box.top)),
            min(width, float(box.right)),
            min(height, float(box.bottom)), -1, -1, 0)
            new_boxes.append(box)
        boxes = new_boxes


        example = {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }

        if not self.test:
            example = self.pil_transform(example)

        if self.transform is not None:
            example = self.transform(example)

        if True:

            pass

        return example


    def collate_fn(self, batch):
#        print(type(batch[0]['image']))
        if isinstance(batch[0]['image'], Image.Image):
            return batch

        imgs = [example['image'] for example in batch]
        if not self.test:
            boxes  = [example['boxes'] for example in batch]
            labels = [example['labels'] for example in batch]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
#        print(num_imgs, self.test)
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im
            if not self.test:
 #               print(labels[i])
                if boxes[i].shape[0] == 0:
                    continue
                try:
                    loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=[cfg.width,cfg.height])
                except:
                    print(boxes[i])
                    1/0
  #              print(cls_target.sum())
                loc_targets.append(loc_target)
                cls_targets.append(cls_target)
        if not self.test:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets).type(torch.LongTensor)
        return inputs



class VocLikeDataset(Dataset):

    def __init__(self, image_dir, annotation_file, imageset_fn, image_ext, classes, encoder, transform=None, test=False):
        self.image_dir_path = image_dir
        self.image_ext = image_ext
        self.filenames=imageset_fn
        if not test:
            self.annotation_dir = AnnotationDir(annotation_file,classes)

        self.encoder = encoder
        self.transform = transform
        self.test = test


    def __getitem__(self, index):

        image_fn = self.filenames[index]
        fn=os.path.join(self.image_dir_path.split('/')[-1],image_fn)
        image_path = os.path.join(self.image_dir_path, image_fn)
        image = Image.open(os.path.join('data', image_fn))
        example={}
        example['image']=image
        if not self.test:
            boxes = self.annotation_dir.get_boxes(image_fn.split('/')[-1])
            example['boxes']=boxes

        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return len(self.filenames)

    def collate_fn(self, batch):
        imgs = [example['image'] for example in batch]
        if not self.test:
            boxes  = [example['boxes'] for example in batch]
            labels = [example['labels'] for example in batch]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        print(num_imgs)
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im
            if not self.test:
                print(labels[i])
                loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=[cfg.width,cfg.height])
                loc_targets.append(loc_target)
                cls_targets.append(cls_target)
        if not self.test:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs
