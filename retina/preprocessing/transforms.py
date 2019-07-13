import numpy as np
import random
from PIL import Image

import torch

from copy import deepcopy
from preprocessing.bbox import BoundingBox

import sys

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        for t in self.transforms:
            example = t(example)
        return example


class ResizeKeepAR:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, example):
        image = example['image']
        im_w, im_h = image.size

        if im_w < self.w or im_h < self.h:
            if im_h < im_w:
                scale = self.h / im_h
                image = image.resize((int(0.5 + im_w * scale), self.h))
            else:
                scale = self.w / im_w
                image = image.resize((self.w, int(0.5 + im_h * scale)))

            new_boxes = []
            for box in example['boxes']:
                new_boxes.append(
                    BoundingBox(
                        box.left * scale,
                        box.top * scale,
                        box.right * scale,
                        box.bottom * scale,
                        image.size[0],
                        image.size[1],
                        box.label
                    )
                )
            boxes = new_boxes

            return {'image': image, 'boxes': boxes, 'labels': example['labels']}
        return example



class RandomCrop:
    def __init__(self, w, h, s, min_area, min_visibility):
        self.w = w
        self.h = h
        self.s = s
        self.min_area = min_area
        self.min_visibility = min_visibility

    @staticmethod
    def _iom(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxBArea)

        # return the intersection over union value
        return iou


    def __call__(self, example):
        image = example['image']
        boxes = example['boxes']
        labels = example['labels']
        im_w, im_h = image.size

        try:

#         print(im_w, im_h)
            box = np.random.choice(boxes)
            from copy import deepcopy
            orig = deepcopy(box)

            if (box.right - box.left) > self.w:
                x_from = max(0, box.left-self.s)
                x_to = x_from + 1
            else:

                x_from = max(0, box.right - self.w)
                x_to = min(im_w - self.w, box.left) + 1

    #         print(x_from, x_to)
            x = np.random.randint(int(x_from), int(x_to))

            if (box.bottom - box.top) > self.h:
                y_from = max(0, box.top-self.s)
                y_to = y_from + 1
            else:
                y_from = max(0, box.bottom - self.h)
                y_to = min(im_h - self.h, box.top)  + 1

    #         print(y_from, y_to)
            y = np.random.randint(int(y_from), int(y_to))
        except:
            print('bbox:', box.left, box.top, box.right, box.bottom)
            print('im_h: {}, im_w: {}'.format(im_h, im_w))
            print('x:')
            print('x_from; max(0,',box.right - self.w)
            print('x_to: min',im_w - self.w, box.left)
            print(x_from, x_to)
            print('y:')
            print('y_from: max(0,', box.bottom - self.h)
            print('y_to: min', im_h - self.h, box.top   )
            print(y_from, y_to)
            sys.stdout.flush()
        crop = [x, y, x+self.w, y+self.h]

        image = image.crop(crop)

        new_boxes = []
        new_labels = []
        for box, label in zip(boxes, labels):
            iom = RandomCrop._iom(crop, [box.left, box.top, box.right, box.bottom])
#             print(iom)

            if iom < self.min_visibility:
                continue

            new_boxes.append(
                BoundingBox(
                    max(0, box.left - x),
                    max(0, box.top - y),
                    min(self.w - 1, box.right - x),
                    min(self.h - 1, box.bottom - y),
                    self.w,
                    self.h,
                    label
                )
            )
            new_labels.append(label)
        if len(new_boxes) == 0:
            print(new_boxes)
            box = orig
            print('bbox:', box.left, box.top, box.right, box.bottom)
            print('im_h: {}, im_w: {}'.format(im_h, im_w))
            print('x:')
            print('x_from; max(0,',box.right - self.w)
            print('x_to: min',im_w - self.w, box.left)
            print(x_from, x_to)
            print('y:')
            print('y_from: max(0,', box.bottom - self.h)
            print('y_to: min', im_h - self.h, box.top   )
            print(y_from, y_to)
            sys.stdout.flush()

        return {'image': image, 'boxes': new_boxes, 'labels': new_labels}


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, example):
        image, boxes, labels = example['image'], example['boxes'], example['labels']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': image, 'boxes': boxes, 'labels': labels}


class Scale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, example):
        image, boxes = example['image'], example['boxes']

        width, height = image.size
        if isinstance(self.output_size, int):
            if width < height:
                new_width, new_height = width / height * self.output_size, self.output_size
            else:
                new_width, new_height = self.output_size, height / width * self.output_size
        else:
            new_width, new_height = self.output_size
        new_width, new_height = int(new_width), int(new_height)
        image=image.resize((new_width, new_height))
        boxes=[box.resize(new_width, new_height) for box in boxes]

        return {'image': image, 'boxes': boxes}


class RandomHorizontalFlip:
    def __call__(self, example):
        image, boxes = example['image'], example['boxes']
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [box.flip() for box in boxes]
        return {'image': image, 'boxes': boxes, 'labels': example['labels']}


class AdditiveGaussian:
    def __init__(self, scale=0.08, p=0.9):
        self.scale = scale
        self.p = p

    def __call__(self, example):
        image = np.array(example['image'])
        if np.random.binomial(1, self.p):
            image = image + np.random.normal(loc=0, scale=255*self.scale, size=image.shape)
            image = np.clip(image, 0, 255).astype(np.uint8)
        return {'image': Image.fromarray(image), 'boxes': example['boxes'], 'labels': example['labels']}



class ToTensor:
    def __call__(self, example):
        image, boxes = example['image'], example['boxes']
        image = np.array(image).transpose((2, 0, 1))
        labels = np.array([box.label for box in boxes])
        boxes = np.array([[box.left, box.top, box.right, box.bottom] for box in boxes])
        image, boxes, labels = torch.from_numpy(image), torch.from_numpy(boxes), torch.from_numpy(labels)
        if isinstance(image, torch.ByteTensor):
            image = image.float().div(255)
        return {'image': image, 'boxes': boxes, 'labels': labels}


class Unnormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
