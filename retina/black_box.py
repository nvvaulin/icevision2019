import logging
import os
from itertools import chain
from time import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose

import config as cfg
from encoder import DataEncoder
from retinanet import RetinaNet

val_transform = Compose([
    ToTensor(), Normalize(cfg.mean, cfg.std)
])


def draw_prediction(img, bboxes, scores, labels):
    img = np.array(img)
    import cv2
    # cv2.line(img, (0, 800), (2448, 800), (0, 0, 255), 2)
    # cv2.line(img, (0, 1056), (2448, 1056), (0, 0, 255), 2)
    #
    # cv2.line(img, (0, 600), (2448, 600), (0, 255, 255), 2)
    # cv2.line(img, (0, 1112), (2448, 1112), (0, 255, 255), 2)

    for b, l, s in zip(bboxes, labels, scores):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        cv2.putText(img, '{} ({:d})'.format(l, int(100 * s)), (int(b[0]), int(b[1]) - 10), 1, 2, (0, 0, 0), 6)
        cv2.putText(img, '{} ({:d})'.format(l, int(100 * s)), (int(b[0]), int(b[1]) - 10), 1, 2, (255, 255, 255), 3)
    return Image.fromarray(img)


def get_crops(img, x_from, y_from, x_to, y_to, size, step, scale):
    size_to = int(.5 + size * scale)
    for y in range(y_from, y_to, step):
        for x in range(x_from, x_to, step):
            crop = img.crop((x, y, x + size, y + size)).resize((size_to, size_to))
            yield crop, (x, y, x, y), scale


def crop(img):
    small_crops = get_crops(img, -128, 800, 2400, 900, 256, 256, 2)
    middle_crops = get_crops(img, -256, 600-256*2, 2200, 700, 512, 256, 1)
#     big_crops = get_crops(img, -256, -256, 2000, 1000, 1024, 512, 0.5)
#     print(list(middle_crops))
#     return zip(*chain(middle_crops))
#     return zip(*chain(small_crops, middle_crops, big_crops))
    return zip(*chain(small_crops, middle_crops))


def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def run_prediction_batched(net, crops, shifts, scales):
    transformed = [val_transform(crop) for crop in crops]
    ti = torch.stack(transformed).half().to(net.device)
    loc_preds, cls_preds = net(ti)
    boxes, labels, scores = DataEncoder().decode(loc_preds, cls_preds, input_size=(512, 512), device=net.device,
                                                 shifts=shifts, scales=scales)
    if boxes is None:
        return None, None, None
    return boxes, labels, scores


def crop_prediction(crop, net):
    transformed = val_transform(crop)[None, ...]
    ti = transformed.half().to(net.device)
    #     ti = transformed.to(net.device)
    loc_preds, cls_preds = net(ti)
    shifts = torch.FloatTensor([0]).to(net.device)
    scales = torch.FloatTensor([1]).to(net.device)
    boxes, labels, scores = DataEncoder().decode(loc_preds, cls_preds, input_size=(512, 512), device=net.device,
                                                 shifts=shifts, scales=scales)
    if boxes is None:
        return None, None, None
    keep = nms(boxes, scores.flatten(), thresh=0.2)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    return boxes, labels, scores


def hires_prediction(img, net, verbose=True):
    tik = time()
    crops, shifts, scales = crop(img)
    shifts = torch.FloatTensor(shifts).to(net.device)
    scales = torch.FloatTensor(scales).to(net.device)
    tok = time()
    if verbose:
        print('{:d}ms for cropping {} patches'.format(int(1000 * (tok - tik)), len(crops)))

    tik = time()
    s = 10

    res = []
    for i in range(0, len(scales), s):
        if verbose:
            print(i, i + s)
        boxes, labels, scores = run_prediction_batched(net, crops[i:(i + s)], shifts[i:(i + s)], scales[i:(i + s)])
        if boxes is not None:
            res.append([boxes, labels, scores])
    if not res:
        return None, None, None
    boxes, labels, scores = map(np.concatenate, zip(*res))
    tok = time()
    if verbose:
        print('{:d}ms for predicting'.format(int(1000 * (tok - tik))))

    if boxes is None:
        return None, None, None

    tik = time()
    keep = nms(boxes, scores.flatten(), thresh=0.3)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    tok = time()
    if verbose:
        print('{:d}ms for nms'.format(int(1000 * (tok - tik))))

    return boxes, labels, scores


class RetinaDetector:
    def __init__(self, device='cuda', verbose=False):
        self.verbose = verbose
        self.net = RetinaNet(backbone=cfg.backbone, num_classes=2, pretrained=False)
        checkpoint = torch.load(os.path.join('ckpts', 'retina_fp16_2e4', '12_ckpt.pth'), map_location=device)
        errors = self.net.load_state_dict(checkpoint['net'])
        logging.warning('Errors from loading Retina model: {}'.format(errors))
        self.net = self.net.half().eval().to(device)
        self.net.device = device

    def detect(self, img):
        with torch.no_grad():
            boxes, labels, scores = hires_prediction(img, self.net, verbose=self.verbose)
        return boxes, labels, scores
