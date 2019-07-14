import argparse
import os
import pickle
import sys
import time
from contextlib import contextmanager
from multiprocessing.dummy import Pool
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
from PIL import Image

from SiamMask import SiamMultiTracker
from classification import SignClassifier

sys.path.append('../retina')

from black_box import RetinaDetector


########################geretal iter tools#################################
def iterate_async(it):
    pool = Pool(1)
    i = next(it, None)
    while not (i is None):
        future = pool.apply_async(next, (it, None))
        yield i
        i = future.get()


def simulate_detector(it, stride=10):
    for i, (p, im, bboxes) in enumerate(it):
        if (i % stride) == 0:
            yield p, im, bboxes
        else:
            yield p, im, bboxes[:0, :]


def iterate_batched(it, batch_size):
    batch = []
    for i in it:
        batch.append(i)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def iterate_unbatch(it):
    for batch in it:
        for i in batch:
            yield i


def iterate_box(img_iterator):
    for imname, im, bboxes in img_iterator:
        for box in bboxes:
            yield imname, im, box


def iterate_img(box_iterator, box_len=-1):
    cur_imname = ''
    cur_im = None
    bboxes = None
    for imname, im, box in box_iterator:
        if imname != cur_imname:
            if cur_im != None:
                bboxes = np.array(bboxes, dtype=np.float32)
                yield cur_imname, cur_im, bboxes
            bboxes = []
            cur_imname = imname
            cur_im = im
        bboxes.append(box)

    if cur_im is not None:
        bboxes = np.array(bboxes, dtype=np.float32)
        yield cur_imname, cur_im, bboxes


def iterate_profiler(it, name, log_stap=10):
    t1 = time.time()
    tit = []
    ty = []
    num_bbox = []
    for j, i in enumerate(it):
        num_bbox.append(len(i[-1]))
        t2 = time.time()
        tit.append(t2 - t1)
        yield i
        t1 = time.time()
        ty.append(t1 - t2)
        if len(ty) >= log_stap:
            print('time %s %f, yield %f, num_boxes %f' % (
                name,
                np.array(tit).mean(),
                np.array(ty).mean(),
                np.array(num_bbox).mean(),
            ))
            num_bbox = []
            tit = []
            ty = []


########################tracker#################################
def iterate_tracker(img_iterator, multi_tracker, **kwargs):
    '''
    yield imname,img,bboxes(x1,y1,x2,y2,score,class,sclass_score,track_id,track_score)
    '''
    for imname, im, bboxes in img_iterator:
        yield imname, im, multi_tracker.update(im, bboxes)


def iterate_remove_tracks(img_iterator, multi_tracker, min_score=0.):
    for imname, im, bboxes in img_iterator:
        remove_ids = bboxes[:, 7][bboxes[:, 6] < min_score]
        multi_tracker.remove_tracks(remove_ids)
        yield imname, im, bboxes


def iterate_classifier_by_img(bbox_iterator, batch_size=8):
    classifier = SignClassifier(ch_path='6_ckpt.pth')

    for imname, im, bboxes in bbox_iterator:
        if len(bboxes) == 0:
            if bboxes.shape[1] < 7:
                bboxes = np.concatenate((bboxes, bboxes[:, :2]), 1)
            yield imname, im, bboxes
        else:
            crops = [SignClassifier.crop(im, box[:4]) for box in bboxes]
            p = [classifier.predict(b) for b in iterate_batched(crops, batch_size)]
            p = np.concatenate(p)
            if bboxes.shape[1] >= 7:
                bboxes[:, 5:7] = p
            else:
                bboxes = np.concatenate((bboxes, p), 1)
            yield imname, im, bboxes


def iterate_classifier(bbox_iterator, batch_size=128):
    classifier = SignClassifier(ch_path='6_ckpt.pth')

    def iterate_crop(batch_box_iterator):
        for batch in batch_box_iterator:
            yield batch, [SignClassifier.crop(im, box[:4]) for imname, im, box in batch]

    def iterate_batch_prediction(crop_iterator):
        for batch, crops in crop_iterator:
            pred = classifier.predict(crops)
            res_batch = []
            for (imname, im, box), p in zip(batch, pred):
                if len(box) >= 7:
                    box[5:7] = p
                else:
                    box = np.concatenate((box, p))
                res_batch.append((imname, im, box))
            yield res_batch

    it = iterate_box(bbox_iterator)
    it = iterate_batched(it, batch_size)
    it = iterate_crop(it)
    it = iterate_async(it)
    it = iterate_batch_prediction(it)
    it = iterate_unbatch(it)
    it = iterate_img(it)
    for i in it:
        yield i


def iterate_from_log(root, log_path):
    '''
    yield bbox_iterator yields
    '''
    df = pd.read_csv(log_path)
    cols = ['xtl', 'ytl', 'xbr', 'ybr', 'score', 'track', 'track_score', 'class', 'class_score']
    cols = [col for col in cols if col in df.columns]
    df['rinx'] = np.array([-int(i.split('.')[0]) for i in df.imname.values])

    all_images = sorted(Path(root).iterdir())

    for _, bbox in df.groupby('rinx', sort=True):
        imname = bbox['imname'].values[0]
        imname = all_images[int(imname.split('.')[0])]
        img = imageio.imread(imname)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        img = Image.fromarray(img)
        yield imname, img, bbox[cols].values.astype(np.float32)


def iterate_log(iterator, log_path, t='w'):
    '''
    yield bbox_iterator yields
    '''
    log = open(log_path, t)
    cols = ['xtl', 'ytl', 'xbr', 'ybr', 'score', 'class', 'class_score', 'track', 'track_score']
    for i, (imname, img, bbox) in enumerate(iterator):
        if i == 0:
            log.write(','.join(['imname'] + [cols[j] for j in range(bbox.shape[1])]))
        log.write('\n' + '\n'.join([','.join([imname] + ['%.3f' % j for j in box]) for box in bbox]))
        log.flush()
        yield imname, img, bbox


def iterate_imgs(root, imlist, **kwargs):
    '''
    yield imname,img(PIL)
    '''
    for imname in imlist:
        yield imname, Image.open(os.path.join(root, imname))


def iterate_detector(img_iterator, **kwargs):
    '''
    yield imname,img,bboxes(x1,y1,x2,y2,score)
    '''
    detector = RetinaDetector(**kwargs)
    for imname, img in img_iterator:
        boxes, labels, scores = detector.detect(img)
        assert boxes.shape[0] == scores.shape[0]
        assert boxes.shape[1] == 4
        yield imname, img, np.hstack([boxes, scores.reshape(-1, 1)])


def draw_results(img, bboxes):
    '''
    boxes: bbox,dscore,class,cscore,track,tscore
    '''
    img = np.array(img)[:, :, ::-1].copy()
    id2cl = pickle.load(open('classification/id2class.pkl', 'rb'))
    for box in bboxes:
        x1, y1, x2, y2 = tuple(box[:4].astype(np.int32))
        color = [0, 0, 255]
        text = str(int(box[4] * 10))
        if len(box) >= 7:
            cl = id2cl[int(box[5])]
            if cl != 'other':
                color[1] = 255
            text = text + ';' + cl + ' ' + str(int(box[6] * 10))
        if len(box) >= 9:
            if box[-1] > 0:
                color[0] = 255
            text = text + ';' + str(int(box[8] * 10))

        cv2.putText(img, text, (x1, y1), 1, 2, (0, 0, 0), 6)
        cv2.putText(img, text, (x1, y1), 1, 2, (255, 255, 255), 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return img


@contextmanager
def VideoWriter(*args, **kwargs):
    out = cv2.VideoWriter(*args, **kwargs)
    try:
        yield out
    finally:
        out.release()


def iterate_video(box_iterator, out_path, vsize=(1024, 1024)):
    '''
    boxes: bbox,dscore,class,cscore,track,tscore
    '''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    with VideoWriter(out_path, fourcc, 30.0, vsize) as out:
        for imname, im, bboxes in box_iterator:
            im = draw_results(im, bboxes)
            out.write(cv2.resize(im, vsize))
            yield imname, im, bboxes


def main(frames_path, log_path, video_path):
    mtracker = SiamMultiTracker()
    imlist = sorted(os.listdir(frames_path), reverse=True)
    it = iterate_imgs(frames_path, imlist)
    it = iterate_async(it)
    it = iterate_detector(it)
    it = iterate_profiler(it, 'load img', 100)
    it = iterate_classifier_by_img(it)
    it = iterate_profiler(it, 'classify', 100)
    it = iterate_async(it)
    it = iterate_profiler(iterate_tracker(it, mtracker), 'tracker', 100)
    it = iterate_classifier(it)
    it = iterate_profiler(it, 'classify_t', 100)
    it = iterate_remove_tracks(it, mtracker)
    it = iterate_log(it, 'log.csv')
    it = iterate_async(it)
    it = iterate_video(it, video_path)
    it = iterate_profiler(it, 'pipeline', 100)
    for i, (p, im, bboxes) in enumerate(it):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-path', help='Path to directory with frames', required=True)
    parser.add_argument('--log-path', help='Path to CSV with detector output')
    parser.add_argument('--video-path', help='Path where output video should be saved', required=True)
    args = parser.parse_args()
    main(args.frames_path, args.log_path, args.video_path)
