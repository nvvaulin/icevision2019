import argparse
import os
import pickle
import time
from multiprocessing.dummy import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from SiamMask import SiamTracker, IoU

from classification import SignClassifier


########################geretal iter tools#################################
def iterate_async(it):
    pool = Pool(1)
    i = next(it, None)
    while not (i is None):
        future = pool.apply_async(next, (it, None))
        yield i
        i = future.get()


def iterate_batched(it, batch_size):
    batch = []
    for i in it:
        batch.append(i)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) >= 0:
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

    if cur_im != None:
        bboxes = np.array(bboxes, dtype=np.float32)
        yield cur_imname, cur_im, bboxes


def iterate_profiler(it, name, log_stap=10):
    t1 = time.time()
    tit = []
    ty = []
    for j, i in enumerate(it):
        t2 = time.time()
        tit.append(t2 - t1)
        yield i
        t1 = time.time()
        ty.append(t1 - t2)
        if len(ty) >= log_stap:
            print('time %s %f, yield %f' % (name, np.array(tit).mean(), np.array(ty).mean()))
            tit = []
            ty = []


########################tracker#################################


def iterate_tracker(detector_iterator, min_iou=0.5, min_det_conf=0.9, min_track_score=0.95, **kwargs):
    '''
    yield imname,img,bboxes(x1,y1,x2,y2,score,class,sclass_score,track_id,track_score)
    '''
    tracker = SiamTracker()
    states = []
    for imname, img, bboxes in detector_iterator:
        np_img = np.array(img)[:, :, ::-1]

        # update
        states = [tracker.track(state, np_img) for state in states]
        states = [i for i in states if i['score'] > min_track_score]
        for i in states:
            i['input_box'] = []

        # find new
        if len(states) > 0 and len(bboxes) > 0:
            tboxes = np.array([i['box'] for i in states])
            iou = IoU(tboxes, bboxes[:, :4])
            for i, ii in enumerate(iou):
                if ii.max() > min_iou:
                    states[i]['input_box'] = bboxes[ii.argmax()]
            not_tracked_boxes = bboxes[(iou.max(0) < min_iou) & (bboxes[:, -1] <= min_det_conf)]
            new_track_boxes = bboxes[(iou.max(0) < min_iou) & (bboxes[:, -1] > min_det_conf)]
        else:
            new_track_boxes = bboxes[bboxes[:, -1] > min_det_conf]
            not_tracked_boxes = bboxes[(bboxes[:, -1] <= min_det_conf)]

        for box in new_track_boxes:
            state = tracker.get_state(np_img, box[:4])
            state['input_box'] = box
            states.append(state)

        # make result
        result = np.zeros((len(states) + len(not_tracked_boxes), not_tracked_boxes.shape[1] + 2), dtype=np.float32) - 1.
        for i, s in enumerate(states):
            result[i][:4] = s['box']
            ibox = s['input_box']
            if len(ibox) > 4:
                result[i][4:len(ibox)] = ibox[4:]
            result[i][-2] = s['track_id']
            result[i][-1] = s['score']

        result[len(states):, :len(box)] = not_tracked_boxes

        yield imname, img, result


def iterate_classifier(bbox_iterator, batch_size=128):
    classifier = SignClassifier(ch_path='6_ckpt.pth')

    def iterate_crop(batch_box_iterator):
        for batch in batch_box_iterator:
            yield batch, [SignClassifier.crop(im, box[:4]) for imname, im, box in batch]

    def iterate_batch_prediction(crop_iterator):
        for batch, crops in crop_iterator:
            pred = classifier.predict(crops)
            batch = [(imname, im, np.concatenate((box, p))) for (imname, im, box), p in zip(batch, pred)]
            yield batch

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
    cols = [i for i in cols if i in df.columns]
    df['rinx'] = np.array([-int(i.split('.')[0]) for i in df.imname.values])
    for _, bbox in df.groupby('rinx'):
        imname = bbox['imname'].values[0]
        img = Image.open(os.path.join(root, imname))
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


def iterate_video(box_iterator, out_path, vsize=(1024, 1024)):
    '''
    boxes: bbox,dscore,class,cscore,track,tscore
    '''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, vsize)
    for imname, im, bboxes in box_iterator:
        im = draw_results(im, bboxes)
        out.write(cv2.resize(im, vsize))
        yield imname, im, bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-path', default='data/2018-03-23_1352_right')
    parser.add_argument('--log-path', default='logs/2018-03-23_1352_right.csv')
    parser.add_argument('--video-path', default='2018-03-23_1352_right.avi')
    args = parser.parse_args()

    it = iterate_from_log(args.frames_path, args.log_path)
    it = iterate_async(it)
    it = iterate_profiler(it, 'load img', 100)
    it = iterate_classifier(it)
    it = iterate_profiler(it, 'classify', 100)
    it = iterate_box(it)
    it = iterate_img(it)
    it = iterate_async(it)
    it = iterate_profiler(iterate_tracker(it), 'tracker', 100)
    it = iterate_log(it, 'log.csv')
    it = iterate_async(it)
    it = iterate_video(it, args.video_path)
    it = iterate_profiler(it, 'pipeline', 100)
    for i, (p, im, bboxes) in enumerate(it):
        pass
