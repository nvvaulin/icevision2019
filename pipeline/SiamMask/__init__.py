import os
import sys

sys.path = [os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), 'experiments/siammask_sharp')] + sys.path
from tools.test import *
from easydict import EasyDict
from custom import Custom
import numpy as np


class SiamTracker(object):
    def __init__(self, resume=os.path.join(os.path.dirname(__file__), 'experiments/siammask_sharp/SiamMask_DAVIS.pth'), \
                 config=os.path.join(os.path.dirname(__file__), 'experiments/siammask_sharp/config_davis.json')):
        self.args = EasyDict(resume=resume,
                             config=config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        # Setup Model
        self.cfg = load_config(self.args)
        self.siammask = Custom(anchors=self.cfg['anchors'])
        assert isfile(self.args.resume), 'Please download {} first.'.format(self.args.resume)
        self.siammask = load_pretrain(self.siammask, self.args.resume)
        self.siammask = self.siammask.eval().half().to(self.device)
        
    def get_state(self,im,bbox):
        x,y = bbox[0],bbox[1]
        w,h = bbox[2]-x,bbox[3]-y
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        state = siamese_init(im, target_pos, target_sz, self.siammask, self.cfg['hp'], device=self.device)
        state['track_id'] = np.random.randint(1000000)
        state['box'] = bbox
        state['score'] = 1.
        return state

    def track(self, state, im):
        new_state = siamese_track(state, im, self.siammask, mask_enable=False, refine_enable=False, device=self.device)
        new_state['track_id'] = state['track_id']
        p = new_state['target_pos']
        sz = new_state['target_sz']
        new_state['box'] = np.concatenate((p - sz / 2, p + sz / 2))
        return new_state


def IoU(a, b):
    aa = (a[:, 2:] - a[:, :2])
    aa = aa[:, 0] * aa[:, 1]
    ab = (b[:, 2:] - b[:, :2])
    ab = ab[:, 0] * ab[:, 1]

    a = a[:, None, :] + np.zeros_like(b)[None, :, :]
    b = b[None, :, :] + np.zeros_like(a)
    i = np.clip((np.minimum(a[:, :, 2:], b[:, :, 2:]) - np.maximum(a[:, :, :2], b[:, :, :2])), 0, 1e10)
    i = i[:, :, 0] * i[:, :, 1]
    return np.clip(i / (aa[:, None] + ab[None, :] - i), 0, 1)


class SiamMultiTracker(object):
    def __init__(self, min_iou=0.5, min_det_conf=0.5, min_track_score=0.9, \
                 resume=os.path.join(os.path.dirname(__file__), 'experiments/siammask_sharp/SiamMask_DAVIS.pth'), \
                 config=os.path.join(os.path.dirname(__file__), 'experiments/siammask_sharp/config_davis.json')):
        self.tracker = SiamTracker(resume=resume, config=config)
        self.states = []
        self.min_iou = min_iou
        self.min_det_conf = min_det_conf
        self.min_track_score = min_track_score

    def update(self, img, bboxes):
        '''
        yield imname,img,bboxes(x1,y1,x2,y2,score,class,sclass_score,track_id,track_score)
        '''
        min_iou = self.min_iou
        min_det_conf = self.min_det_conf
        min_track_score = self.min_track_score
        np_img = np.array(img)[:, :, ::-1]

        # update
        self.states = [self.tracker.track(state, np_img) for state in self.states]
        self.states = [i for i in self.states if i['score'] > min_track_score]
        for i in self.states:
            i['input_box'] = []

        # find new
        if len(self.states) > 0 and len(bboxes) > 0:
            tboxes = np.array([i['box'] for i in self.states])
            iou = IoU(tboxes, bboxes[:, :4])
            for i, ii in enumerate(iou):
                if ii.max() > min_iou:
                    self.states[i]['input_box'] = bboxes[ii.argmax()]
            not_tracked_boxes = bboxes[(iou.max(0) < min_iou) & (bboxes[:, -1] <= min_det_conf)]
            new_track_boxes = bboxes[(iou.max(0) < min_iou) & (bboxes[:, -1] > min_det_conf)]
        else:
            new_track_boxes = bboxes[bboxes[:, -1] > min_det_conf]
            not_tracked_boxes = bboxes[(bboxes[:, -1] <= min_det_conf)]

        for box in new_track_boxes:
            state = self.tracker.get_state(np_img, box[:4])
            state['input_box'] = box
            self.states.append(state)

        # make result
        result = np.zeros((len(self.states) + len(not_tracked_boxes), not_tracked_boxes.shape[1] + 2),
                          dtype=np.float32) - 1.
        for i, s in enumerate(self.states):
            result[i][:4] = s['box']
            ibox = s['input_box']
            if len(ibox) > 4:
                result[i][4:len(ibox)] = ibox[4:]
            result[i][-2] = s['track_id']
            result[i][-1] = s['score']

        result[len(self.states):, :not_tracked_boxes.shape[1]] = not_tracked_boxes

        return result

    def remove_tracks(self, track_ids):
        track_ids = set([int(i) for i in track_ids])
        self.states = [s for s in self.states if not (int(s['track_id']) in track_ids)]
