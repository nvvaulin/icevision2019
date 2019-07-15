import torch
import os
from .model import SignModel
from .data import get_preprocessing
from .utils import t2np,np_softmax
import pickle
from multiprocessing.dummy import Pool
import numpy as np
import json

CLASSES_REMAPPING_FILE = os.path.join(os.path.dirname(__file__), '../../classification/remapping.json')

class SignClassifier(object):
    input_shape = (64,64)
    def __init__(self,ch_path='class_ckpts/init/6_ckpt.pth',cl_path='id2class.pkl',pool_size=4):
        with open(CLASSES_REMAPPING_FILE) as f:
            self.classes = json.load(f)
        self.class_names = [elem[0] for elem in sorted(self.classes.iteritems(), key=lambda x: x[1])]

        self.transform = get_preprocessing(input_shape=SignClassifier.input_shape)
        self.model = SignModel(n_classes=len(self.classes)).half().to('cuda').eval()
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__),ch_path))
        self.model.load_state_dict(checkpoint[0])
        self.pool = Pool(pool_size)

    def predict(self,imgs):
        with torch.no_grad():
            x = self.pool.map(lambda i: self.transform(i).unsqueeze(0), imgs)
            x = torch.cat(x).half().to('cuda')
            class_probs = t2np(torch.sigmoid(self.model(x)))

        crop_class = class_probs.argmax(1)
        class_probs = class_probs.max(1)
        class_probs[class_probs < 0.5] *= -1
        return np.concatenate((crop_class[:,None],class_probs[:,None]),1)

    @staticmethod
    def crop(img,bbox):
        return img.crop(tuple(bbox.astype(np.int)))
