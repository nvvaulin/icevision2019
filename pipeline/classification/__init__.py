import torch
import os
from .model import SignModel
from .data import get_preprocessing
from .utils import t2np,np_softmax
import pickle
from multiprocessing.dummy import Pool
import numpy as np

class SignClassifier(object):
    input_shape = (64,64)
    def __init__(self,ch_path='class_ckpts/init/6_ckpt.pth',cl_path='id2class.pkl',pool_size=4):
        self.classes = pickle.load(open(os.path.join(os.path.dirname(__file__),cl_path),'rb'))
        self.transform = get_preprocessing(input_shape=SignClassifier.input_shape)
        self.model = SignModel(n_classes=len(self.classes)).half().to('cuda').eval()
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__),ch_path))
        self.model.load_state_dict(checkpoint[0])
        self.pool = Pool(pool_size)
        for i,c in enumerate(self.classes):
            if c == 'other':
                self.other_inx = i
        
    def predict(self,imgs):
        x = self.pool.map(lambda i: self.transform(i).unsqueeze(0), imgs)
        x = torch.cat(x).half().to('cuda')
        sf = np_softmax(t2np(self.model(x)))
        cl = sf.argmax(1)
        sf = sf.max(1)
        sf[cl==self.other_inx] = -sf[cl==self.other_inx]
        return np.concatenate((cl[:,None],sf[:,None]),1)
    
    @staticmethod
    def crop(img,bbox):
        return img.crop(tuple(bbox.astype(np.int)))