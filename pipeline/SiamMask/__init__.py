import os,sys
sys.path = [os.path.dirname(__file__),os.path.join(os.path.dirname(__file__),'experiments/siammask_sharp')]+sys.path
import glob
from tools.test import *
from easydict import EasyDict
from custom import Custom
import numpy as np

class SiamTracker(object):
    def __init__(self,resume=os.path.join(os.path.dirname(__file__),'experiments/siammask_sharp/SiamMask_DAVIS.pth'),\
                 config=os.path.join(os.path.dirname(__file__),'experiments/siammask_sharp/config_davis.json')):
        self.args = EasyDict(resume=resume,
                             config=config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        # Setup Model
        self.cfg = load_config(self.args)
        self.siammask = Custom(anchors=self.cfg['anchors'])
        assert isfile(self.args.resume), 'Please download {} first.'.format(self.args.resume)
        self.siammask = load_pretrain(self.siammask, self.args.resume)
        self.siammask.eval().to(self.device)
        
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
    
    def track(self,state,im):
        new_state = siamese_track(state, im,self.siammask, mask_enable=False, refine_enable=False, device=self.device)
        new_state['track_id'] = state['track_id']        
        p = new_state['target_pos']
        sz = new_state['target_sz']
        new_state['box'] = np.concatenate((p-sz/2,p+sz/2))
        return new_state
    

def IoU(a,b):
    aa = (a[:,2:]-a[:,:2])
    aa = aa[:,0]*aa[:,1]
    ab = (b[:,2:]-b[:,:2])
    ab = ab[:,0]*ab[:,1]
    
    a = a[:,None,:]+np.zeros_like(b)[None,:,:]
    b = b[None,:,:]+np.zeros_like(a)
    i = np.clip((np.minimum(a[:,:,2:],b[:,:,2:]) - np.maximum(a[:,:,:2],b[:,:,:2])),0,1e10)
    i = i[:,:,0]*i[:,:,1]
    return np.clip(i/(aa[:,None]+ab[None,:]-i),0,1)