import pandas as pd
import numpy as np
from PIL import Image
import os

def iterate_from_log(root,log_path):
    '''
    yield bbox_iterator yields 
    '''
    df = pd.read_csv(log_path)
    cols = ['xtl','ytl','xbr','ybr','score','track','class']
    cols = [i for i in cols if i in df.columns]
    for imname,bbox in df.groupby('imname'):
        img = Image.open(os.path.join(root,imname))
        yield imname,img,bbox[cols].values.astype(np.float32)
        

def iterate_log(iterator,log_path,t='w'):
    '''
    yield bbox_iterator yields 
    '''
    log = open(log_path,t)
    cols = ['xtl','ytl','xbr','ybr','score','track','class']
    for i,(imname,img,bbox) in enumerate(iterator):
        if i == 0:
            log.write(','.join(['imname']+[cols[j] for j in range(bbox.shape[1])]))
        log.write('\n'+'\n'.join([','.join([imname]+['%.3f'%j for j in box]) for box in bbox]))
        log.flush()            
        yield imname,img,bbox

def iterate_imgs(root,imlist,**kwargs):
    '''
    yield imname,img(PIL)
    '''
    for imname in imlist:
        yield imname,Image.open(os.path.join(root,imname))
    
def iterate_detector(img_iterator,**kwargs):
    '''
    yield imname,img,bboxes(x1,y1,x2,y2,score)
    '''    
    
    
def iterate_tracker(detector_iterator,**kwargs):
    '''
    yield imname,img,bboxes(x1,y1,x2,y2,score,track_id)
    '''    
    for imname,img,bboxes in track_iterator:
        yield imname,img,np.concatenate((bboxes,np.ones((len(bboxes),1))),1)
        
# def iterate_async(iterator):
#     pass
    
def iterate_classify(track_iterator,**kwargs):
    '''
    yield imname,img,bboxes(x1,y1,x2,y2,score,track_id,class)
    '''
    for imname,img,bboxes in track_iterator:
        yield imname,img,np.concatenate((bboxes,np.ones((len(bboxes),1))),1)
        
#     batch = []
#     paths = []
#     for path,im,bboxes in track_iterator:
#         for bbox in bboxes:
#             paths.append(path)
#             batch.append(   

def iterate_video(bbox_iterator,out_path,**kwargs):
    '''
    write video, drow boxes,tracks,classes,scores if available
    yield bbox_iterator yields 
    '''
    
    
def main(imglist):
    it = iterate_imgs(root,imlist)
    it = iterate_log(iterate_detector(it))
    it = iterate_log(iterate_tracker(it))
    it = iterate_log(iterate_classify(it))
    it = video_iterator(it)
    for i in it:
        pass
    