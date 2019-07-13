import numpy as np
from glob import glob
from PIL import Image

def t2np(x):
    return x.to('cpu').detach().numpy()


def unnormalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return ((t2np(t).transpose((1, 2, 0)) * std + mean) * 255).astype('uint8')


def np_softmax(y_hat):
    y_hat = y_hat - y_hat.max(-1, keepdims=True)
    return np.exp(y_hat) / np.exp(y_hat).sum(-1, keepdims=True)


def get_label_examples(label, path='classification_data/train/{}/*.jpg'):
    impaths = glob(path.format(label))
    canvas = np.zeros((64, 64*len(impaths[:5]), 3), dtype=np.uint8)
    for i, impath in enumerate(impaths[:5]):
        img = np.array(Image.open(impath).resize((64, 64)))
        canvas[:, i*64:(i+1)*64] = img
    return Image.fromarray(canvas), len(impaths),
