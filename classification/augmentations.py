import numpy as np
from PIL import Image
from torchvision.transforms import Compose

class GaussianNoise():
    def __init__(self, std=0.01, p=.9):
        self.std = std
        self.p = p

    def __call__(self, x):
        x = np.array(x).astype('float32')
        if np.random.binomial(1, self.p):
            x = x + np.random.normal(loc=0, scale=self.std*255, size=x.shape)
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def get_augmentations():
    return Compose([
        GaussianNoise(std=0.08, p=0.9),
    ])
