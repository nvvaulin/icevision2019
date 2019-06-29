from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    RandomAffine,
    Normalize
)

from augmentations import get_augmentations


def get_preprocessing(input_shape=(64, 64),
                      mean=(0.485, 0.456, 0.406),
                      std=[0.229, 0.224, 0.225],
                      train=False):

    resize = Resize(input_shape)
    normalize = Normalize(mean, std)

    if train:
        return Compose([
            resize,
            get_augmentations(),
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            resize,
            ToTensor(),
            normalize
        ])
