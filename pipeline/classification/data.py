from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    RandomAffine,
    Normalize
)


def get_preprocessing(input_shape=(64, 64),
                      mean=(0.485, 0.456, 0.406),
                      std=[0.229, 0.224, 0.225]):

    resize = Resize(input_shape)
    normalize = Normalize(mean, std)
    return Compose([
        resize,
        ToTensor(),
        normalize,
    ])
