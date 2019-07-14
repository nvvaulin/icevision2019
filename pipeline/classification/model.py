from efficientnet_pytorch import EfficientNet
from torch import nn
from efficientnet_pytorch.utils import relu_fn

def get_feature_layer(self, inputs):
    """ Returns output of the final convolution layer """
    x = relu_fn(self._bn0(self._conv_stem(inputs)))
    
    for idx, block in enumerate(self._blocks):
        drop_connect_rate = self._global_params.drop_connect_rate
        if drop_connect_rate:
            drop_connect_rate *= float(idx) / len(self._blocks)
        x = block(x, drop_connect_rate=drop_connect_rate)
    return x


class SignModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.backbone.extract_features = lambda inputs: get_feature_layer(self.backbone, inputs)
        self.cls = nn.Linear(448, n_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.cls(x)
        return x
