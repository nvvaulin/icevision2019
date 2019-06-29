from efficientnet_pytorch import EfficientNet
from torch import nn


class SignModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.cls = nn.Linear(448, n_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.cls(x)
        return x
