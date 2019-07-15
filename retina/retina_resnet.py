import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch import nn

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlockFeatures(BasicBlock):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     print(args, kwargs)
    #     self.bn1 = nn.GroupNorm(16, self.conv1.out_channels)
    #     self.bn2 = nn.GroupNorm(16, self.conv2.out_channels)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        conv2_rep = out
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv2_rep


class BottleneckFeatures(Bottleneck):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.bn1 = nn.GroupNorm(16, self.conv1.out_channels)
    #     self.bn2 = nn.GroupNorm(16, self.conv2.out_channels)
    #     self.bn3 = nn.GroupNorm(16, self.conv3.out_channels)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        conv3_rep = out
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv3_rep


class ResNetFeatures(ResNet):


    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     print(self.conv1)
    #     self.bn1 = nn.GroupNorm(8, self.conv1.out_channels)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, c2 = self.layer1(x)
        x, c3 = self.layer2(x)
        x, c4 = self.layer3(x)
        x, c5 = self.layer4(x)

        return c2, c3, c4, c5



def resnet18(pretrained=False, **kwargs):
    model = ResNetFeatures(BasicBlockFeatures, [2, 2, 2, 2], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNetFeatures(BasicBlockFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 23, 3], **kwargs)
    # checkpoint = torch.load(os.path.join('ckpts', 'resnet101', '29_ckpt.pth'), map_location='cuda')
    # model_state  = net.state_dict()
    # # checkpoint = torch.load('efnet_detector_wo_cls.pth', map_location='cuda')
    # model_state.update(checkpoint['net'])
    # net.load_state_dict(model_state)
    # net.load_state_dict(checkpoint)
    # start_epoch = checkpoint['epoch']
    if pretrained:
        # print('')
        # model_state = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet101'])
        # checkpoint = {k: v for k, v in checkpoint.items() if 'bn' not in k}
        # model_state.update(checkpoint)
        model.load_state_dict(checkpoint)

    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNetFeatures(BottleneckFeatures, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model
