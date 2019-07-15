import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import *
from efnet import EfficientNet
from itertools import chain

import sys
torch.backends.cudnn.enabled = True



def classification_layer_init(layer, pi=0.01):
    nn.init.normal_(layer.weight, mean=0, std=0.01)
    nn.init.constant_(layer.bias, val=-math.log((1 - pi) / pi))

def init_conv_weights(layer):
    nn.init.normal_(layer.weight, mean=0, std=0.01)
    nn.init.constant_(layer.bias, val=0)
    return layer

def conv1x1(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)
    return layer

def conv3x3(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer


def conv3x3_bn(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    bn = nn.GroupNorm(16, out_channels)
    return layer, bn


def upsample(feature, sample_feature, scale_factor=2):
    out_channels=sample_feature.size()[1:]
    return F.upsample(feature,scale_factor=scale_factor)


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias



class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()

        self.resnet = resnet
# 56, 160, 448 b4
# 40 112 320
        # b1
        # l1, l2, l3 = 40, 112, 320
        # b4
#         l1, l2, l3 = 56, 160, 448
        self.pyramid_transformation_3 = conv1x1(512, 256)
        # self.pyramid_transformation_3 = conv1x1(l1, 256)
#         self.pyramid_transformation_3 = conv1x1(56, 256)

        # self.pyramid_transformation_4 = conv1x1(l2, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
#         self.pyramid_transformation_4 = conv1x1(160, 256)

        # self.pyramid_transformation_5 = conv1x1(l3, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)
#         self.pyramid_transformation_5 = conv1x1(448, 256)

        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
#         self.pyramid_transformation_6 = conv3x3(l3, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)

        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)

        self.dropout = nn.Dropout(p=0.5)
        # self.dropblock_3 = LinearScheduler(DropBlock2D(block_size=4, drop_prob=0), 0, 0.1, 2000)
        # self.dropblock_4 = LinearScheduler(DropBlock2D(block_size=2, drop_prob=0), 0, 0.1, 1000)
        # self.dropblock_5 = LinearScheduler(DropBlock2D(block_size=2, drop_prob=0), 0, 0.1, 500)


    def forward(self, x):
        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)
#         resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        # self.dropblock_3.step()
        # # print(self.dropblock_3.dropblock.drop_prob)
        # self.dropblock_4.step()
        # self.dropblock_5.step()

        resnet_feature_3=self.dropout(resnet_feature_3)
        resnet_faeture_4=self.dropout(resnet_feature_4)
        resnet_feature_5=self.dropout(resnet_feature_5)

        pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)

        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        upsampled_feature_5 = upsample(pyramid_feature_5, pyramid_feature_4)
        pyramid_feature_4 = self.upsample_transform_1(torch.add(upsampled_feature_5, pyramid_feature_4))

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        upsampled_feature_4 = upsample(pyramid_feature_4, pyramid_feature_3)
        pyramid_feature_3 = self.upsample_transform_2(torch.add(upsampled_feature_4, pyramid_feature_3))

        return pyramid_feature_3, pyramid_feature_4, pyramid_feature_5, pyramid_feature_6, pyramid_feature_7


class SubNet(nn.Module):
    def __init__(self, k, anchors=9, depth=4, cls=False, activation=F.relu):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.activation = activation

        drop_prob = 0.1 if cls else 0
        drop_prob = 0
        # self.dropout = LinearScheduler(DropBlock2D(block_size=3, drop_prob=0), 0, drop_prob, 1000)
        self.dropout = nn.Dropout(p=0.5)
        mod_list = list(conv3x3(256, 256, padding=1) for _ in range(depth))
        # print(mod_list)
        self.base = nn.ModuleList(mod_list)

        self.output = nn.Conv2d(256, k * anchors, kernel_size=3, stride=1,padding=1)

        self.bn=nn.BatchNorm2d(256)
        self.gn=GroupNorm(256, 16)

        if cls:
            classification_layer_init(self.output)
        else:
            init_conv_weights(self.output)


    def forward(self, x):
        for layer in self.base:
            x = layer(x)
#             x = self.bn(x)
            x = self.gn(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.output(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2) * x.size(3) * self.anchors, -1)
        return x


class RetinaNet(nn.Module):
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
    }

    def __init__(self, backbone='resnet101', num_classes=1, pretrained=True):
        super().__init__()
#         super(RetinaNet, self).__init__()

        self.resnet = RetinaNet.backbones[backbone](pretrained=pretrained)
#         self.resnet = EfficientNet.from_pretrained('efficientnet-b4', 1)
        # self.resnet.load_pretrained_weights
        # print(self.resnet._bn0)

        self.feature_pyramid = FeaturePyramid(self.resnet)

        self.subnet_classes = SubNet(k=num_classes,cls=True)
        self.subnet_boxes = SubNet(4)

    def forward(self, x):
        pyramid_features = self.feature_pyramid(x)
        class_predictions = [self.subnet_classes(p) for p in pyramid_features]
        bbox_predictions = [self.subnet_boxes(p) for p in pyramid_features]
        return torch.cat(bbox_predictions, 1), torch.cat(class_predictions, 1)
