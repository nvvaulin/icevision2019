# from collections import namedtuple
# from maskrcnn_benchmark.utils.registry import Registry
# from maskrcnn_benchmark.modeling.make_layers import group_norm
# from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from torch import nn
import torch
from torch import nn
from torch.nn import functional as F

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import relu_fn
import sys
sys.path.append('dropblock')
from dropblock import DropBlock2D, LinearScheduler



from efnet_utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, norm_func):
        super().__init__()

        self.dropblock = LinearScheduler(DropBlock2D(block_size=3, drop_prob=0), 0, 0.1, 1000)

        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            # self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            self._bn0 = norm_func(oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn1 = norm_func(oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn2 = norm_func(final_oup)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x




        x = self._bn2((self._project_conv(x)))
        # x = self._bn2(self.dropblock(self._project_conv(x)))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + (inputs)  # skip connection
        return x


# class EfficientNetFeaturesExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
#         # self.feature_layers = (4, 10, 15)
#         self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
#         self.feature_layers = (9, 21, 31)
#
#     def forward(self, inputs):
#         outputs = []
#
#         # Stem
#         x = relu_fn(self.backbone._bn0(self.backbone._conv_stem(inputs)))
#
#         # Blocks
#         for idx, block in enumerate(self.backbone._blocks):
#             drop_connect_rate = self.backbone._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self.backbone._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if idx in self.feature_layers:
#                 outputs.append(x)
#         return outputs


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args, global_params):
        super().__init__()

        Conv2d = get_same_padding_conv2d(image_size=None)

        # block_args, global_params = get_model_params(cfg.BACKBONE.CONV_BODY, override_params=False)

        self._blocks_args, self._global_params = blocks_args, global_params
        # self._blocks_args, self._global_params = get_model_params('efficientnet-b4', override_params=False)
        self._features_idx = set((9, 21, 31))
        # self._features_idx = set((7, 15, 22))
        self._fpn_in_channels = [40, 112, 320]
        self._fpn_out_channels = 320

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        #self._norm_func = lambda out: nn.GroupNorm(8, out)
        self._norm_func = lambda out: nn.BatchNorm2d(out)
        # if cfg.MODEL.EFNET.BN == 'GN':
        #     self._norm_func = group_norm
        # else:
        #     self._norm_func = FrozenBatchNorm2d
        # self._bn0 = FrozenBatchNorm2d(out_channels)
        # self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._bn0 = self._norm_func(out_channels)
        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, self._norm_func))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, self._norm_func))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._bn1 = self._norm_func(out_channels)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def forward(self, inputs):

        outputs = []

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self._features_idx:
                outputs.append(x)
        return outputs

    # def forward(self, inputs):
    #     """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
    #
    #     # Convolution layers
    #     x = self.extract_features(inputs)
    #
    #     # Pooling and final linear layer
    #     x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
    #     if self._dropout:
    #         x = F.dropout(x, p=self._dropout, training=self.training)
    #     x = self._fc(x)
    #     return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
