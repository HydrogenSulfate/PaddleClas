# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was based on https://github.com/lukemelas/EfficientNet-PyTorch
# reference: https://arxiv.org/abs/1905.11946

import math
import re

import numpy as np
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, Normal, Uniform
from ppcls.utils.config import AttrDict
from ....utils.save_load import (load_dygraph_pretrain,
                                 load_dygraph_pretrain_from_url)

MODEL_URLS = {"EfficientNetV2_S": "TODO"}

__all__ = list(MODEL_URLS.keys())


def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2


inp_shape = {"S": [384, 192, 192, 96, 48, 24, 24, 12], }


class Conv2ds(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride=1,
                 padding=0,
                 groups=None,
                 name=None,
                 act=None,
                 use_bias=None,
                 padding_type=None,
                 model_name=None,
                 cur_stage=None):
        super(Conv2ds, self).__init__()
        assert act in [None, "swish", "sigmoid"]
        self.act = act

        def get_padding(filter_size, stride=1, dilation=1):
            padding = ((stride - 1) + dilation * (filter_size - 1)) // 2
            return padding

        inps = inp_shape["S"][cur_stage]
        self.need_crop = False
        if padding_type == "SAME":
            top_padding, bottom_padding = cal_padding(inps, stride,
                                                      filter_size)
            left_padding, right_padding = cal_padding(inps, stride,
                                                      filter_size)
            height_padding = bottom_padding
            width_padding = right_padding
            if top_padding != bottom_padding or left_padding != right_padding:
                height_padding = top_padding + stride
                width_padding = left_padding + stride
                self.need_crop = True
            padding = [height_padding, width_padding]
        elif padding_type == "VALID":
            height_padding = 0
            width_padding = 0
            padding = [height_padding, width_padding]
        elif padding_type == "DYNAMIC":
            padding = get_padding(filter_size, stride)
        else:
            padding = padding_type

        groups = 1 if groups is None else groups
        self._conv = nn.Conv2D(
            input_channels,
            output_channels,
            filter_size,
            groups=groups,
            stride=stride,
            padding=padding,
            weight_attr=None,
            bias_attr=use_bias
            if not use_bias else ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        x = self._conv(inputs)
        if self.act == "swish":
            x = F.swish(x)
        elif self.act == "sigmoid":
            x = F.sigmoid(x)

        if self.need_crop:
            x = x[:, :, 1:, 1:]
        return x


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = AttrDict()
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        t = AttrDict(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=int(options['s']),
            conv_type=int(options['c']) if 'c' in options else 0, )
        return t

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d' % block.strides,
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % block.conv_type,
            'f%d' % block.fused_conv,
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.

        Args:
        string_list: a list of strings, each string is a notation of block.

        Returns:
        A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.

        Args:
        blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
        a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


#################### EfficientNet V2 configs ####################
v2_base_block = [  # The baseline config for v2 models.
    'r1_k3_s1_e1_i32_o16_c1',
    'r2_k3_s2_e4_i16_o32_c1',
    'r2_k3_s2_e4_i32_o48_c1',
    'r3_k3_s2_e4_i48_o96_se0.25',
    'r5_k3_s1_e6_i96_o112_se0.25',
    'r8_k3_s2_e6_i112_o192_se0.25',
]

v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]

v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]

v2_l_block = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]

v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]
efficientnetv2_params = {
    # (block, width, depth, train_size, eval_size, dropout, randaug, mixup, aug)
    'efficientnetv2-s':  # 83.9% @ 22M
    (v2_s_block, 1.0, 1.0, 300, 384, 0.2, 10, 0, 'randaug'),
    'efficientnetv2-m':  # 85.2% @ 54M
    (v2_m_block, 1.0, 1.0, 384, 480, 0.3, 15, 0.2, 'randaug'),
    'efficientnetv2-l':  # 85.7% @ 120M
    (v2_l_block, 1.0, 1.0, 384, 480, 0.4, 20, 0.5, 'randaug'),
    'efficientnetv2-xl':
    (v2_xl_block, 1.0, 1.0, 384, 512, 0.4, 20, 0.5, 'randaug'),

    # For fair comparison to EfficientNetV1, using the same scaling and autoaug.
    'efficientnetv2-b0':  # 78.7% @ 7M params
    (v2_base_block, 1.0, 1.0, 192, 224, 0.2, 0, 0, 'effnetv1_autoaug'),
    'efficientnetv2-b1':  # 79.8% @ 8M params
    (v2_base_block, 1.0, 1.1, 192, 240, 0.2, 0, 0, 'effnetv1_autoaug'),
    'efficientnetv2-b2':  # 80.5% @ 10M params
    (v2_base_block, 1.1, 1.2, 208, 260, 0.3, 0, 0, 'effnetv1_autoaug'),
    'efficientnetv2-b3':  # 82.1% @ 14M params
    (v2_base_block, 1.2, 1.4, 240, 300, 0.3, 0, 0, 'effnetv1_autoaug'),
}


def efficientnetv2_config(model_name='efficientnetv2-s'):
    """EfficientNetV2 model config."""
    block, width, depth, train_size, eval_size, dropout, randaug, mix, aug = (
        efficientnetv2_params[model_name])

    cfg = AttrDict(
        model=AttrDict(
            model_name=model_name,
            blocks_args=BlockDecoder().decode(block),
            width_coefficient=width,
            depth_coefficient=depth,
            dropout_rate=dropout,
            feature_size=1280,
            bn_type=None,  # 'tpu_bn'(看起来等价于syncbn),
            bn_momentum=0.9,
            bn_epsilon=1e-3,
            gn_groups=8,
            depth_divisor=8,
            min_depth=8,
            act_fn='silu',
            survival_prob=0.8,  # debug修改过，原本是0.8！！
            local_pooling=False,
            headbias=None,
            conv_dropout=None,
            num_classes=1000),
        train=AttrDict(
            isize=train_size, stages=4, sched=True),
        eval=AttrDict(isize=eval_size),
        data=AttrDict(
            augname=aug, ram=randaug, mixup_alpha=mix, cutmix_alpha=mix), )
    return cfg


################################################################################
def get_model_config(model_name: str):
    """Main entry for model name to config."""
    if model_name.startswith('efficientnetv2-'):
        return efficientnetv2_config(model_name)
    raise ValueError(f'Unknown model_name {model_name}')


def round_filters(filters,
                  width_coefficient,
                  depth_divisor,
                  min_depth,
                  skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = width_coefficient
    divisor = depth_divisor
    min_depth = min_depth
    if skip or not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    return int(new_filters)


def round_repeats(repeats, multiplier, skip=False):
    """Round number of filters based on depth multiplier."""
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def activation_fn(act_fn: str):
    """Customized non-linear activation type."""
    if not act_fn:
        return nn.Silu()
    elif act_fn in ("silu", "swish"):
        return nn.Swish()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "relu6":
        return nn.ReLU6()
    elif act_fn == "elu":
        return nn.ELU()
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU()
    elif act_fn == "selu":
        return nn.SELU()
    elif act_fn == "mish":
        return nn.Mish()
    else:
        raise ValueError("Unsupported act_fn {}".format(act_fn))


def drop_path(x, training=False, survival_prob=1.0):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if not training:
        return x
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    keep_prob = paddle.to_tensor(survival_prob)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class SE(nn.Layer):
    """Squeeze-and-excitation layer."""

    def __init__(self, local_pooling, act_fn, in_filters, se_filters,
                 output_filters, cur_stage, padding_type):
        super(SE, self).__init__()

        self._local_pooling = local_pooling
        self._act = activation_fn(act_fn)

        # Squeeze and Excitation layer.
        self._se_reduce = Conv2ds(
            in_filters,
            se_filters,
            1,
            stride=1,
            cur_stage=cur_stage,
            padding_type=padding_type)
        self._se_expand = Conv2ds(
            se_filters,
            output_filters,
            1,
            stride=1,
            cur_stage=cur_stage,
            padding_type=padding_type)

    def forward(self, x):
        if self._local_pooling:
            se_tensor = F.adaptive_avg_pool2d(x, output_size=1)
        else:
            se_tensor = paddle.mean(x, axis=[2, 3], keepdim=True)
        se_tensor = self._se_expand(self._act(self._se_reduce(se_tensor)))
        return F.sigmoid(se_tensor) * x


class MBConvBlock(nn.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

    Attributes:
        endpoints: dict. A list of internal tensors.
    """

    def __init__(self, se_ratio, input_filters, expand_ratio, kernel_size,
                 strides, output_filters, bn_momentum, bn_epsilon,
                 local_pooling, conv_dropout, cur_stage, padding_type):
        """Initializes a MBConv block.

        Args:
            block_args: BlockArgs, arguments to create a Block.
            mconfig: GlobalParams, a set of global parameters.
            name: layer name.
        """
        super(MBConvBlock, self).__init__()

        self.se_ratio = se_ratio
        self.input_filters = input_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_filters = output_filters

        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        self._local_pooling = local_pooling
        self.act_fn = None
        self.conv_dropout = conv_dropout

        self._act = activation_fn(None)
        self._has_se = (self.se_ratio is not None and 0 < self.se_ratio <= 1)
        """Builds block according to the arguments."""
        expand_filters = self.input_filters * self.expand_ratio
        kernel_size = self.kernel_size

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        if self.expand_ratio != 1:
            self._expand_conv = Conv2ds(
                self.input_filters,
                expand_filters,
                1,
                stride=1,
                use_bias=False,
                cur_stage=cur_stage,
                padding_type=padding_type)
            self._norm0 = nn.BatchNorm2D(
                expand_filters,
                self.bn_momentum,
                self.bn_epsilon,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = Conv2ds(
            expand_filters,
            expand_filters,
            kernel_size,
            padding=kernel_size // 2,
            stride=self.strides,
            groups=expand_filters,
            use_bias=False,
            cur_stage=cur_stage,
            padding_type=padding_type)

        self._norm1 = nn.BatchNorm2D(
            expand_filters,
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        if self._has_se:
            num_reduced_filters = max(1,
                                      int(self.input_filters * self.se_ratio))
            self._se = SE(self._local_pooling, None, expand_filters,
                          num_reduced_filters, expand_filters, cur_stage,
                          padding_type)
        else:
            self._se = None

        # Output phase.
        self._project_conv = Conv2ds(
            expand_filters,
            self.output_filters,
            1,
            stride=1,
            use_bias=False,
            cur_stage=cur_stage,
            padding_type=padding_type)
        self._norm2 = nn.BatchNorm2D(
            self.output_filters,
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.drop_out = nn.Dropout(self.conv_dropout)

    def residual(self, inputs, x, survival_prob):
        if (self.strides == 1 and self.input_filters == self.output_filters):
            # Apply only if skip connection presents.
            if survival_prob:
                x = drop_path(x, self.training, survival_prob)
            x = paddle.add(x, inputs)

        return x

    def forward(self, inputs, survival_prob=None):
        """Implementation of call().

        Args:
            inputs: the inputs tensor.
            survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
            A output tensor.
        """
        x = inputs
        if self.expand_ratio != 1:
            x = self._act(self._norm0(self._expand_conv(x)))

        x = self._act(self._norm1(self._depthwise_conv(x)))

        if self.conv_dropout and self.expand_ratio > 1:
            x = self.drop_out(x)

        if self._se:
            x = self._se(x)

        x = self._norm2(self._project_conv(x))
        x = self.residual(inputs, x, survival_prob)

        return x


class FusedMBConvBlock(MBConvBlock):
    """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""

    def __init__(self, se_ratio, input_filters, expand_ratio, kernel_size,
                 strides, output_filters, bn_momentum, bn_epsilon,
                 local_pooling, conv_dropout, cur_stage, padding_type):
        """Builds block according to the arguments."""
        super(MBConvBlock, self).__init__()
        self.se_ratio = se_ratio
        self.input_filters = input_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_filters = output_filters

        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        self._local_pooling = local_pooling
        self.act_fn = None
        self.conv_dropout = conv_dropout

        self._act = activation_fn(None)
        self._has_se = (self.se_ratio is not None and 0 < self.se_ratio <= 1)

        expand_filters = self.input_filters * self.expand_ratio
        kernel_size = self.kernel_size
        if self.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = Conv2ds(
                self.input_filters,
                expand_filters,
                kernel_size,
                padding=kernel_size // 2,
                stride=self.strides,
                use_bias=False,
                cur_stage=cur_stage,
                padding_type=padding_type)
            self._norm0 = nn.BatchNorm2D(
                expand_filters,
                self.bn_momentum,
                self.bn_epsilon,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        if self._has_se:
            num_reduced_filters = max(1,
                                      int(self.input_filters * self.se_ratio))
            self._se = SE(self._local_pooling, None, expand_filters,
                          num_reduced_filters, expand_filters, cur_stage,
                          padding_type)
        else:
            self._se = None

        # Output phase:
        self._project_conv = Conv2ds(
            expand_filters,
            self.output_filters,
            1 if (self.expand_ratio != 1) else kernel_size,
            padding=(1 if (self.expand_ratio != 1) else kernel_size) // 2,
            stride=1 if (self.expand_ratio != 1) else self.strides,
            use_bias=False,
            cur_stage=cur_stage,
            padding_type=padding_type)
        self._norm1 = nn.BatchNorm2D(
            self.output_filters,
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.drop_out = nn.Dropout(conv_dropout)

    def forward(self, inputs, survival_prob=None):
        """Implementation of call().

        Args:
            inputs: the inputs tensor.
            training: boolean, whether the model is constructed for training.
            survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
            A output tensor.
        """
        x = inputs
        if self.expand_ratio != 1:
            x = self._act(self._norm0(self._expand_conv(x)))

        if self.conv_dropout and self.expand_ratio > 1:
            x = self.drop_out(x)

        if self._se:
            x = self._se(x)

        x = self._norm1(self._project_conv(x))
        if self.expand_ratio == 1:
            x = self._act(x)  # add act if no expansion.

        x = self.residual(inputs, x, survival_prob)
        return x


class Stem(nn.Layer):
    """Stem layer at the begining of the network."""

    def __init__(self, width_coefficient, depth_divisor, min_depth, skip,
                 bn_momentum, bn_epsilon, act_fn, stem_filters, cur_stage,
                 padding_type):
        super(Stem, self).__init__()
        self._conv_stem = Conv2ds(
            3,
            round_filters(stem_filters, width_coefficient, depth_divisor,
                          min_depth, skip),
            3,
            padding=1,
            stride=2,
            use_bias=False,
            cur_stage=cur_stage,
            padding_type=padding_type)
        self._norm = nn.BatchNorm2D(
            round_filters(stem_filters, width_coefficient, depth_divisor,
                          min_depth, skip),
            bn_momentum,
            bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._act = activation_fn(act_fn)

    def forward(self, inputs):
        return self._act(self._norm(self._conv_stem(inputs)))


class Head(nn.Layer):
    """Head layer for network outputs."""

    def __init__(self,
                 in_filters,
                 feature_size,
                 bn_momentum,
                 bn_epsilon,
                 act_fn,
                 dropout_rate,
                 local_pooling,
                 width_coefficient,
                 depth_divisor,
                 min_depth,
                 skip=False):
        super(Head, self).__init__()
        self.in_filters = in_filters
        self.feature_size = feature_size
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.dropout_rate = dropout_rate
        self.local_pooling = local_pooling
        self._conv_head = nn.Conv2D(
            in_filters,
            round_filters(self.feature_size or 1280, width_coefficient,
                          depth_divisor, min_depth, skip),
            kernel_size=1,
            stride=1,
            bias_attr=False)
        self._norm = nn.BatchNorm2D(
            round_filters(self.feature_size or 1280, width_coefficient,
                          depth_divisor, min_depth, skip),
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._act = activation_fn(act_fn)

        self._avg_pooling = nn.AdaptiveAvgPool2D(output_size=1)

        if self.dropout_rate > 0:
            self._dropout = nn.Dropout(self.dropout_rate)
        else:
            self._dropout = None

    def forward(self, x):
        """Call the layer."""
        outputs = self._act(self._norm(self._conv_head(x)))

        if self.local_pooling:
            outputs = F.adaptive_avg_pool2d(outputs, output_size=1)
            if self._dropout:
                outputs = self._dropout(outputs)
            if self._fc:
                outputs = paddle.squeeze(outputs, axis=[2, 3])
                outputs = self._fc(outputs)
        else:
            outputs = self._avg_pooling(outputs)
            if self._dropout:
                outputs = self._dropout(outputs)
        return paddle.flatten(outputs, start_axis=1)


class EfficientNetV2(nn.Layer):
    """A class implements tf.keras.Model.

        Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self,
                 model_name="efficientnetv2-s",
                 blocks_args=None,
                 mconfig=None,
                 include_top=True,
                 class_num=1000,
                 padding_type="SAME"):
        """Initializes an `Model` instance.

        Args:
            model_name: A string of model name.
            model_config: A dict of model configurations or a string of hparams.
        Raises:
            ValueError: when blocks_args is not specified as a list.
        """
        super(EfficientNetV2, self).__init__()
        self.blocks_args = blocks_args
        self.mconfig = mconfig
        """Builds a model."""
        self._blocks = nn.LayerList()

        cur_stage = 0
        # Stem part.
        self._stem = Stem(
            self.mconfig.width_coefficient,
            self.mconfig.depth_divisor,
            self.mconfig.min_depth,
            False,
            self.mconfig.bn_momentum,
            self.mconfig.bn_epsilon,
            self.mconfig.act_fn,
            stem_filters=self.blocks_args[0].input_filters,
            cur_stage=cur_stage,
            padding_type=padding_type)
        cur_stage += 1

        # Builds blocks.
        for block_args in self.blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(
                block_args.input_filters, self.mconfig.width_coefficient,
                self.mconfig.depth_divisor, self.mconfig.min_depth, False)
            output_filters = round_filters(
                block_args.output_filters, self.mconfig.width_coefficient,
                self.mconfig.depth_divisor, self.mconfig.min_depth, False)

            repeats = round_repeats(block_args.num_repeat,
                                    self.mconfig.depth_coefficient)
            block_args.update(
                dict(
                    input_filters=input_filters,
                    output_filters=output_filters,
                    num_repeat=repeats))

            # The first block needs to take care of stride and filter size increase.
            conv_block = {
                0: MBConvBlock,
                1: FusedMBConvBlock
            }[block_args.conv_type]
            self._blocks.append(
                conv_block(block_args.se_ratio, block_args.input_filters,
                           block_args.expand_ratio, block_args.kernel_size,
                           block_args.strides, block_args.output_filters,
                           self.mconfig.bn_momentum, self.mconfig.bn_epsilon,
                           self.mconfig.local_pooling,
                           self.mconfig.conv_dropout, cur_stage, padding_type))
            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                block_args.input_filters = block_args.output_filters
                block_args.strides = 1
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    conv_block(block_args.se_ratio, block_args.input_filters,
                               block_args.expand_ratio, block_args.kernel_size,
                               block_args.strides, block_args.output_filters,
                               self.mconfig.bn_momentum, self.mconfig.
                               bn_epsilon, self.mconfig.local_pooling, self.
                               mconfig.conv_dropout, cur_stage, padding_type))
            cur_stage += 1

        # Head part.
        self._head = Head(
            self.blocks_args[-1].output_filters, self.mconfig.feature_size,
            self.mconfig.bn_momentum, self.mconfig.bn_epsilon,
            self.mconfig.act_fn, self.mconfig.dropout_rate,
            self.mconfig.local_pooling, self.mconfig.width_coefficient,
            self.mconfig.depth_divisor, self.mconfig.min_depth, False)

        # top part for classification
        if include_top and class_num:
            self._fc = nn.Linear(
                self.mconfig.feature_size,
                class_num,
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        else:
            self._fc = None

        def _init_weights(m):
            if isinstance(m, nn.Conv2D):
                out_filters, in_filters, kernel_height, kernel_width = m.weight.shape
                if in_filters == 1 and out_filters > in_filters:
                    out_filters = in_filters
                fan_out = int(kernel_height * kernel_width * out_filters)
                Normal(mean=0.0, std=np.sqrt(2.0 / fan_out))(m.weight)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / np.sqrt(m.weight.shape[1])
                Uniform(-init_range, init_range)(m.weight)
                Constant(0.0)(m.bias)

        self.apply(_init_weights)

    def forward(self, inputs):
        """Implementation of call().

        Args:
            inputs: input tensors.
            training: boolean, whether the model is constructed for training.
            with_endpoints: If true, return a list of endpoints.

        Returns:
            output tensors.
        """

        # Calls Stem layers
        outputs = self._stem(inputs)
        # print(f"stem: {outputs.mean().item():.10f}")

        # Calls blocks.
        for idx, block in enumerate(self._blocks):
            survival_prob = self.mconfig.survival_prob
            if survival_prob:
                drop_rate = 1.0 - survival_prob
                survival_prob = 1.0 - drop_rate * float(idx) / len(
                    self._blocks)
            outputs = block(outputs, survival_prob=survival_prob)

        # Head to obtain the final feature.
        outputs = self._head(outputs)
        # Calls final dense layers and returns logits.
        if self._fc:
            outputs = self._fc(outputs)

        return outputs


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def EfficientNetV2_S(model_config=None,
                     include_top=True,
                     pretrained=False,
                     use_ssld=False,
                     **kwargs):
    """Get a EfficientNet V1 or V2 model instance.

    This is a simply utility for finetuning or inference.

    Args:
        model_config: A dict of model configurations or a string of hparams.
        include_top: whether to include the final dense layer for classification.

    Returns:
        A single tensor if with_endpoints if False; otherwise, a list of tensor.
    """
    model_config = efficientnetv2_config("efficientnetv2-s")
    model = EfficientNetV2("efficientnetv2-s", model_config.model.blocks_args,
                           model_config.model, include_top)
    _load_pretrained(
        pretrained, model, MODEL_URLS["EfficientNetV2_S"], use_ssld=use_ssld)
    return model
