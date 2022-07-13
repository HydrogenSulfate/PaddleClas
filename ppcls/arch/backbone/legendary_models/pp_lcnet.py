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

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm2D, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.arch.backbone.base.theseus_layer import RCCAModule, GeneralizedMeanPooling, GeneralizedMeanPoolingP, RGAModule
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from ppcls.utils import logger

MODEL_URLS = {
    "PPLCNet_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    "PPLCNet_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
    "PPLCNet_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
    "PPLCNet_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    "PPLCNet_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
    "PPLCNet_x1_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
    "PPLCNet_x2_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
    "PPLCNet_x2_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams"
}

MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

__all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1,
                 lr_mult=1.0,
                 use_ibn_b=False):
        super().__init__()

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal(), learning_rate=lr_mult),
            bias_attr=False)

        self.bn = BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult))
        self.use_ibn_b = use_ibn_b
        if use_ibn_b:
            self.InstanceNorm = nn.InstanceNorm2D(num_filters)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_ibn_b:
            x = self.InstanceNorm(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False,
                 lr_mult=1.0,
                 use_ibn_b=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
            lr_mult=lr_mult)
        if use_se:
            self.se = SEModule(num_channels,
                               lr_mult=lr_mult)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            lr_mult=lr_mult,
            use_ibn_b=use_ibn_b)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(TheseusLayer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class PPLCNet(TheseusLayer):
    def __init__(self,
                 stages_pattern,
                 scale=1.0,
                 class_num=1000,
                 dropout_prob=0.2,
                 class_expand=1280,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 use_last_conv=True,
                 return_patterns=None,
                 return_stages=None,
                 # 下方为新增的模型调优相关参数
                 last_stride=2,     # last_stride
                 use_cc=False,      # Criss-Cross Attention
                 cc_after=6,        # Insert Criss-Cross Attention after `cc_after` blocks
                 use_gem=False,     # GeneralizedMeanPooling
                 use_gemp=False,    # GeneralizedMeanPoolingP
                 pnorm=1.1,
                 use_rga=False,     # Relation-Aware Global Attention
                 use_ibn_b=False,   # IBN-Net type B
                 ibn_act_list=[3, 4, 5, 6]  # IBN-Net type B
    ):
        super().__init__()

        # last stride
        if last_stride == 1:
            assert NET_CONFIG["blocks6"][-2][-2] != 1, \
                "last stride已经是1无需设置"
            NET_CONFIG["blocks6"][-2][-2] = 1
            logger.info(f"{'=' * 10} Using last_stride=1(set block6.conv1.stride to 1)")

        # Criss-Cross Attention
        self.use_cc = use_cc
        if use_cc:
            self.cc_after = cc_after
            logger.info(f"{'=' * 10} Using Criss-Cross Attention")
            self.cc_att = RCCAModule(
                in_channels=make_divisible(NET_CONFIG[f"blocks{self.cc_after}"][-1][2] * scale),
                inter_channels=make_divisible(NET_CONFIG[f"blocks{self.cc_after}"][-1][2] * scale)//4,
                reduction=8,
                recurrence=2
            )

        # Relation-Aware Global Attention
        self.use_rga = use_rga
        if use_rga:
            logger.info(f"{'=' * 10} Using Relation-Aware Global Attention")
            h, w = 256, 128
            down_fac = {
                3: 4,
                4: 8,
                5: 16,
                6: 16
            }
            self.rga_modules = nn.LayerList([
                RGAModule(
                    in_channel=make_divisible(NET_CONFIG[f"blocks{_i}"][0][2] * scale),
                    in_spatial=(h//down_fac[_i])*(w//down_fac[_i]),
                    use_spatial=True,
                    use_channel=True,
                    cha_ratio=8,
                    spa_ratio=8,
                    down_ratio=8
                )
                for _i in [3, 4, 5, 6]
            ])
        self.use_ibn_b = use_ibn_b
        if use_ibn_b:
            logger.info(f"{'=' * 10} Using IBN-B, and ibn_act_list = {ibn_act_list}")
        self.scale = scale
        self.class_expand = class_expand
        self.lr_mult_list = lr_mult_list
        self.use_last_conv = use_last_conv
        if isinstance(self.lr_mult_list, str):
            self.lr_mult_list = eval(self.lr_mult_list)

        assert isinstance(self.lr_mult_list, (
            list, tuple
        )), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list))
        assert len(self.lr_mult_list
                   ) == 6, "lr_mult_list length should be 5 but got {}".format(
                       len(self.lr_mult_list))

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2,
            lr_mult=self.lr_mult_list[0])

        self.blocks2 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[1],
                use_ibn_b=(use_ibn_b is True) and (2 in ibn_act_list))
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[2],
                use_ibn_b=(use_ibn_b is True) and (3 in ibn_act_list))
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[3],
                use_ibn_b=(use_ibn_b is True) and (4 in ibn_act_list))
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[4],
                use_ibn_b=(use_ibn_b is True) and (5 in ibn_act_list))
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[5])
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.avg_pool = AdaptiveAvgPool2D(1)
        # GeneralizedMeanPooling
        self.use_gem = use_gem
        if use_gem:
            logger.info(f"{'=' * 10} Using GeneralizedMeanPooling")
            self.avg_pool = GeneralizedMeanPooling(norm=3)

        # GeneralizedMeanPoolingP
        self.use_gemp = use_gemp
        if use_gemp:
            logger.info(f"{'=' * 10} Using GeneralizedMeanPoolingP(norm={pnorm})")
            self.avg_pool = GeneralizedMeanPoolingP(norm=pnorm)
        if self.use_last_conv:
            self.last_conv = Conv2D(
                in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.hardswish = nn.Hardswish()
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = Linear(self.class_expand if self.use_last_conv else NET_CONFIG["blocks6"][-1][2], class_num)

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):
        x = self.conv1(x)  # 256x128

        x = self.blocks2(x)  # 128x64
        if self.use_cc and self.cc_after == 2:
            x = self.cc_att(x)

        x = self.blocks3(x)  # 64x32
        if self.use_cc and self.cc_after == 3:
            x = self.cc_att(x)
        if self.use_rga:
            x = self.rga_modules[0](x)

        x = self.blocks4(x)  # 32x16
        if self.use_cc and self.cc_after == 4:
            x = self.cc_att(x)
        if self.use_rga:
            x = self.rga_modules[1](x)

        x = self.blocks5(x)  # 16x8
        if self.use_cc and self.cc_after == 5:
            x = self.cc_att(x)
        if self.use_rga:
            x = self.rga_modules[2](x)

        x = self.blocks6(x)  # 16x8
        if self.use_cc and self.cc_after == 6:
            x = self.cc_att(x)
        if self.use_rga:
            x = self.rga_modules[3](x)

        x = self.avg_pool(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
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


def PPLCNet_x0_25(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_25` model depends on args.
    """
    model = PPLCNet(
        scale=0.25, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_25"], use_ssld)
    return model


def PPLCNet_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_35` model depends on args.
    """
    model = PPLCNet(
        scale=0.35, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_35"], use_ssld)
    return model


def PPLCNet_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_5` model depends on args.
    """
    model = PPLCNet(
        scale=0.5, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_5"], use_ssld)
    return model


def PPLCNet_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_75` model depends on args.
    """
    model = PPLCNet(
        scale=0.75, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_75"], use_ssld)
    return model


def PPLCNet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_0` model depends on args.
    """
    model = PPLCNet(
        scale=1.0, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x1_0"], use_ssld)
    return model


def PPLCNet_x1_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x1_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_5` model depends on args.
    """
    model = PPLCNet(
        scale=1.5, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x1_5"], use_ssld)
    return model


def PPLCNet_x2_0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x2_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x2_0` model depends on args.
    """
    model = PPLCNet(
        scale=2.0, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x2_0"], use_ssld)
    return model


def PPLCNet_x2_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x2_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x2_5` model depends on args.
    """
    model = PPLCNet(
        scale=2.5, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x2_5"], use_ssld)
    return model
