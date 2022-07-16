# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm2D, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal, Constant, Normal, Uniform, KaimingUniform

from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from ppcls.utils import logger

MODEL_URLS = {
    "PPLCNetV2_base":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNetV2_base_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

NET_CONFIG = {
    # in_channels, kernel_size, split_pw, use_rep, use_se, use_shortcut
    "stage1": [64, 3, False, False, False, False],
    "stage2": [128, 3, False, False, False, False],
    "stage3": [256, 5, True, True, True, False],
    "stage4": [512, 5, False, True, False, True],
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
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        if self.use_act:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class SEModule(TheseusLayer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class RepDepthwiseSeparable(TheseusLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dw_size=3,
                 split_pw=False,
                 use_rep=False,
                 use_se=False,
                 use_shortcut=False):
        super().__init__()
        self.is_repped = False
        # print(f"stride={stride}")
        self.dw_size = dw_size
        self.split_pw = split_pw
        self.use_rep = use_rep
        self.use_se = use_se
        self.use_shortcut = True if use_shortcut and stride == 1 and in_channels == out_channels else False

        if self.use_rep:
            self.dw_conv_list = nn.LayerList()
            for kernel_size in range(self.dw_size, 0, -2):
                if kernel_size == 1 and stride != 1:
                    continue
                dw_conv = ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channels,
                    use_act=False)
                self.dw_conv_list.append(dw_conv)
            self.dw_conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=dw_size,
                stride=stride,
                padding=(dw_size - 1) // 2,
                groups=in_channels)
        else:
            self.dw_conv = ConvBNLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=dw_size,
                stride=stride,
                groups=in_channels)

        self.act = nn.ReLU()

        if use_se:
            self.se = SEModule(in_channels)

        if self.split_pw:
            pw_ratio = 0.5
            self.pw_conv_1 = ConvBNLayer(
                in_channels=in_channels,
                kernel_size=1,
                out_channels=int(out_channels * pw_ratio),
                stride=1)
            self.pw_conv_2 = ConvBNLayer(
                in_channels=int(out_channels * pw_ratio),
                kernel_size=1,
                out_channels=out_channels,
                stride=1)
        else:
            self.pw_conv = ConvBNLayer(
                in_channels=in_channels,
                kernel_size=1,
                out_channels=out_channels,
                stride=1)

    def forward(self, x):
        if self.use_rep:
            input_x = x
            if self.is_repped:
                x = self.act(self.dw_conv(x))
            else:
                y = self.dw_conv_list[0](x)
                for dw_conv in self.dw_conv_list[1:]:
                    y += dw_conv(x)
                x = self.act(y)
        else:
            x = self.dw_conv(x)

        if self.use_se:
            x = self.se(x)
        if self.split_pw:
            x = self.pw_conv_1(x)
            x = self.pw_conv_2(x)
        else:
            x = self.pw_conv(x)
        if self.use_shortcut:
            x = x + input_x
        return x

    def rep(self):
        if self.use_rep:
            self.is_repped = True
            kernel, bias = self._get_equivalent_kernel_bias()
            self.dw_conv.weight.set_value(kernel)
            self.dw_conv.bias.set_value(bias)

    def _get_equivalent_kernel_bias(self):
        kernel_sum = 0
        bias_sum = 0
        for dw_conv in self.dw_conv_list:
            kernel, bias = self._fuse_bn_tensor(dw_conv)
            kernel = self._pad_tensor(kernel, to_size=self.dw_size)
            kernel_sum += kernel
            bias_sum += bias
        return kernel_sum, bias_sum

    def _fuse_bn_tensor(self, branch):
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def _pad_tensor(self, tensor, to_size):
        from_size = tensor.shape[-1]
        if from_size == to_size:
            return tensor
        pad = (to_size - from_size) // 2
        return F.pad(tensor, [pad, pad, pad, pad])


# DOLG related code below
def _calculate_fan_in_and_fan_out(tensor: paddle.Tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        num_input_fmaps = tensor.shape[0]
        num_output_fmaps = tensor.shape[1]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def init_weights(m: nn.Layer, zero_init_gamma: bool = True) -> None:
    if isinstance(m, nn.Conv2D):
        # Note that there is no bias due to BN
        fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
        # m.weight.data.normal_()
        Normal(mean=0.0, std=math.sqrt(2.0 / fan_out))(m.weight)
    elif isinstance(m, nn.BatchNorm2D):
        # zero_init_gamma = hasattr(m, "final_bn") and m.final_bn and zero_init_gamma
        # m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        # m.bias.data.zero_()
        Constant(0.0 if zero_init_gamma else 1.0)(m.weight)
        Constant(0.0)(m.weight)
    elif isinstance(m, nn.Linear):
        Normal(mean=0.0, std=0.01)(m.weight)
        Constant(0.0)(m.bias)


class SpatialAttention2d(nn.Layer):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 act_fn: str = "relu",
                 with_aspp: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 torch_style_init: bool = True):
        super(SpatialAttention2d, self).__init__()

        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(in_c)

        self.conv1 = nn.Conv2D(in_c, out_c, 1, 1)
        self.bn = nn.BatchNorm2D(out_c, epsilon=eps, momentum=momentum)

        if act_fn.lower() in ["relu"]:
            self.act1 = nn.ReLU()

        elif act_fn.lower() in ["leakyrelu", "leaky", "leaky_relu"]:
            self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2D(out_c, out_channels=1, kernel_size=1, stride=1)
        self.softplus = nn.Softplus(beta=1, threshold=20)  # use default setting.

        if torch_style_init:
            for conv in [self.conv1, self.conv2]:
                if torch_style_init:
                    conv.apply(init_weights)
                    if conv.bias is not None:
                        fan_in, _ = _calculate_fan_in_and_fan_out(conv.weight)
                        bound = 1 / math.sqrt(fan_in)
                        Uniform(-bound, bound)(conv.bias)

    def forward(self, x):
        """
        x : spatial feature map. (b,c,h,w)
        att : softplus attention score
        """
        if self.with_aspp:
            x = self.aspp(x)  # 512
        x = self.conv1(x)  # 512->512
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, axis=1)

        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        att = paddle.expand_as(att_score, feature_map_norm)
        x = att * feature_map_norm
        return x, att_score


class ASPP(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling Module
    """
    def __init__(self, in_c: int, torch_style_init: bool = True):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2D(in_c, 256, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2D(in_c, 256, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.LayerList(self.aspp)

        self.im_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_c, 256, 1, 1),
            nn.ReLU()
        )
        conv_after_dim = 256 * (len(self.aspp) + 1)
        self.conv_after = nn.Sequential(
            nn.Conv2D(conv_after_dim, 512, 1, 1),
            nn.ReLU()
        )

        if torch_style_init:
            for dilation_conv in self.aspp:
                dilation_conv.apply(init_weights)
            for model in self.im_pool:
                if isinstance(model, nn.Conv2D):
                    model.apply(init_weights)
            for model in self.conv_after:
                if isinstance(model, nn.Conv2D):
                    model.apply(init_weights)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h, w), mode="bilinear", align_corners=False)]  # 256
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))  # 256x3
        aspp_out = paddle.concat(aspp_out, 1)  # 256*(3+1)
        x = self.conv_after(aspp_out)  # 512
        return x


class PPLCNetV2(TheseusLayer):
    def __init__(self,
                 scale,
                 depths,
                 last_stride=2,
                 class_num=1000,
                 dropout_prob=0,
                 use_last_conv=True,
                 class_expand=1280,
                 use_dolg=False,
                 with_aspp=False,
                 torch_style_init=False,
                 expand_dropout=True,
                 expand_relu=True,
                 fc_dropout=False,
                 fc_relu=False):
        super().__init__()
        self.scale = scale
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.last_stride = last_stride

        self.stem = nn.Sequential(* [
            ConvBNLayer(
                in_channels=3,
                kernel_size=3,
                out_channels=make_divisible(32 * scale),
                stride=2), RepDepthwiseSeparable(
                    in_channels=make_divisible(32 * scale),
                    out_channels=make_divisible(64 * scale),
                    stride=1,
                    dw_size=3)
        ])

        # stages
        self.stages = nn.LayerList()
        for depth_idx, k in enumerate(NET_CONFIG):
            in_channels, kernel_size, split_pw, use_rep, use_se, use_shortcut = NET_CONFIG[
                k]
            self.stages.append(
                nn.Sequential(* [
                    RepDepthwiseSeparable(
                        in_channels=make_divisible((in_channels if i == 0 else
                                                    in_channels * 2) * scale),
                        out_channels=make_divisible(in_channels * 2 * scale),
                        stride=2 if i == 0 and depth_idx != len(NET_CONFIG) - 1 else self.last_stride if i == 0 and depth_idx == len(NET_CONFIG) - 1 else 1,
                        dw_size=kernel_size,
                        split_pw=split_pw,
                        use_rep=use_rep,
                        use_se=use_se,
                        use_shortcut=use_shortcut)
                    for i in range(depths[depth_idx])
                ]))
        # last stride
        if last_stride == 1:
            logger.info(f"{'=' * 10} Using last_stride=1(set pplcnet.laststride to 1)")

        # dolg
        self.use_dolg = use_dolg
        if use_dolg:
            logger.info(f"{'=' * 10} Using DOLG to fuse stage3 and stage4's output, with_aspp={with_aspp}")
            self.pool_global = AdaptiveAvgPool2D(1)
            self.stage3_channels = make_divisible(NET_CONFIG["stage3"][0] * 2 * scale)
            self.stage4_channels = make_divisible(NET_CONFIG["stage4"][0] * 2 * scale)
            assert self.stage3_channels == 512 and self.stage4_channels == 1024, \
                f"assert self.stage3_channels == {self.stage3_channels} and self.stage4_channels == {self.stage4_channels}"

            self.fc_t = nn.Linear(self.stage4_channels, self.stage3_channels, bias_attr=True)

            if torch_style_init:
                KaimingUniform(fan_in=6*self.stage4_channels)(self.fc_t.weight)
                if self.fc_t.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(self.fc_t.weight)
                    bound = 1.0 / math.sqrt(fan_in)
                    Uniform(-bound, bound)(self.fc_t.bias)

            self.localmodel = SpatialAttention2d(self.stage3_channels, self.stage3_channels, "relu", with_aspp)
            self.pool_local = nn.AdaptiveAvgPool2D((1, 1))

        self.avg_pool = AdaptiveAvgPool2D(1)

        if self.use_last_conv:
            self.last_conv = Conv2D(
                in_channels=make_divisible(NET_CONFIG["stage4"][0] * 2 *
                                           scale),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.act = nn.ReLU() if expand_relu else nn.Identity()
            if not expand_relu:
                logger.info(f"{'=' * 10} expand_relu={expand_relu}")
            if not expand_dropout:
                logger.info(f"{'=' * 10} expand_dropout={expand_dropout}")
            if expand_dropout and dropout_prob > 0:
                self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
            else:
                self.dropout = nn.Identity()
                logger.info(f"{'=' * 10} use_last_conv without Dropout(dropout_prob={dropout_prob:.1f})")
        else:
            logger.info(f"{'=' * 10} use_last_conv=False")

        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        in_features = self.class_expand if self.use_last_conv else make_divisible(NET_CONFIG[
            "stage4"][0] * 2 * scale)
        self.fc = Linear(in_features, class_num)

        assert (self.use_last_conv and (not fc_relu) and (not fc_dropout)) or \
            (not self.use_last_conv)
        self.fc_relu = fc_relu
        self.fc_dropout = fc_dropout
        if fc_relu:
            self.act = nn.ReLU()
            logger.info(f"{'=' * 10} use ReLU after self.fc")
        if fc_dropout:
            if dropout_prob > 0:
                self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
                logger.info(f"{'=' * 10} use DropOut({dropout_prob}) after self.fc")
            else:
                self.dropout = nn.Identity()
                logger.info(f"{'=' * 10} use self.fc without Dropout(dropout_prob={dropout_prob:.1f})")

    def _globalmodel_forward(self, x):
        """
        [B, 64, 112, 112] <- stem
        [B, 128, 56, 56] <- stage1
        [B, 256, 28, 28] <- stage2
        [B, 512, 14, 14] <- stage3
        [B, 1024, 14, 14] <- stage4
        """
        x = self.stem(x)
        x = self.stages[0](x)
        x = self.stages[1](x)

        # get stage3 and stage4' output
        f3 = self.stages[2](x)
        f4 = self.stages[3](f3)
        return f3, f4

    def forward_dolg(self, x):
        f3, f4 = self._globalmodel_forward(x)
        fl, _ = self.localmodel(f3)

        fg_o = self.pool_global(f4)
        fg_o = fg_o.reshape([fg_o.shape[0], self.stage4_channels])

        fg = self.fc_t(fg_o)
        fg_norm = paddle.norm(fg, p=2, axis=1)

        proj = paddle.bmm(fg.unsqueeze(1), paddle.flatten(fl, start_axis=2))
        proj = paddle.bmm(fg.unsqueeze(2), proj).reshape(fl.shape)
        proj = proj / (fg_norm * fg_norm).reshape([-1, 1, 1, 1])
        orth_comp = fl - proj

        fo = self.pool_local(orth_comp)
        fo = fo.reshape([fo.shape[0], self.stage3_channels])

        final_feat = paddle.concat([fg, fo], axis=1)
        final_feat = final_feat.unsqueeze(-1).unsqueeze(-1)
        return final_feat

    def forward(self, x):
        if self.use_dolg:
            x = self.forward_dolg(x)
        else:
            x = self.stem(x)
            for stage in self.stages:
                x = stage(x)
            x = self.avg_pool(x)
        if self.use_last_conv:
            x = self.last_conv(x)  # 1024->in_fea
            x = self.act(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)  # in_fea->out_fea
        if self.fc_relu:
            x = self.act(x)
        if self.fc_dropout:
            x = self.dropout(x)
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


def PPLCNetV2_base(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNetV2_base
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNetV2_base` model depends on args.
    """
    if "dropout_prob" in kwargs:
        dropout_prob = kwargs.pop("dropout_prob")
        assert isinstance(dropout_prob, float), \
            f"dropout_prob({type(dropout_prob)} must be float"
    else:
        dropout_prob = 0.2
    model = PPLCNetV2(
        scale=1.0, depths=[2, 2, 6, 2], dropout_prob=dropout_prob, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNetV2_base"], use_ssld)
    return model
