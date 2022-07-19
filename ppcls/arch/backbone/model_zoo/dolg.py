import math
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import (Assign, Constant, KaimingUniform, Normal,
                                   Uniform)
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.arch.backbone.legendary_models import ResNet50
from ppcls.utils.save_load import (load_dygraph_pretrain,
                                   load_dygraph_pretrain_from_url)


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        if 'http' in pretrained:
            load_dygraph_pretrain_from_url(model, pretrained, use_ssld=False)
        else:
            load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


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


class GeneralizedMeanPooling(nn.Layer):
    """GeneralizedMeanPooling

    Args:
        norm (Optional[int, float]): norm factor p
        output_size (int, optional): output size. Defaults to 1.
        eps (float, optional): epsilon. Defaults to 1e-6.
    """
    def __init__(self, norm: Tuple[int, float], output_size: int = 1, eps: float = 1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clip(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.p)


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """GeneralizedMeanPooling with trainable parameter P

    Args:
        norm (Optional[int, float]): initial norm factor p. Defaults to 3.
        output_size (int, optional): output size. Defaults to 1.
        eps (float, optional): epsilon. Defaults to 1e-6.
    """
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = self.create_parameter(
            shape=(1, ),
            default_initializer=Assign(paddle.full([1, ], norm))
        )
        self.add_parameter("p", self.p)


class SpatialAttention2d(nn.Layer):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 act_fn: str = "relu",
                 with_aspp: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.9):
        super(SpatialAttention2d, self).__init__()

        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(out_c)

        self.conv1 = nn.Conv2D(in_c, out_c, 1, 1)
        self.bn = nn.BatchNorm2D(out_c, epsilon=eps, momentum=momentum)

        if act_fn.lower() in ["relu"]:
            self.act1 = nn.ReLU()

        elif act_fn.lower() in ["leakyrelu", "leaky", "leaky_relu"]:
            self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2D(out_c, out_channels=1, kernel_size=1, stride=1)
        self.softplus = nn.Softplus(beta=1, threshold=20)  # use default setting.

        for conv in [self.conv1, self.conv2]:
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
            x = self.aspp(x)
        x = self.conv1(x)
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
    def __init__(self, in_c: int):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2D(in_c, 512, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2D(in_c, 512, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.LayerList(self.aspp)

        self.im_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2D(in_c, 512, 1, 1),
            nn.ReLU()
        )
        conv_after_dim = 512 * (len(self.aspp) + 1)
        self.conv_after = nn.Sequential(
            nn.Conv2D(conv_after_dim, 1024, 1, 1),
            nn.ReLU()
        )

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
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h, w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = paddle.cat(aspp_out, 1)
        x = self.conv_after(aspp_out)
        return x


class DOLGBackbone(TheseusLayer):
    """ DOLGBackbone model """
    def __init__(self,
                 stage3_channels: int,
                 stage4_channels: int,
                 with_aspp: bool = False,
                 globalmodel_pretrained: str = False):
        """DOLGBackbone model

        Args:
            stage3_channels (int): stage 3 channels
            stage4_channels (int): stage 4 channels
            with_aspp (bool, optional): whether use aspp for spatial attention. Defaults to False.
        """
        super(DOLGBackbone, self).__init__()
        self.globalmodel = ResNet50()
        delattr(self.globalmodel, 'fc')
        self.globalmodel.apply(init_weights)
        for i in range(len(self.globalmodel.blocks)):
            Constant(0.0)(self.globalmodel.blocks[i].conv2.bn.weight)
        _load_pretrained(globalmodel_pretrained, self.globalmodel, None, False)

        self.pool_global = GeneralizedMeanPoolingP(norm=3.0)
        self.fc_t = nn.Linear(stage4_channels, stage3_channels, bias_attr=True)
        KaimingUniform(fan_in=6*stage4_channels)(self.fc_t.weight)
        if self.fc_t.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.fc_t.weight.T)
            bound = 1.0 / math.sqrt(fan_in)
            Uniform(-bound, bound)(self.fc_t.bias)

        self.localmodel = SpatialAttention2d(stage3_channels, stage3_channels, "relu", with_aspp)
        self.pool_local = nn.AdaptiveAvgPool2D((1, 1))

        self.stage3_channels = stage3_channels
        self.stage4_channels = stage4_channels

    def globalmodel_forward(self, x):
        def stage_forward(l_idx: int, r_idx: int, x: paddle.Tensor):
            for i in range(l_idx, r_idx):
                x = self.globalmodel.blocks[i](x)
            return x

        with paddle.static.amp.fp16_guard():
            x = self.globalmodel.stem(x)
            x = self.globalmodel.max_pool(x)
            x = stage_forward(0, 3, x)
            x = stage_forward(3, 7, x)
            f3 = stage_forward(7, 13, x)
            f4 = stage_forward(13, 16, f3)
        return f3, f4

    def forward(self, x):
        """ Global and local orthogonal fusion """

        f3, f4 = self.globalmodel_forward(x)
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

        return final_feat


def DOLG(stage3_channels,
         stage4_channels,
         with_aspp,
         pretrained=False,
         **kwargs):
    model = DOLGBackbone(stage3_channels, stage4_channels, with_aspp, **kwargs)
    _load_pretrained(pretrained, model, None, False)
    return model


# if __name__ == "__main__":
#     import numpy as np
#     bb = DOLG(1024, 2048, 512, False)
#     num_param = 0
#     for k, v in bb.state_dict().items():
#         if hasattr(v, 'shape'):
#             num_param += np.prod(list(v.shape))
#     print(num_param)
#     exit(0)
