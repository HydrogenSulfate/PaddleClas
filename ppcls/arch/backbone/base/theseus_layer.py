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

from typing import Tuple, List, Dict, Union, Callable, Any

import paddle
from paddle import nn
from ppcls.utils import logger


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = {}
        self.res_name = self.full_name()
        self.pruner = None
        self.quanter = None

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"logits": output}
        # 'list' is needed to avoid error raised by popping self.res_dict
        for res_key in list(self.res_dict):
            # clear the res_dict because the forward process may change according to input
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def init_res(self,
                 stages_pattern,
                 return_patterns=None,
                 return_stages=None):
        if return_patterns and return_stages:
            msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
            logger.warning(msg)
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern
        # return_stages is int or bool
        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(
                    return_stages) < 0:
                msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                logger.warning(msg)
                return_stages = [
                    val for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

        if return_patterns:
            self.update_res(return_patterns)

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "The function 'replace_sub()' is deprecated, please use 'upgrade_sublayer()' instead."
        logger.error(DeprecationWarning(msg))
        raise DeprecationWarning(msg)

    def upgrade_sublayer(self,
                         layer_name_pattern: Union[str, List[str]],
                         handle_func: Callable[[nn.Layer, str], nn.Layer]
                         ) -> Dict[str, nn.Layer]:
        """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.

        Args:
            layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
            handle_func (Callable[[nn.Layer, str], nn.Layer]): The function to modify target layer specified by 'layer_name_pattern'. The formal params are the layer(nn.Layer) and pattern(str) that is (a member of) layer_name_pattern (when layer_name_pattern is List type). And the return is the layer processed.

        Returns:
            Dict[str, nn.Layer]: The key is the pattern and corresponding value is the result returned by 'handle_func()'.

        Examples:

            from paddle import nn
            import paddleclas

            def rep_func(layer: nn.Layer, pattern: str):
                new_layer = nn.Conv2D(
                    in_channels=layer._in_channels,
                    out_channels=layer._out_channels,
                    kernel_size=5,
                    padding=2
                )
                return new_layer

            net = paddleclas.MobileNetV1()
            res = net.replace_sub(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
            print(res)
            # {'blocks[11].depthwise_conv.conv': the corresponding new_layer, 'blocks[12].depthwise_conv.conv': the corresponding new_layer}
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        hit_layer_pattern_list = []
        for pattern in layer_name_pattern:
            # parse pattern to find target layer and its parent
            layer_list = parse_pattern_str(pattern=pattern, parent_layer=self)
            if not layer_list:
                continue
            sub_layer_parent = layer_list[-2]["layer"] if len(
                layer_list) > 1 else self

            sub_layer = layer_list[-1]["layer"]
            sub_layer_name = layer_list[-1]["name"]
            sub_layer_index = layer_list[-1]["index"]

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index:
                getattr(sub_layer_parent,
                        sub_layer_name)[sub_layer_index] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            hit_layer_pattern_list.append(pattern)
        return hit_layer_pattern_list

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        layer_list = parse_pattern_str(stop_layer_name, self)
        if not layer_list:
            return False

        parent_layer = self
        for layer_dict in layer_list:
            name, index = layer_dict["name"], layer_dict["index"]
            if not set_identity(parent_layer, name, index):
                msg = f"Failed to set the layers that after stop_layer_name('{stop_layer_name}') to IdentityLayer. The error layer's name is '{name}'."
                logger.warning(msg)
                return False
            parent_layer = layer_dict["layer"]

        return True

    def update_res(
            self,
            return_patterns: Union[str, List[str]]) -> Dict[str, nn.Layer]:
        """update the result(s) to be returned.

        Args:
            return_patterns (Union[str, List[str]]): The name of layer to return output.

        Returns:
            Dict[str, nn.Layer]: The pattern(str) and corresponding layer(nn.Layer) that have been set successfully.
        """

        # clear res_dict that could have been set
        self.res_dict = {}

        class Handler(object):
            def __init__(self, res_dict):
                # res_dict is a reference
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                if hasattr(layer, "hook_remove_helper"):
                    layer.hook_remove_helper.remove()
                layer.hook_remove_helper = layer.register_forward_post_hook(
                    save_sub_res_hook)
                return layer

        handle_func = Handler(self.res_dict)

        hit_layer_pattern_list = self.upgrade_sublayer(
            return_patterns, handle_func=handle_func)

        if hasattr(self, "hook_remove_helper"):
            self.hook_remove_helper.remove()
        self.hook_remove_helper = self.register_forward_post_hook(
            self._return_dict_hook)

        return hit_layer_pattern_list


def save_sub_res_hook(layer, input, output):
    layer.res_dict[layer.res_name] = output


def set_identity(parent_layer: nn.Layer,
                 layer_name: str,
                 layer_index: str = None) -> bool:
    """set the layer specified by layer_name and layer_index to Indentity.

    Args:
        parent_layer (nn.Layer): The parent layer of target layer specified by layer_name and layer_index.
        layer_name (str): The name of target layer to be set to Indentity.
        layer_index (str, optional): The index of target layer to be set to Indentity in parent_layer. Defaults to None.

    Returns:
        bool: True if successfully, False otherwise.
    """

    stop_after = False
    for sub_layer_name in parent_layer._sub_layers:
        if stop_after:
            parent_layer._sub_layers[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index and stop_after:
        stop_after = False
        for sub_layer_index in parent_layer._sub_layers[
                layer_name]._sub_layers:
            if stop_after:
                parent_layer._sub_layers[layer_name][
                    sub_layer_index] = Identity()
                continue
            if layer_index == sub_layer_index:
                stop_after = True

    return stop_after


def parse_pattern_str(pattern: str, parent_layer: nn.Layer) -> Union[
        None, List[Dict[str, Union[nn.Layer, str, None]]]]:
    """parse the string type pattern.

    Args:
        pattern (str): The pattern to discribe layer.
        parent_layer (nn.Layer): The root layer relative to the pattern.

    Returns:
        Union[None, List[Dict[str, Union[nn.Layer, str, None]]]]: None if failed. If successfully, the members are layers parsed in order:
                                                                [
                                                                    {"layer": first layer, "name": first layer's name parsed, "index": first layer's index parsed if exist},
                                                                    {"layer": second layer, "name": second layer's name parsed, "index": second layer's index parsed if exist},
                                                                    ...
                                                                ]
    """

    pattern_list = pattern.split(".")
    if not pattern_list:
        msg = f"The pattern('{pattern}') is illegal. Please check and retry."
        logger.warning(msg)
        return None

    layer_list = []
    while len(pattern_list) > 0:
        if '[' in pattern_list[0]:
            target_layer_name = pattern_list[0].split('[')[0]
            target_layer_index = pattern_list[0].split('[')[1].split(']')[0]
        else:
            target_layer_name = pattern_list[0]
            target_layer_index = None

        target_layer = getattr(parent_layer, target_layer_name, None)

        if target_layer is None:
            msg = f"Not found layer named('{target_layer_name}') specifed in pattern('{pattern}')."
            logger.warning(msg)
            return None

        if target_layer_index and target_layer:
            if int(target_layer_index) < 0 or int(target_layer_index) >= len(
                    target_layer):
                msg = f"Not found layer by index('{target_layer_index}') specifed in pattern('{pattern}'). The index should < {len(target_layer)} and > 0."
                logger.warning(msg)
                return None

            target_layer = target_layer[target_layer_index]

        layer_list.append({
            "layer": target_layer,
            "name": target_layer_name,
            "index": target_layer_index
        })

        pattern_list = pattern_list[1:]
        parent_layer = target_layer
    return layer_list


class CrissCrossAttention(nn.Layer):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.q_conv = nn.Conv2D(in_channels, in_channels // reduction, kernel_size=1)
        self.k_conv = nn.Conv2D(in_channels, in_channels // reduction, kernel_size=1)
        self.v_conv = nn.Conv2D(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(axis=3)
        self.gamma = self.create_parameter(
            shape=(1, ), default_initializer=nn.initializer.Constant(0))
        self.inf_tensor = paddle.full(shape=(1, ), fill_value=float('inf'))

    def forward(self, x):
        b, c, h, w = paddle.shape(x)
        proj_q = self.q_conv(x)
        proj_q_h = proj_q.transpose([0, 3, 1, 2]).reshape(
            [b * w, -1, h]).transpose([0, 2, 1])
        proj_q_w = proj_q.transpose([0, 2, 1, 3]).reshape(
            [b * h, -1, w]).transpose([0, 2, 1])

        proj_k = self.k_conv(x)
        proj_k_h = proj_k.transpose([0, 3, 1, 2]).reshape([b * w, -1, h])
        proj_k_w = proj_k.transpose([0, 2, 1, 3]).reshape([b * h, -1, w])

        proj_v = self.v_conv(x)
        proj_v_h = proj_v.transpose([0, 3, 1, 2]).reshape([b * w, -1, h])
        proj_v_w = proj_v.transpose([0, 2, 1, 3]).reshape([b * h, -1, w])

        energy_h = (paddle.bmm(proj_q_h, proj_k_h) + self.Inf(b, h, w)).reshape(
            [b, w, h, h]).transpose([0, 2, 1, 3])
        energy_w = paddle.bmm(proj_q_w, proj_k_w).reshape([b, h, w, w])
        concate = self.softmax(paddle.concat([energy_h, energy_w], axis=3))

        attn_h = concate[:, :, :, 0:h].transpose([0, 2, 1, 3]).reshape(
            [b * w, h, h])
        attn_w = concate[:, :, :, h:h + w].reshape([b * h, w, w])
        out_h = paddle.bmm(proj_v_h, attn_h.transpose([0, 2, 1])).reshape(
            [b, w, -1, h]).transpose([0, 2, 3, 1])
        out_w = paddle.bmm(proj_v_w, attn_w.transpose([0, 2, 1])).reshape(
            [b, h, -1, w]).transpose([0, 2, 1, 3])
        return self.gamma * (out_h + out_w) + x

    def Inf(self, B, H, W):
        return -paddle.tile(
            paddle.diag(paddle.tile(self.inf_tensor, [H]), 0).unsqueeze(0),
            [B * W, 1, 1])


class RCCAModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 reduction,
                 recurrence=2):
        super().__init__()
        self.recurrence = recurrence
        self.reduce = nn.Conv2D(in_channels, inter_channels, 1, padding=0, bias_attr=False)
        self.cca = CrissCrossAttention(inter_channels, reduction)
        self.W = nn.Sequential(
            nn.Conv2D(in_channels=inter_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(in_channels),
        )
        nn.initializer.Constant(0.0)(self.W[1].weight)
        nn.initializer.Constant(0.0)(self.W[1].bias)

    def forward(self, x):
        feat = self.reduce(x)
        for i in range(self.recurrence):
            feat = self.cca(feat)
        feat = self.W(feat)
        return x + feat


class GeneralizedMeanPooling(nn.Layer):
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clip(min=self.eps).pow(self.p)
        return nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = self.create_parameter(
            shape=(1, ),
            default_initializer=nn.initializer.Constant(1. * norm)
        )
        self.add_parameter("p", self.p)


class GeneralizedMeanPoolingScale(GeneralizedMeanPoolingP):
    """增强数值大小，无论正负
    """
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = self.create_parameter(
            shape=(1, ),
            default_initializer=nn.initializer.Constant(1. * norm)
        )
        self.add_parameter("p", self.p)

    def forward(self, x):
        x_abs = paddle.abs(x)
        x_sign = paddle.sign(x)
        return nn.functional.adaptive_avg_pool2d(x_abs.pow(self.p) * x_sign, self.output_size).pow(1. / self.p)


class GeneralizedMeanPoolingMin(GeneralizedMeanPoolingP):
    """增强数值大小，无论正负
    """
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = self.create_parameter(
            shape=(1, ),
            default_initializer=nn.initializer.Constant(1. * norm)
        )
        self.add_parameter("p", self.p)
        self.bn = nn.BatchNorm2D(1024, bias_attr=paddle.ParamAttr(learning_rate=1e-20))

    def forward(self, x):
        x_min = paddle.min(x, axis=[2, 3], keepdim=True)
        x = (x - x_min).pow(self.p)
        x = nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p) + x_min
        return self.bn(x)


class GeneralizedMeanPoolingAugPos(GeneralizedMeanPoolingP):
    """增强正数数值，保留负数数值
    """
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = self.create_parameter(
            shape=(1, ),
            default_initializer=nn.initializer.Constant(1. * norm)
        )
        self.add_parameter("p", self.p)

    def forward(self, x):
        x_aug = x.clip(min=self.eps).pow(self.p)
        return nn.functional.adaptive_avg_pool2d(paddle.where(x < self.eps, x, x_aug), self.output_size).pow(1. / self.p)


class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        att = paddle.concat([avg_out, max_out], axis=1)
        att = self.conv1(att)
        att = self.sigmoid(att)
        return x * att, att


class RGAModule(nn.Layer):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGAModule, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        logger.info(f"{'=' * 10} Use_Spatial_Att: {self.use_spatial}, Use_Channel_Att: {self.use_channel}")

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2D(in_channels=self.in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2D(in_channels=self.in_spatial, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2D(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2D(in_channels=self.in_channel*2, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2D(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(num_channel_s//down_ratio),
                nn.ReLU(),
                nn.Conv2D(in_channels=num_channel_s//down_ratio, out_channels=1, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(1)
            )
        if self.use_channel:    
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2D(in_channels=num_channel_c, out_channels=num_channel_c//down_ratio, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(num_channel_c//down_ratio),
                nn.ReLU(),
                nn.Conv2D(in_channels=num_channel_c//down_ratio, out_channels=1, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2D(in_channels=self.in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2D(in_channels=self.in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2D(in_channels=self.in_spatial, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2D(in_channels=self.in_spatial, out_channels=self.inter_spatial, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.shape

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.reshape([b, self.inter_channel, -1])
            theta_xs = theta_xs.transpose((0, 2, 1))
            phi_xs = phi_xs.reshape([b, self.inter_channel, -1])
            Gs = paddle.matmul(theta_xs, phi_xs)
            Gs_in = Gs.transpose((0, 2, 1)).reshape([b, h*w, h, w])
            Gs_out = Gs.reshape([b, h*w, h, w])
            Gs_joint = paddle.concat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)

            g_xs = self.gx_spatial(x)
            g_xs = paddle.mean(g_xs, axis=1, keepdim=True)
            ys = paddle.concat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            if not self.use_channel:
                out = nn.functional.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                x = nn.functional.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # channel attention
            xc = x.reshape([b, c, -1]).transpose((0, 2, 1)).unsqueeze(-1)
            theta_xc = self.theta_channel(xc).squeeze(-1).transpose((0, 2, 1))
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = paddle.matmul(theta_xc, phi_xc)
            Gc_in = Gc.transpose((0, 2, 1)).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = paddle.concat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)

            g_xc = self.gx_channel(xc)
            g_xc = paddle.mean(g_xc, axis=1, keepdim=True)
            yc = paddle.concat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose((0, 2, 1, 3))
            out = nn.functional.sigmoid(W_yc) * x

            return out
