#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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


def calculate_gain(nonlinearity, param=None):
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d'
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(
                param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(
                "negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(
            mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation
    with paddle.no_grad():
        tensor.set_value(paddle.uniform(tensor.shape, min=-bound, max=bound))
        return tensor


class SoftTriple(nn.Layer):
    def __init__(self, la, gamma, tau, margin, feat_dim, class_num, K, feature_from='features'):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1.0 / gamma
        self.tau = tau
        self.margin = margin
        self.class_num = class_num
        self.K = K
        self.feature_from = feature_from
        # self.fc = Parameter(torch.Tensor(feat_dim, class_num*K))
        self.fc = self.create_parameter(shape=(self.num_classes, self.feat_dim))
        self.add_parameter("fc", self.fc)
        self.weight = paddle.zeros([class_num * K, class_num * K], dtype=paddle.bool)
        for i in range(0, class_num):
            for j in range(0, K):
                self.weight[i * K + j, i * K + j + 1:(i + 1) * K] = True
        kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, input, target):
        input = input[self.feature_from]
        centers = nn.functional.normalize(self.fc, p=2, axis=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.class_num, self.K)
        prob = nn.functional.softmax(simStruc * self.gamma, axis=2)
        simClass = paddle.sum(prob * simStruc, axis=2)
        marginM = paddle.zeros_like(simClass)
        marginM[paddle.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = nn.functional.cross_entropy(self.la * (simClass - marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = paddle.sum(
                paddle.sqrt(2.0 + 1e-5 - 2.0 * paddle.mask_select(simCenter, self.weight))) \
                / (self.class_num * self.K * (self.K - 1.0))
            return lossClassify + self.tau * reg
        else:
            return lossClassify
