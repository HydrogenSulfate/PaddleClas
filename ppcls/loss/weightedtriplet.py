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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn


class SoftMarginLoss(nn.Layer):
    """SoftMarginLoss
    """
    def __init__(self, reduce=True):
        super(SoftMarginLoss, self).__init__()
        self.reduce = reduce

    def forward(self, logits, labels):
        """forward
        """
        B = logits.shape[0]
        loss = paddle.sum(
            paddle.log(
                1 + paddle.exp(-labels * logits)
            )
        )
        if self.reduce:
            loss /= B
        return loss


class WeightedRegularizedTriplet(nn.Layer):
    """WeightedRegularizedTriplet
    """
    def __init__(self, normalize_feature=True, feature_from="features"):
        super(WeightedRegularizedTriplet, self).__init__()
        self.ranking_loss = SoftMarginLoss()
        self.normalize_feature = normalize_feature
        self.feature_from = feature_from

    def _softmax_weights(self, dist, mask):
        """_softmax_weights
        """
        max_v = paddle.max(dist * mask, axis=1, keepdim=True)  # [N, 1]
        diff = dist - max_v  # [N, N]
        Z = paddle.sum(paddle.exp(diff) * mask, axis=1, keepdim=True) + 1e-6  # avoid division by zero  # [N, 1]
        W = paddle.exp(diff) * mask / Z  # [N, N]
        return W

    def forward(self, input, target):
        """forward
        """
        inputs = input[self.feature_from]

        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        bs = inputs.shape[0]

        # compute distance
        dist = paddle.pow(inputs, 2).sum(axis=1, keepdim=True).expand([bs, bs])
        dist = dist + dist.t()
        dist = paddle.addmm(
            input=dist, x=inputs, y=inputs.t(), alpha=-2.0, beta=1.0)
        dist = paddle.clip(dist, min=1e-12).sqrt()

        # hard negative mining
        is_pos = paddle.expand(target, (
            bs, bs)).equal(paddle.expand(target, (bs, bs)).t())
        is_neg = paddle.expand(target, (
            bs, bs)).not_equal(paddle.expand(target, (bs, bs)).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [bs, 1]
        '''
        dist_ap, relative_p_inds = paddle.max(
            paddle.reshape(dist[is_pos], (bs, -1)), axis=1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [bs, 1]
        dist_an, relative_n_inds = paddle.min(
            paddle.reshape(dist[is_neg], (bs, -1)), axis=1, keepdim=True)
        '''
        dist_ap = dist * is_pos
        dist_an = dist * is_neg

        weights_ap = self._softmax_weights(dist_ap, is_pos)
        weights_an = self._softmax_weights(-dist_an, is_neg)
        furthest_positive = paddle.sum(dist_ap * weights_ap, axis=1)
        closest_negative = paddle.sum(dist_an * weights_an, axis=1)

        y = paddle.ones_like(furthest_positive)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return {"WeightedRegularizedTriplet": loss}
