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


class TripletLossV2_cos(nn.Layer):
    """Triplet loss with hard positive/negative mining.
    paper : [Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/pdf/1503.03832.pdf)
    code reference: https://github.com/okzhili/Cartoon-face-recognition/blob/master/loss/triplet_loss.py
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self,
                 margin=0.5,
                #  normalize_feature=True,
                 feature_from="features"):
        super(TripletLossV2_cos, self).__init__()
        self.margin = margin
        self.feature_from = feature_from
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)
        # self.normalize_feature = normalize_feature

    def forward(self, input, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (num_classes)
        """
        inputs = input[self.feature_from]

        # if self.normalize_feature:
        inputs = 1.0 * inputs / (paddle.expand_as(
            paddle.norm(
                inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        bs = inputs.shape[0]

        # compute similarity
        sim = paddle.matmul(inputs, inputs, transpose_y=True)  # [n,m]  # [bs,bs], value in [0,4]
        # hard negative mining
        is_pos = paddle.expand(target, (
            bs, bs)).equal(paddle.expand(target, (bs, bs)).t())
        is_neg = paddle.expand(target, (
            bs, bs)).not_equal(paddle.expand(target, (bs, bs)).t())

        # `sim_ap` means similarity(anchor, positive)
        ## both `sim_ap` and `relative_p_inds` with shape [N, 1]
        '''
        sim_ap, relative_p_inds = paddle.max(
            paddle.reshape(sim[is_pos], (bs, -1)), axis=1, keepdim=True)
        # `sim_an` means similarity(anchor, negative)
        # both `sim_an` and `relative_n_inds` with shape [N, 1]
        sim_an, relative_n_inds = paddle.min(
            paddle.reshape(sim[is_neg], (bs, -1)), axis=1, keepdim=True)
        '''
        sim_ap = paddle.min(paddle.reshape(
            paddle.masked_select(sim, is_pos), (bs, -1)),
                             axis=1,
                             keepdim=True)  # 最难的正样本->sim最小
        # `sim_an` means similarity(anchor, negative)
        # both `sim_an` and `relative_n_inds` with shape [N, 1]
        sim_an = paddle.max(paddle.reshape(
            paddle.masked_select(sim, is_neg), (bs, -1)),
                             axis=1,
                             keepdim=True)  # 最难的负样本->sim最大
        # shape [N]
        sim_ap = paddle.squeeze(sim_ap, axis=1)
        sim_an = paddle.squeeze(sim_an, axis=1)

        # Compute ranking hinge loss
        y = paddle.ones_like(sim_an)
        loss = self.ranking_loss(sim_ap, sim_an, y)  # 优化目标：[san - sap + m]+
        return {"TripletLossV2_cos": loss}


class TripletLoss(nn.Layer):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)

    def forward(self, input, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (num_classes)
        """
        inputs = input["features"]

        bs = inputs.shape[0]
        # Compute pairwise distance, replace by the official when merged
        dist = paddle.pow(inputs, 2).sum(axis=1, keepdim=True).expand([bs, bs])
        dist = dist + dist.t()
        dist = paddle.addmm(
            input=dist, x=inputs, y=inputs.t(), alpha=-2.0, beta=1.0)
        dist = paddle.clip(dist, min=1e-12).sqrt()

        mask = paddle.equal(
            target.expand([bs, bs]), target.expand([bs, bs]).t())
        mask_numpy_idx = mask.numpy()
        dist_ap, dist_an = [], []
        for i in range(bs):
            # dist_ap_i = paddle.to_tensor(dist[i].numpy()[mask_numpy_idx[i]].max(),dtype='float64').unsqueeze(0)
            # dist_ap_i.stop_gradient = False
            # dist_ap.append(dist_ap_i)
            dist_ap.append(
                max([
                    dist[i][j] if mask_numpy_idx[i][j] == True else float(
                        "-inf") for j in range(bs)
                ]).unsqueeze(0))
            # dist_an_i = paddle.to_tensor(dist[i].numpy()[mask_numpy_idx[i] == False].min(), dtype='float64').unsqueeze(0)
            # dist_an_i.stop_gradient = False
            # dist_an.append(dist_an_i)
            dist_an.append(
                min([
                    dist[i][k] if mask_numpy_idx[i][k] == False else float(
                        "inf") for k in range(bs)
                ]).unsqueeze(0))

        dist_ap = paddle.concat(dist_ap, axis=0)
        dist_an = paddle.concat(dist_an, axis=0)

        # Compute ranking hinge loss
        y = paddle.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return {"TripletLoss": loss}
