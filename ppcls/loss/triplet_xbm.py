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
from ppcls.utils import logger


class CrossBatchMemory(object):
    """Cross Batch Memory implemented with Circular Queue

    Args:
        size (int): memory size
        num_features (int): size of embedding dims
        ensemble_all_gpus (bool, optional): if gather embeddings from all other gpus. Defaults to False.
    """
    def __init__(self, size: int, num_features: int, ensemble_all_gpus: bool = False):
        self.K = size
        self.feats = paddle.zeros([size, num_features])
        self.targets = paddle.zeros([size, 1], dtype="int64")
        self.ptr = 0
        self.cur_size = 0
        self.ensemble_all_gpus = ensemble_all_gpus

    def _gather_from_gpu(self, gather_object: paddle.Tensor, concat_axis=0) -> paddle.Tensor:
        """gather Tensor from all gpus into a list and concatenate them on `concat_axis`.

        Args:
            gather_object (paddle.Tensor): gather object Tensor
            concat_axis (int, optional): axis for concatenation. Defaults to 0.

        Returns:
            paddle.Tensor: gatherd & concatenated Tensor
        """
        gather_object_list = []
        paddle.distributed.all_gather(gather_object_list, gather_object.cuda())
        return paddle.concat(gather_object_list, axis=concat_axis)

    @property
    def is_full(self):
        return self.cur_size >= self.K

    def get(self) -> paddle.Tensor:
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats: paddle.Tensor, targets: paddle.Tensor) -> None:
        qsize = targets.shape[0]
        if self.ensemble_all_gpus:
            feats = self._gather_from_gpu(feats, 0)
            targets = self._gather_from_gpu(targets, 0)
            qsize = targets.shape[0]

        if self.ptr + qsize > self.K:
            self.feats[-qsize:] = feats
            self.targets[-qsize:] = targets
            self.ptr = 0
            self.cur_size = self.K
        else:
            self.feats[self.ptr: self.ptr + qsize] = feats
            self.targets[self.ptr: self.ptr + qsize] = targets
            self.ptr += qsize
            self.cur_size += qsize


class TripletLossV2_XBM(nn.Layer):
    """TripHard Loss with memory bank according to 'Cross-Batch Memory for Embedding Learning'
    Args:
        xbm_start_iter (int): _description_
        xbm_size (int): _description_
        xbm_num_features (int): _description_
        margin (float, optional): _description_. Defaults to 0.5.
        normalize_feature (bool, optional): _description_. Defaults to True.
        feature_from (str, optional): _description_. Defaults to "features".
        softmargin (bool, optional): _description_. Defaults to False.
        xbm_ensemble_all_gpus (bool, optional): _description_. Defaults to False.
    """

    def __init__(self,
                 # xbm params below...
                 xbm_start_iter: int,
                 xbm_size: int,
                 xbm_num_features: int,
                 xbm_ensemble_all_gpus: bool,
                 margin: float = 0.5,
                 normalize_feature=True,
                 feature_from="features",
                 softmargin=False):
        super(TripletLossV2_XBM, self).__init__()
        self.margin = margin
        self.feature_from = feature_from
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature
        self.softmargin = softmargin
        self.xbm = CrossBatchMemory(xbm_size, xbm_num_features, xbm_ensemble_all_gpus)
        self.xbm_start_iter = xbm_start_iter
        self.cur_iter = 0

    def forward(self, input, target):
        inputs = input[self.feature_from]
        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        loss_cur = self.forward_cur_batch(inputs, target)
        loss = {"TripletLossV2": loss_cur}
        if self.cur_iter >= self.xbm_start_iter:
            loss_xbm = self.forward_xbm(inputs, target)
            loss["TripletLossV2_XBM"] = loss_xbm
        self.cur_iter += 1

        return loss

    def forward_cur_batch(self, inputs, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (num_classes)
        """
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
        ## both `dist_ap` and `relative_p_inds` with shape [N, 1]
        '''
        dist_ap, relative_p_inds = paddle.max(
            paddle.reshape(dist[is_pos], (bs, -1)), axis=1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = paddle.min(
            paddle.reshape(dist[is_neg], (bs, -1)), axis=1, keepdim=True)
        '''
        try:
            dist_ap = paddle.max(paddle.reshape(
                paddle.masked_select(dist, is_pos), (bs, -1)),
                                axis=1,
                                keepdim=True)
            # `dist_an` means distance(anchor, negative)
            # both `dist_an` and `relative_n_inds` with shape [N, 1]
            dist_an = paddle.min(paddle.reshape(
                paddle.masked_select(dist, is_neg), (bs, -1)),
                                axis=1,
                                keepdim=True)
            # shape [N]
            dist_ap = paddle.squeeze(dist_ap, axis=1)
            dist_an = paddle.squeeze(dist_an, axis=1)

            # Compute ranking hinge loss
            y = paddle.ones_like(dist_an)
            if self.softmargin:
                loss = nn.functional.softplus(dist_ap - dist_an).mean()
            else:
                loss = self.ranking_loss(dist_an, dist_ap, y)
            return loss
        except Exception as e:
            logger.info(e)
            logger.info(target)
            exit(0)

    def forward_xbm(self, inputs, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (num_classes)
        """
        if self.xbm.cur_size > 0:
            xbm_feats, xbm_targets = self.xbm.get()
            num_b = inputs.shape[0]
            num_x = xbm_feats.shape[0]

            # hard negative mining
            is_pos = paddle.expand(target, (
                num_b, num_x)).equal(paddle.expand(xbm_targets, (num_x, num_b)).t())  # [nb,nx]
            is_neg = paddle.expand(target, (
                num_b, num_x)).not_equal(paddle.expand(xbm_targets, (num_x, num_b)).t())  # [nb,nx]

            # compute distance
            dist = paddle.pow(inputs, 2).sum(axis=1, keepdim=True).expand([num_b, num_x]) + \
                paddle.pow(xbm_feats, 2).sum(axis=1, keepdim=True).expand([num_x, num_b]).t()  # [nb, nx]
            dist = dist.addmm(x=inputs, y=xbm_feats.t(), beta=1, alpha=-2)  # [nb,f]x[f,nx]+[nb,nx]
            dist = paddle.clip(dist, min=1e-12).sqrt()

            loss = paddle.zeros([1, ])
            for _i in range(num_b):
                pos_val = paddle.masked_select(dist[_i], is_pos[_i])
                if len(pos_val) > 0:
                    dist_ap_i = paddle.max(pos_val)

                    neg_val = paddle.masked_select(dist[_i], is_neg[_i])
                    if len(neg_val) > 0:
                        dist_an_i = paddle.min(neg_val)
                        # Compute ranking hinge loss
                        y = paddle.ones_like(dist_an_i)
                        if self.softmargin:
                            loss_i = nn.functional.softplus(dist_ap_i - dist_an_i).mean()
                        else:
                            loss_i = self.ranking_loss(dist_an_i, dist_ap_i, y)
                        loss += loss_i
            loss /= num_b
        else:
            loss = paddle.zeros([1, ])
        self.xbm.enqueue_dequeue(inputs.detach(), target.detach())
        return loss
