# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import, division

import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import paddle
from paddle.io import DistributedBatchSampler
from ppcls.utils import logger


class GraphSampler(DistributedBatchSampler):
    """
    First, randomly sample P identities.
    Then for each identity randomly sample K instances.
    Therefore batch size is P*K, and the sampler called GraphSampler.
    Args:
        dataset (paddle.io.Dataset): list of (img_path, pid, cam_id).
        num_instance(int): number of instances per identity in a batch.
        batch_size (int): number of examples in a batch.
        shuffle(bool): whether to shuffle indices order before generating
            batch indices. Default False.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instance,
                 shuffle=True,
                 drop_last=True,
                 model=None,
                 memory_size=1024,
                 test_transform_ops=None):
        super().__init__(
            dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        self.model = model
        assert batch_size % num_instance == 0, \
            "GraphSampler configs error, num_instance must be a divisor of batch_size."
        assert hasattr(self.dataset,
                       "labels"), "Dataset must have labels attribute."
        self.num_instance = num_instance
        self.num_people = batch_size // num_instance
        self.memory_size = memory_size
        self.test_transform_ops = test_transform_ops
        self.label_dict = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.label_dict[label].append(idx)  # pid->index
        self.label_list = list(self.label_dict.keys())
        self.num_labels = len(self.label_list)
        for pid in self.label_list:
            random.shuffle(self.label_dict[pid])

        self.repr_index = None
        self.repr_pointer = [0 for _ in range(self.num_labels)]

        assert len(self.label_list) * self.num_instance > self.batch_size, \
            "batch size should be smaller than len(self.label_list) * self.num_instance"
        self.prob_list = np.array([1 / len(self.label_list)] * len(self.label_list))
        diff = np.abs(sum(self.prob_list) - 1)
        if diff > 0.00000001:
            self.prob_list[-1] = 1 - sum(self.prob_list[:-1])
            if self.prob_list[-1] > 1 or self.prob_list[-1] < 0:
                logger.error("GraphSampler prob list error")
            else:
                logger.info(
                    "GraphSampler: sum of prob list not equal to 1, diff is {}, change the last prob".format(diff)
                )

    @paddle.no_grad()
    def _extract_features(self, model, data_loader) -> paddle.Tensor:
        model.eval()
        features = []
        for i, batch in enumerate(data_loader):
            if i % 20 == 0:
                logger.info(f"GraphSampler._extract_features [{i}]/[{len(data_loader)}]")
            outputs = model(batch[0], batch[1])
            features.append(outputs["features"].cpu())
        features = paddle.concat(features, axis=0)
        return features

    def __iter__(self):
        repr_index = []
        for pid in self.label_list:
            index = np.random.choice(self.label_dict[pid], size=1)[0]
            repr_index.append(index)
        # repr_index的每个下标都对应某个类里的一条数据
        repr_dataset = deepcopy(self.dataset)
        repr_dataset.images = [repr_dataset.images[i] for i in repr_index]
        repr_dataset.labels = [repr_dataset.labels[i] for i in repr_index]
        repr_dataset.set_transform_ops(self.test_transform_ops)
        repr_data_loader = paddle.io.DataLoader(
            dataset=repr_dataset,
            batch_size=64,
            num_workers=0,
            drop_last=False,
            shuffle=False,
        )
        logger.info(f"{'='*10} len(repr_dataloader)={len(repr_data_loader)}")

        features = self._extract_features(self.model, repr_data_loader)  # [num_classes, F]
        logger.info(f"{'=' * 10} extracted_repr_features.shape={features.shape}")

        topk_index = np.zeros([self.num_labels, self.num_people - 1], dtype="int64")  # [num_classes, P-1]
        with paddle.no_grad():
            for i in range(0, self.num_labels, self.memory_size):
                ed = min(self.num_labels, i + self.memory_size)
                bsz = ed - i
                feat_block = features[i:ed]
                dist2 = (paddle.pow(feat_block, 2).sum(axis=1, keepdim=True).expand([bsz, self.num_labels]) +
                         paddle.pow(features, 2).sum(axis=1, keepdim=True).expand([self.num_labels, bsz]).t()).addmm(x=feat_block, y=features.t(), beta=1, alpha=-2)  # [m, num_classes]
                dist2[:, i:ed] = dist2[:, i:ed] + paddle.eye(bsz) * 1e15  # [bsz, num_classes]
                # dist2 = dist2 / 10
                # dist2 = paddle.nn.functional.sigmoid(dist2)
                # assert dist2.min() >= 0.0, \
                #     f"dist2.min() should >=0, but got {dist2.min().item()}"
                block_index = paddle.argsort(dist2, axis=1, descending=False)[:, :self.num_people - 1]  # 取前k-1个距离最近的其它样本

                topk_index[i:ed] = block_index.cpu().numpy()

        i_list = list(range(self.num_labels))
        random.shuffle(i_list)
        for i in i_list:  # 循环num_labels次
            id_index = topk_index[i, :].tolist()  # [P-1, ]
            # assert i not in id_index, f"i({i}) already in id_index({id_index})"
            id_index.append(i)  # [P, ]
            index = []
            for j in id_index:  # P个人的下标，每个人随机选K个
                pid = self.label_list[j]
                img_indexes = self.label_dict[pid]
                len_p = len(img_indexes)
                index_p = []
                remain = self.num_instance  # K
                while remain > 0:
                    end = self.repr_pointer[j] + remain
                    idx = img_indexes[self.repr_pointer[j]:end]
                    index_p.extend(idx)
                    remain -= len(idx)
                    self.repr_pointer[j] = end
                    if end >= len_p:
                        random.shuffle(img_indexes)
                        self.repr_pointer[j] = 0
                assert(len(index_p) == self.num_instance)
                index.extend(index_p)  # K个压入
            yield index  # 返回batchsize(P*K)个下标

    # def __len__(self):
    #     return len(self.num_labels)
