#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os

import cv2
import numpy as np
from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .common_dataset import CommonDataset


class LandmarkDataset(CommonDataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform_ops=None,
            delimiter=None):
        self.delimiter = delimiter if delimiter is not None else " "
        super(LandmarkDataset, self).__init__(image_root, cls_label_path, transform_ops)

    def _load_anno(self, seed=None):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, line[0]))
                self.labels.append(np.int64(line[1]))
                assert os.path.exists(self.images[-1])
        logger.info(f"# of images {len(self.images)}")
        logger.info(f"# of labels {len(set(self.labels))}")
        # self.images = self.images[:1000]
        # self.labels = self.labels[:1000]

    def __getitem__(self, idx):
        try:
            # 二进制读取
            with open(self.images[idx], "rb") as f:
                img = f.read()
            data = np.frombuffer(img, dtype='uint8')
            img = cv2.imdecode(data, 1)

            # imread读取
            # img = cv2.imread(self.images[idx])

            img = img.astype(np.float32, copy=False)
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            return (img, self.labels[idx])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
