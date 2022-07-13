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

import jpeg4py as jpeg
import numpy as np
from PIL import Image
from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .common_dataset import CommonDataset


class ImageNetDataset(CommonDataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform_ops=None,
            delimiter=None,
            relabel=False,
            backend="cv2",
            use_jpeg4py=False):
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        self.backend = backend
        self.use_jpeg4py = use_jpeg4py
        super(ImageNetDataset, self).__init__(image_root, cls_label_path, transform_ops)

    def __getitem__(self, idx):
        try:
            if self.use_jpeg4py:
                image = jpeg.JPEG(self.images[idx]).decode()
                image = Image.fromarray(image).convert('RGB')
            else:
                img = Image.open(self.images[idx]).convert('RGB')
            if self.backend == "cv2":
                img = np.array(img, dtype="float32").astype(np.uint8)
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            if self.backend == "cv2":
                img = img.transpose((2, 0, 1))
            return (img, self.labels[idx])
        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def set_transform_ops(self, transform_ops):
        self._transform_ops = transform_ops

    def _load_anno(self, seed=None):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)

            label_set = set()
            for line in lines:
                line = line.strip().split()
                label_set.add(int(line[1]))
            oldlabel_2_newlabel = {oldlabel: newlabel for newlabel, oldlabel in enumerate(label_set)}

            for l in lines:
                l = l.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, l[0]))
                if self.relabel:
                    self.labels.append(oldlabel_2_newlabel[np.int64(l[1])])
                else:
                    self.labels.append(np.int64(l[1]))
                # self.labels.append(np.int64(l[1]))
                assert os.path.exists(self.images[-1])
        logger.info(f"images: {len(self.images)}, labels: {len(label_set)}({min(self.labels)}~{max(self.labels)})")
