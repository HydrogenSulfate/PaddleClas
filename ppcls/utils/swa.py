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

from copy import deepcopy

import paddle
import paddle.distributed as dist
from paddle.nn import (BatchNorm, BatchNorm1D, BatchNorm2D, BatchNorm3D,
                       SyncBatchNorm)
from ppcls.utils import logger

from .ema import ExponentialMovingAverage


class StochasticWeightAverage(ExponentialMovingAverage):
    """
    Stochastic Weight Averaging
    refer to https://arxiv.org/abs/1803.05407
    """
    _BN_CLASSES = (BatchNorm, BatchNorm1D, BatchNorm2D, BatchNorm3D,
                   SyncBatchNorm)

    def __init__(self, model, cyclic_length=1):
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.count = 1
        self.cyclic_length = cyclic_length

    def update(self, model):
        # dynamic decay in SWA
        self.decay = (self.count / (self.count + 1))
        self._update(
            model,
            update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        self.count += 1

    @paddle.no_grad()
    def update_bn(self, loader, print_batch_step=20):
        # store statistics of bn modules in model
        momentum = {}
        bn_layers_cnt = 0
        for module in self.module.sublayers(True):
            if isinstance(module, self._BN_CLASSES):
                momentum[module] = module._momentum
                # reset mean an var to zero for precise-BN
                module._mean = paddle.zeros_like(module._mean)
                module._variance = paddle.ones_like(module._variance)
                bn_layers_cnt += 1

        # do nothing if there is no bn module
        if bn_layers_cnt == 0:
            return

        logger.info(f"Staring Update statistics for {bn_layers_cnt} BN layers")
        was_training = self.module.training
        self.module.train()
        # set momentum of bn modules to 0.0, for getting mean and var per batch
        for bn_module in momentum.keys():
            bn_module._momentum = 0.0

        # compute statistics of bn modules by forward in dataloader(always train_datalaoder) iterally
        inputs_seen = 0
        for i, batch in enumerate(loader):
            batch_size = batch[0].shape
            current_momentum = inputs_seen / (inputs_seen + batch_size)
            for bn_module in momentum.keys():
                bn_module._momentum = current_momentum

            # forward pass
            self.module(batch[0], batch[1])

            inputs_seen += batch_size
            if i % print_batch_step == 0 or (i + 1) == len(loader):
                logger.info(
                    f"Updating statistics for BatchNorm layers process: [{i}/{len(loader)}"
                )

        for bn_module in momentum.keys():
            bn_module._momentum = momentum[bn_module]

        self.module.train(was_training)
