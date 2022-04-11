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
from __future__ import absolute_import, division, print_function

import datetime
from ppcls.utils import logger
from ppcls.utils.misc import AverageMeter


def update_metric(trainer, out, batch, batch_size):
    # calc metric
    if trainer.train_metric_func is not None:
        metric_dict = trainer.train_metric_func(out, batch[-1])
        for key in metric_dict:
            if key not in trainer.output_info:
                trainer.output_info[key] = AverageMeter(key, '7.5f')
            trainer.output_info[key].update(metric_dict[key].numpy()[0],
                                            batch_size)


def update_loss(trainer, loss_dict, batch_size):
    # update_output_info
    for key in loss_dict:
        if key not in trainer.output_info:
            trainer.output_info[key] = AverageMeter(key, '7.5f')
        trainer.output_info[key].update(loss_dict[key].numpy()[0], batch_size)


def log_info(trainer, batch_size, epoch_id, iter_id):
    if len(trainer.lr_sch) <= 1:
        lr_msg = "lr: {:.8f}".format(trainer.lr_sch[0].get_lr())
    else:
        lr_msg = "lr_model: {:.8f}".format(trainer.lr_sch[0].get_lr())
        lr_msg += ", lr_loss: {:.8f}".format(trainer.lr_sch[1].get_lr())

    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, trainer.output_info[key].avg)
        for key in trainer.output_info
    ])
    time_msg = "s, ".join([
        "{}: {:.5f}".format(key, trainer.time_info[key].avg)
        for key in trainer.time_info
    ])

    ips_msg = "ips: {:.5f} samples/s".format(
        batch_size / trainer.time_info["batch_cost"].avg)
    eta_sec = ((trainer.config["Global"]["epochs"] - epoch_id + 1
                ) * len(trainer.train_dataloader) - iter_id
               ) * trainer.time_info["batch_cost"].avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    logger.info("[Train][Epoch {}/{}][Iter: {}/{}]{}, {}, {}, {}, {}".format(
        epoch_id, trainer.config["Global"]["epochs"], iter_id,
        len(trainer.train_dataloader), lr_msg, metric_msg, time_msg, ips_msg,
        eta_msg))

    if len(trainer.lr_sch) <= 1:
        logger.scaler(
            name="lr",
            value=trainer.lr_sch[0].get_lr(),
            step=trainer.global_step,
            writer=trainer.vdl_writer)
    else:
        logger.scaler(
            name="lr_model",
            value=trainer.lr_sch[0].get_lr(),
            step=trainer.global_step,
            writer=trainer.vdl_writer)
        logger.scaler(
            name="lr_loss",
            value=trainer.lr_sch[1].get_lr(),
            step=trainer.global_step,
            writer=trainer.vdl_writer)

    for key in trainer.output_info:
        logger.scaler(
            name="train_{}".format(key),
            value=trainer.output_info[key].avg,
            step=trainer.global_step,
            writer=trainer.vdl_writer)
