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

# reference: https://arxiv.org/abs/1801.07698

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class ArcMargin(nn.Layer):
    def __init__(self,
                 embedding_size,
                 class_num,
                 margin=0.5,
                 scale=80.0,
                 easy_margin=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.weight = self.create_parameter(
            shape=[self.embedding_size, self.class_num],
            is_bias=False,
            default_initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, input, label=None):
        input_norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=1, keepdim=True))
        input = paddle.divide(input, input_norm)

        weight_norm = paddle.sqrt(
            paddle.sum(paddle.square(self.weight), axis=0, keepdim=True))
        weight = paddle.divide(self.weight, weight_norm)

        cos = paddle.matmul(input, weight)
        if not self.training or label is None:
            return cos
        sin = paddle.sqrt(1.0 - paddle.square(cos) + 1e-6)
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        phi = cos * cos_m - sin * sin_m

        th = math.cos(self.margin) * (-1)
        mm = math.sin(self.margin) * self.margin
        if self.easy_margin:
            phi = self._paddle_where_more_than(cos, 0, phi, cos)
        else:
            phi = self._paddle_where_more_than(cos, th, phi, cos - mm)

        one_hot = paddle.nn.functional.one_hot(label, self.class_num)
        one_hot = paddle.squeeze(one_hot, axis=[1])
        output = paddle.multiply(one_hot, phi) + paddle.multiply(
            (1.0 - one_hot), cos)
        output = output * self.scale
        return output

    def _paddle_where_more_than(self, target, limit, x, y):
        mask = paddle.cast(x=(target > limit), dtype='float32')
        output = paddle.multiply(mask, x) + paddle.multiply((1.0 - mask), y)
        return output


class ArcMarginP(nn.Layer):
    def __init__(self,
                 embedding_size,
                 class_num,
                 margin=0.5,
                 scale=80.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.margin = margin
        self.scale = scale

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        weight = self.create_parameter(
            shape=[self.embedding_size, self.class_num],
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0)
        )
        self.add_parameter("weight", weight)
        t = paddle.zeros([1, ])
        self.register_buffer("t", t, persistable=True)

    def normalize(self, x, axis):
        x_norm = paddle.sqrt(paddle.sum(paddle.square(x), axis=axis, keepdim=True))
        x = paddle.divide(x, x_norm)
        return x

    def forward(self, features, targets):
        """ArcMarginP forward

        Args:
            features (Tensor): [B, F]
            targets (Tensor): [B, ]

        Returns:
            Tensor: [B, C]
        """
        if targets is not None and targets.ndim >= 2:
            targets = targets.reshape([-1, ])
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features, axis=1), F.normalize(self.weight, axis=0))  # [B,C]
        cos_theta = cos_theta.clip(min=-1, max=1)  # for numerical stability [B,C]

        target_logit = cos_theta[paddle.arange(0, features.shape[0]), targets].reshape([-1, 1])  # [B, ]

        sin_theta = paddle.sqrt(1.0 - paddle.square(target_logit))  # [B,C]
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)  # [B,C]
        mask = cos_theta > cos_theta_m
        final_target_logit = paddle.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)  # [B, ]

        hard_example = cos_theta[mask]  # [B,C]
        with paddle.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta = cos_theta.put_along_axis(axis=1, indices=targets.reshape([-1, 1]), values=final_target_logit)
        pred_class_logits = cos_theta * self.scale
        return pred_class_logits


# import torch
# import torch.nn as tnn
# import torch.nn.functional as tF
# import numpy as np


# class ArcfaceT(tnn.Module):
#     """ Additive Angular Margin Loss """
#     def __init__(self, in_feat, num_classes):
#         super().__init__()
#         self.in_feat = in_feat
#         self._num_classes = num_classes
#         self._s = 30
#         self._m = 0.15

#         self.cos_m = math.cos(self._m)
#         self.sin_m = math.sin(self._m)
#         self.threshold = math.cos(math.pi - self._m)
#         self.mm = math.sin(math.pi - self._m) * self._m

#         self.weight = torch.nn.Parameter(torch.ones(num_classes, in_feat)*0.00)
#         self.register_buffer('t', torch.zeros(1))

#     def forward(self, features, targets):
#         # get cos(theta)
#         print(targets.shape)
#         cos_theta = tF.linear(tF.normalize(features), tF.normalize(self.weight))
#         cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

#         target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)
#         print(target_logit.shape)
#         sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
#         cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
#         print(cos_theta.shape)
#         print(cos_theta_m.shape)
#         mask = cos_theta > cos_theta_m
#         final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

#         hard_example = cos_theta[mask]
#         with torch.no_grad():
#             self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
#         cos_theta[mask] = hard_example * (self.t + hard_example)
#         cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
#         pred_class_logits = cos_theta * self._s
#         return pred_class_logits


# if __name__ == "__main__":
#     arcp = ArcMarginP(512, 81313, 0.15, 30)
#     arcp.weight.set_value(np.load("/workspace/hesensen/DOLG_reprod/PaddleClas/dbg.weight.npy"))
#     arct = ArcfaceT(512, 81313)
#     arct.weight.data = torch.from_numpy(np.load("/workspace/hesensen/DOLG_reprod/PaddleClas/dbg.weight.npy").T)
#     arct = arct.cuda()

#     topt = torch.optim.SGD(
#         arct.parameters(),
#         lr=0.005,
#         momentum=0.9,
#         weight_decay=1e-4,
#         nesterov=True
#     )
#     popt = paddle.optimizer.Momentum(
#         0.005,
#         momentum=0.9,
#         parameters=arcp.parameters(),
#         use_nesterov=True,
#         weight_decay=paddle.regularizer.L2Decay(1e-4)
#     )

#     fake_feat = np.load("/workspace/hesensen/DOLG_reprod/PaddleClas/dbg.features.npy")
#     fake_label = np.load("/workspace/hesensen/DOLG_reprod/PaddleClas/dbg.targets.npy").reshape(-1)

#     fake_feat_paddle = paddle.to_tensor(fake_feat)
#     fake_feat_paddle.stop_gradient = False

#     fake_feat_torch = torch.from_numpy(fake_feat).cuda()
#     fake_feat_torch.requres_grad = True

#     fake_label_paddle = paddle.to_tensor(fake_label)
#     fake_label_torch = torch.from_numpy(fake_label).cuda()

#     logits_paddle = arcp(fake_feat_paddle, fake_label_paddle)
#     logits_torch = arct(fake_feat_torch, fake_label_torch)

#     loss_paddle = paddle.nn.functional.cross_entropy(logits_paddle, fake_label_paddle)
#     loss_torch = torch.nn.functional.cross_entropy(logits_torch, fake_label_torch)
#     print(f"{loss_paddle.item():.20f}")
#     print(f"{loss_torch.item():.20f}")

#     loss_paddle.backward()
#     loss_torch.backward()

#     grad_paddle = arcp.weight.grad.T.numpy()
#     grad_torch = arct.weight.grad.cpu().numpy()

#     print(f"{grad_paddle.min():.10f} {grad_paddle.mean():.10f} {grad_paddle.max():.10f}")
#     print(f"{grad_torch.min():.10f} {grad_torch.mean():.10f} {grad_torch.max():.10f}")

#     # grad_error = grad_paddle - grad_torch
#     print(grad_paddle.shape, grad_paddle.dtype)
#     print(grad_torch.shape, grad_torch.dtype)

#     topt.step()
#     popt.step()

#     topt.zero_grad()
#     popt.clear_grad()

#     print(f"{arcp.weight.min().item():.10f} {arcp.weight.mean().item():.10f} {arcp.weight.max().item():.10f}")
#     print(f"{arct.weight.min().item():.10f} {arct.weight.mean().item():.10f} {arct.weight.max().item():.10f}")

#     print(np.allclose(arcp.weight.numpy().T, arct.weight.detach().cpu().numpy()))
