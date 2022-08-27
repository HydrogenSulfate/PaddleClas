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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle
from ppcls.engine.engine import Engine
from ppcls.utils import config
import numpy as np
import cv2


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool=False,
                      colormap: int=cv2.COLORMAP_JET,
                      original_cmp: bool=False) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format

    Args:
        img (np.ndarray): image of shape [h,w,c], value in [0, 1.0]
        mask (np.ndarray): grad mask
        use_rgb (bool, optional): convert mask to (R,G,B), (B,G,R) by default. Defaults to False.
        colormap (int, optional): pseudo color mapping method for visualization. Defaults to cv2.COLORMAP_JET.
        original_cmp (bool, optional): whether stack img left of returned image for better comparison. Defaults to False.

    Returns:
        np.ndarray: mixed image of original and mask with shape [h,w,3], value in [0, 255], uint8 format
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    if heatmap.shape != img.shape:
        heatmap = cv2.resize(
            heatmap, img.shape[0:2], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap.clip(0, 1)
    cam = heatmap + img
    cam = cam / np.max(cam)
    ret = np.uint8(255 * cam)
    if original_cmp:
        ret = np.concatenate(
            [np.uint8(255 * img), ret], axis=1)  # concat on width([img|ret])
    ret = cv2.cvtColor(ret, cv2.COLOR_RGB2BGR)
    return ret


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    engine = Engine(config, mode="eval")

    print_batch_step = engine.config["Global"]["print_batch_step"]

    gradcam_config = engine.config["Global"].get("GradCam", {})

    label_index = gradcam_config.get(
        "target_label", None)  # None for maximum logit index(default)

    output_dir = engine.config["Global"].get("output_dir", "./output")
    output_dir = os.path.join(output_dir, engine.config["Arch"]["name"],
                              "grad_cam")
    os.makedirs(output_dir, exist_ok=True)
    target_pattern = engine.config["Arch"].get("return_patterns", None)
    assert isinstance(target_pattern, list) and isinstance(
        target_pattern[0], str
    ), f"target_pattern must specified to an string, but got {target_pattern}"
    target_pattern = target_pattern[0]

    norm_mean = engine.config["DataLoader"]["Eval"]["dataset"][
        "transform_ops"][-1]["NormalizeImage"]["mean"]
    norm_std = engine.config["DataLoader"]["Eval"]["dataset"]["transform_ops"][
        -1]["NormalizeImage"]["std"]
    engine.model.eval()

    for iter_id, batch in enumerate(engine.eval_dataloader):
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0])
        input_channels = batch[0].shape[1]
        # image input
        if engine.amp and engine.amp_eval:
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=engine.amp_level):
                out = engine.model(batch[0])
        else:
            out = engine.model(batch[0])
        target_feature = out[target_pattern]
        if isinstance(out, dict) and "Student" in out:
            out = out["Student"]
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]
        # step1. get logit value at label_index
        if label_index is None:
            target_index = paddle.argmax(
                out, axis=1, keepdim=True)[0, 0].item()  # [batchsize, ]
            target_logit = out[:, target_index]  # [batchsize, ]
        else:
            target_index = label_index
            target_logit = out[:, target_index]  # [batchsize, ]

        # step2. compute differentiation: d(logit[label_index])/d(target_feature)
        grad = paddle.grad(
            outputs=[target_logit],
            inputs=[target_feature],
            create_graph=False,
            retain_graph=False)[0]
        alpha = paddle.mean(grad, axis=(2, 3), keepdim=True)  # [b,c,1,1]
        with paddle.no_grad():
            grad_cam = paddle.sum(
                paddle.nn.functional.relu(alpha * target_feature),
                axis=1,
                keepdim=True)  # [b,1,h,w]

            # get RGB image(preprocessed) and grad_cam map.
            ori_image = (batch[0]) * paddle.to_tensor(norm_std).reshape(
                [1, input_channels, 1, 1]) + paddle.to_tensor(
                    norm_mean).reshape([1, input_channels, 1, 1])

            grad_cam_gray_scale = (grad_cam - grad_cam.min()) / (
                grad_cam.max() - grad_cam.min())

            mixed_image = show_cam_on_image(
                img=ori_image[0].numpy().transpose((1, 2, 0)).clip(0, 1),
                mask=grad_cam_gray_scale[0].numpy().transpose(
                    (1, 2, 0)).clip(0, 1),
                use_rgb=True,
                colormap=cv2.COLORMAP_JET,
                original_cmp=True)  # [h,w(2w),c] in uint8
            cv2.imwrite(
                f"{output_dir}/iter_id_{iter_id}_label_{target_index}.jpg",
                mixed_image)
            if iter_id >= 20:
                break
