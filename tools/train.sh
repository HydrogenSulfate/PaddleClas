#!/usr/bin/env bash

# for single card train
# python3.7 tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# 1卡调试
# python3.7 tools/train.py -c ppcls/configs/ImageNet/EfficientNetV2/EfficientNetV2_S.yaml

# 8卡训练
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c ppcls/configs/ImageNet/EfficientNetV2/EfficientNetV2_S.yaml
