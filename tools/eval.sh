#!/usr/bin/env bash

# for single card eval
# python3.7 tools/eval.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/eval.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
tools/eval.py -c ppcls/configs/ImageNet/EfficientNetV2/EfficientNetV2_S.yaml -o Global.pretrained_model="/workspace/hesensen/EfficientNetV2_reprod/tf_EfficientNetV2/efficientnetv2/efficientnetv2-s_1k.h5"
