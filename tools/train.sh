#!/usr/bin/env bash

# for single card train
# python3.7 tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# 全量数据8卡训练脚本
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" \
tools/train.py -c ./ppcls/configs/Vehicle/DOLG.yaml

# market1501 4卡共32bs对齐
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
tools/train.py -c ./ppcls/configs/Vehicle/DOLG_market1501.yaml

# 小数据单卡对齐脚本
export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=2
python3.7 \
tools/train.py -c ./ppcls/configs/Vehicle/DOLG.yaml

export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=2
python3.7 \
tools/eval.py -c ./ppcls/configs/Vehicle/DOLG.yaml \
-o Global.pretrained_model="pretrained_model/r50_dolg_512"

# export CUDA_VISIBLE_DEVICES=0
# python3.7 tools/train.py -c ./ppcls/configs/Vehicle/DOLG.yaml
