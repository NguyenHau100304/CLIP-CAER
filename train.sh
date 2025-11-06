#!/bin/bash
# Chạy từ thư mục RAER/

python CLIP-CAER/main.py \
    --mode train \
    --exper-name test \
    --gpu 0 \
    --epochs 20 \
    --batch-size 8 \
    --lr 0.01 \
    --lr-image-encoder 0.00001 \
    --lr-prompt-learner 0.001 \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --milestones 10 15 \
    --gamma 0.1 \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --print-freq 10 \
    \
    --root-dir ./RAER \
    --train-annotation annotation/train.txt \
    --test-annotation annotation/test.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face ./RAER/bounding_box/face.json \
    --bounding-box-body ./RAER/bounding_box/body.json \
    \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True