#!/bin/bash

python main.py \
    --mode eval \
    --gpu 2 \
    --exper-name test_eval \
    --eval-checkpoint /media/D/zlm/code/CLIP_CAER/outputs_1/test-[07-09]-[22:24]/model_best.pth \
    --root-dir /media/F/FERDataset/AER-DB \
    --test-annotation RAER/test_abs.txt \
    --clip-path /media/D/zlm/code/single_four/models/ViT-B-32.pt \
    --bounding-box-face /media/F/FERDataset/AER-DB/RAER/bounding_box/face_abs.json \
    --bounding-box-body /media/F/FERDataset/AER-DB/RAER/bounding_box/body_abs.json \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42
