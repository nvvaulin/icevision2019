#!/bin/bash

experiment=${1:-init}

mkdir -p class_ckpts/$experiment

cp train.sh class_ckpts/$experiment/

python3 train_classifier.py \
    --experiment=$experiment \
    --label_smoothing=0 \
    --init_lr=0.01 \
    --n_cycles=3 \
    --batch_size=128 \
    --amp=O2 \
    --n_workers=10 \
    --train_folder=classification_data/train \
    --val_folder=classification_data/val |& tee class_ckpts/$experiment/log.txt
