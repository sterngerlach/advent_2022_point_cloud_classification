#!/bin/bash
# test.sh

BASE_DIR=$HOME/advent-2022-pointnet
DATASET_DIR=$BASE_DIR/data/modelnet40_ply_hdf5_2048
SOURCE_DIR=$BASE_DIR/net
MODEL_DIR=$SOURCE_DIR/outputs

python3 $SOURCE_DIR/test.py \
  --dataset-dir $DATASET_DIR \
  --category-file $SOURCE_DIR/datasets/modelnet40_all.txt \
  --num-points 1024 \
  --trained-model $MODEL_DIR/2022-12-11-21-41-35-modelnet40-all/best-model.pth \
  --batch-size 32 \
  --device cuda:0 \
  --seed 42

python3 $SOURCE_DIR/test.py \
  --dataset-dir $DATASET_DIR \
  --category-file $SOURCE_DIR/datasets/modelnet40_half1.txt \
  --num-points 1024 \
  --trained-model $MODEL_DIR/2022-12-11-21-53-29-modelnet40-half1/best-model.pth \
  --batch-size 32 \
  --device cuda:0 \
  --seed 42
