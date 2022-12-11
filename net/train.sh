#!/bin/bash
# train.sh

BASE_DIR=$HOME/advent-2022-pointnet
DATASET_DIR=$BASE_DIR/data/modelnet40_ply_hdf5_2048
SOURCE_DIR=$BASE_DIR/net
OUT_DIR=$SOURCE_DIR/outputs

python3 $SOURCE_DIR/train.py \
  --out-dir $OUT_DIR \
  --name modelnet40-all \
  --dataset-dir $DATASET_DIR \
  --category-file $SOURCE_DIR/datasets/modelnet40_all.txt \
  --num-points 1024 \
  --batch-size 32 \
  --epochs 100 \
  --device cuda:0 \
  --seed 42

python3 $SOURCE_DIR/train.py \
  --out-dir $OUT_DIR \
  --name modelnet40-half1 \
  --dataset-dir $DATASET_DIR \
  --category-file $SOURCE_DIR/datasets/modelnet40_half1.txt \
  --num-points 1024 \
  --batch-size 32 \
  --epochs 100 \
  --device cuda:0 \
  --seed 42
