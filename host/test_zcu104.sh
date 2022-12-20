#!/bin/bash
# test_zcu104.sh

BASE_DIR=/home/xilinx/advent_2022_point_cloud_classification
DATASET_DIR=$BASE_DIR/data/modelnet40_ply_hdf5_2048
SOURCE_DIR=$BASE_DIR/host
MODEL_DIR=$BASE_DIR/net/outputs
BITSTREAM=$1

if [[ ! -f $BITSTREAM ]]; then
  echo "Specified bitstream does not exist: $BITSTREAM"
  exit 1
fi

python3 $SOURCE_DIR/test_zcu104.py \
  --dataset-dir $DATASET_DIR \
  --category-file $BASE_DIR/net/datasets/modelnet40_all.txt \
  --num-points 1024 \
  --bitstream $BITSTREAM \
  --trained-model $MODEL_DIR/2022-12-11-21-41-35-modelnet40-all/best-model.pth \
  --batch-size 1 \
  --seed 42
