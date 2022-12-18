#!/bin/bash
# time_zcu104.sh

BASE_DIR=/home/xilinx/advent_2022_point_cloud_classification
DATASET_DIR=$BASE_DIR/data/modelnet40_ply_hdf5_2048
SOURCE_DIR=$BASE_DIR/host
MODEL_DIR=$BASE_DIR/net/outputs
BITSTREAM=$1

if [[ ! -f $BITSTREAM ]]; then
  echo "Specified bitstream does not exist: $BITSTREAM"
  exit 1
fi

python3 $SOURCE_DIR/time_zcu104.py \
  --bitstream $BITSTREAM \
  --num-points 1024 \
  --runs 10
