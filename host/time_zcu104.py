# coding: utf-8
# time_zcu104.py

import argparse
import numpy as np
import os
import sys
import torch
import time

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir)))

from net.model import PointNetCls
from model_zcu104 import PointNetClsZCU104

def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bitstream", required=True, type=str)
    parser.add_argument("--num-points", default=1024, type=int)
    parser.add_argument("--runs", default=10, type=int)

    args = parser.parse_args()

    return args

def main():
    # Parse the command-line arguments
    args = parse_command_line()

    # Create a PointNet classification model
    model = PointNetCls(num_classes=40)
    # Create an FPGA model
    model_zcu104 = PointNetClsZCU104(model, args.bitstream, args.num_points)

    model.eval()
    model_zcu104.eval()

    # Test the output
    # Create a random input point cloud
    point_cloud = torch.rand(size=(1, args.num_points, 3))
    out_cpu = model(point_cloud)
    out_zcu104 = model_zcu104(point_cloud)

    print(f"Output (CPU):\n{out_cpu}")
    print(f"Output (FPGA):\n{out_zcu104}")

    # Measure the inference times
    times_cpu = []
    times_zcu104 = []

    for _ in range(args.runs):
        # Create a random input point cloud
        point_cloud = torch.rand(size=(1, args.num_points, 3))

        t0 = time.monotonic()
        model(point_cloud)
        elapsed_cpu = (time.monotonic() - t0) * 1e3

        t0 = time.monotonic()
        model_zcu104(point_cloud)
        elapsed_zcu104 = (time.monotonic() - t0) * 1e3

        times_cpu.append(elapsed_cpu)
        times_zcu104.append(elapsed_zcu104)

    time_avg_cpu = np.mean(times_cpu)
    time_std_cpu = np.std(times_cpu)
    time_avg_zcu104 = np.mean(times_zcu104)
    time_std_zcu104 = np.std(times_zcu104)
    speedup_factor = time_avg_cpu / time_avg_zcu104

    print(f"Inference time (CPU): " \
          f"mean: {time_avg_cpu:.3f}ms, " \
          f"std: {time_std_cpu:.3f}ms")
    print(f"Inference time (FPGA): " \
          f"mean: {time_avg_zcu104:.3f}ms, " \
          f"std: {time_std_zcu104:.3f}ms")
    print(f"Speedup: {speedup_factor:.3f}x")

if __name__ == "__main__":
    main()
