# coding: utf-8
# test_zcu104.py

import argparse
import numpy as np
import os
import sys
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir)))

from net.model import PointNetCls
from net.modelnet_dataset import get_test_set
from net.utils import setup_seed
from model_zcu104 import PointNetClsZCU104

def parse_command_line():
    parser = argparse.ArgumentParser()

    # Options for datasets
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--category-file", default=None, type=str)
    parser.add_argument("--num-points", default=1024, type=int)

    # Options for testing
    parser.add_argument("--bitstream", required=True, type=str)
    parser.add_argument("--trained-model", required=True, type=str)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--seed", required=True, type=int)

    args = parser.parse_args()

    return args

def test(args: argparse.Namespace,
         model: torch.nn.Module,
         model_zcu104: torch.nn.Module,
         test_loader: torch.utils.data.DataLoader):
    print(f"Testing PointNet ...")

    # model.eval()
    model_zcu104.eval()

    # test_loss_total = 0.0
    # correct = 0
    test_loss_total_zcu104 = 0.0
    correct_zcu104 = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 5 == 0:
                print(f"Testing batch {i} ...")

            data, target = batch["points"], batch["label"]

            # out = model(data)
            # pred = out.argmax(dim=1, keepdim=True)
            # loss = F.cross_entropy(out, target)
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # test_loss_total += loss.item() * len(data)

            out_zcu104 = model_zcu104(data)
            pred_zcu104 = out_zcu104.argmax(dim=1, keepdim=True)
            loss_zcu104 = F.cross_entropy(out_zcu104, target)
            correct_zcu104 += pred_zcu104.eq(
                target.view_as(pred_zcu104)).sum().item()
            test_loss_total_zcu104 += loss_zcu104.item() * len(data)

    # test_loss_avg = test_loss_total / len(test_loader.dataset)
    # test_acc = correct * 1e2 / len(test_loader.dataset)
    test_loss_avg_zcu104 = test_loss_total_zcu104 / len(test_loader.dataset)
    test_acc_zcu104 = correct_zcu104 * 1e2 / len(test_loader.dataset)

    # print(f"Test result (CPU): " \
    #       f"loss: {test_loss_avg:.6f}, " \
    #       f"accuracy: {test_acc:.3f}%, " \
    #       f"correct: {correct}")
    print(f"Test result (FPGA): " \
          f"loss: {test_loss_avg_zcu104:.6f}, " \
          f"accuracy: {test_acc_zcu104:.3f}%, " \
          f"correct: {correct_zcu104}, " \
          f"total: {len(test_loader.dataset)}")

def main():
    # Parse the command-line arguments
    args = parse_command_line()

    # Set the random seed
    setup_seed(args.seed, cudnn_deterministic=True)

    # Create a DataLoader
    test_set = get_test_set(args.dataset_dir,
        args.category_file, args.num_points)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    # Create a PointNet classification model
    model = PointNetCls(num_classes=test_set.num_classes)

    # Load the pretrained model
    if not os.path.isfile(args.trained_model):
        raise RuntimeError(f"Trained model not found: {args.trained_model}")
    model.load_state_dict(torch.load(args.trained_model, map_location="cpu"))

    # Load the FPGA model
    model_zcu104 = PointNetClsZCU104(model, args.bitstream, args.num_points)

    # Run test
    test(args, model, model_zcu104, test_loader)

if __name__ == "__main__":
    main()
