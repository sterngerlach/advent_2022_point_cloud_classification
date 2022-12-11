# coding: utf-8
# test.py

import argparse
import numpy as np
import os
import sys
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data

from model import PointNetCls
from modelnet_dataset import get_test_set
from utils import setup_seed, setup_device

def parse_command_line():
    parser = argparse.ArgumentParser()

    # Options for datasets
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--category-file", default=None, type=str)
    parser.add_argument("--num-points", default=1024, type=int)

    # Options for testing
    parser.add_argument("--trained-model", required=True, type=str)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", required=True, type=int)

    args = parser.parse_args()

    return args

def test(args: argparse.Namespace,
         device: torch.device,
         model: torch.nn.Module,
         test_loader: torch.utils.data.DataLoader):
    # Transfer the model to the device
    model = model.to(device)

    print(f"Testing PointNet ...")

    model.eval()

    test_loss_total = 0.0
    correct = 0

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            data, target = batch["points"], batch["label"]
            data, target = data.to(device), target.to(device)

            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            loss = F.cross_entropy(out, target)

            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss_total += loss.item() * len(data)

    test_loss_avg = test_loss_total / len(test_loader.dataset)
    test_acc = correct * 1e2 / len(test_loader.dataset)

    print(f"Test result: " \
          f"loss: {test_loss_avg:.6f}, " \
          f"accuracy: {test_acc:.3f}%, " \
          f"correct: {correct}")

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

    # Create a device
    args.device = setup_device(args.device, fallback_to_cpu=True)

    # Create a PointNet classification model
    model = PointNetCls(num_classes=test_set.num_classes)

    # Load the pretrained model
    if not os.path.isfile(args.trained_model):
        raise RuntimeError(f"Trained model not found: {args.trained_model}")
    model.load_state_dict(torch.load(args.trained_model, map_location="cpu"))

    # Run test
    test(args, args.device, model, test_loader)

if __name__ == "__main__":
    main()
