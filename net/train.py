# coding: utf-8
# train.py

import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import tqdm

from model import PointNetCls
from modelnet_dataset import get_train_set, get_test_set
from utils import setup_seed, setup_device, \
                  setup_output_directory, setup_logger

_logger = logging.getLogger(__file__)

def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--out-dir", default=None, type=str)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--dry-run", action="store_true")

    # Options for datasets
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--category-file", default=None, type=str)
    parser.add_argument("--num-points", default=1024, type=int)

    # Options for training
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", required=True, type=int)

    args = parser.parse_args()

    return args

def train_one_epoch(args: argparse.Namespace,
                    device: torch.device,
                    epoch: int,
                    model: torch.nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer):
    model.train()

    train_loss_total = 0.0
    correct = 0

    for _, batch in enumerate(tqdm.tqdm(train_loader)):
        data, target = batch["points"], batch["label"]
        data, target = data.to(device), target.to(device)

        out = model(data)
        pred = out.argmax(dim=1, keepdim=True)
        loss = F.cross_entropy(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss_total += loss.item() * len(data)

    train_loss_avg = train_loss_total / len(train_loader.dataset)
    train_acc = correct * 1e2 / len(train_loader.dataset)

    _logger.info(f"Train epoch: {epoch}, " \
                 f"loss: {train_loss_avg:.6f}, " \
                 f"accuracy: {train_acc:.3f}%, " \
                 f"correct: {correct}")

def test_one_epoch(args: argparse.Namespace,
                   device: torch.device,
                   epoch: int,
                   model: torch.nn.Module,
                   val_loader: torch.utils.data.DataLoader):
    model.eval()

    val_loss_total = 0.0
    correct = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm.tqdm(val_loader)):
            data, target = batch["points"], batch["label"]
            data, target = data.to(device), target.to(device)

            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            loss = F.cross_entropy(out, target)

            correct += pred.eq(target.view_as(pred)).sum().item()
            val_loss_total += loss.item() * len(data)

    val_loss_avg = val_loss_total / len(val_loader.dataset)
    val_acc = correct * 1e2 / len(val_loader.dataset)

    _logger.info(f"Test epoch: {epoch}, " \
                 f"loss: {val_loss_avg:.6f}, " \
                 f"accuracy: {val_acc:.3f}%, " \
                 f"correct: {correct}")

    return val_acc

def train(args: argparse.Namespace,
          device: torch.device,
          model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Create an Adam optimizer
    optimizer = torch.optim.Adam(learnable_params, lr=1e-3)
    # Create a StepLR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.8)

    best_test_score = -np.inf

    # Transfer the model to the device
    model = model.to(device)

    # Start training and validation
    _logger.info("Training PointNet ...")

    for epoch in range(args.epochs):
        train_one_epoch(args, device, epoch, model, train_loader, optimizer)
        test_score = test_one_epoch(args, device, epoch, model, val_loader)

        scheduler.step()

        # Save the best model for inference
        if best_test_score < test_score:
            best_test_score = test_score
            _logger.info(f"Current epoch is the best: {best_test_score:.6f}")

            if not args.dry_run:
                snap = { "epoch": epoch,
                         "best_score": best_test_score,
                         "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict() }

                best_snap_path = os.path.join(args.out_dir, "best-snap.pth")
                best_model_path = os.path.join(args.out_dir, "best-model.pth")
                torch.save(snap, best_snap_path)
                torch.save(model.state_dict(), best_model_path)

        # Save the last model to resume
        if not args.dry_run:
            snap = { "epoch": epoch,
                     "best_score": best_test_score,
                     "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict() }

            last_snap_path = os.path.join(args.out_dir, "last-snap.pth")
            last_model_path = os.path.join(args.out_dir, "last-model.pth")
            torch.save(snap, last_snap_path)
            torch.save(model.state_dict(), last_model_path)

    _logger.info(f"Done")

def main():
    # Parse the command-line arguments
    args = parse_command_line()

    # Setup the output directory
    if not args.dry_run and args.out_dir is None:
        raise RuntimeError(f"`out-dir` argument is not set")
    if not args.dry_run and args.name is None:
        raise RuntimeError(f"`name` argument is not set")
    if not args.dry_run:
        args.out_dir = setup_output_directory(args.out_dir, args.name)

    # Setup the logger
    setup_logger(args.out_dir, args.dry_run)
    # Set the random seed
    setup_seed(args.seed, cudnn_deterministic=True)

    # Log the command-line arguments
    _logger.info(args)

    # Create a DataLoader
    train_set = get_train_set(args.dataset_dir,
        args.category_file, args.num_points,
        rot_mag=45.0, trans_mag=0.5, random_mag=True,
        jitter_scale=0.01, jitter_clip=0.05)
    val_set = get_test_set(args.dataset_dir,
        args.category_file, args.num_points)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False)

    # Create a device
    args.device = setup_device(args.device, fallback_to_cpu=True)

    # Create a PointNet classification model
    model = PointNetCls(num_classes=train_set.num_classes)

    # Run training and validation
    train(args, args.device, model, train_loader, val_loader)

if __name__ == "__main__":
    main()
