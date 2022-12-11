# coding: utf-8
# utils.py

import logging
import numpy as np
import os
import random
import sys
import torch

from datetime import datetime

_logger = logging.getLogger(__file__)

def setup_seed(seed: int, cudnn_deterministic: bool = False):
    """Set the random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic

def setup_device(device: str, fallback_to_cpu: bool = True):
    if not torch.cuda.is_available():
        if not fallback_to_cpu:
            raise RuntimeError(f"GPU is not available " \
                               f"(`torch.cuda.is_available()` is False)")
        device = "cpu"

    device = torch.device(device)
    _logger.info(f"Device: {device.type}")

    return device

def setup_output_directory(out_dir: str, exp_name: str):
    """Setup the output directory"""
    if not out_dir or out_dir.isspace():
        raise ValueError(f"`out_dir` is empty")
    if not exp_name or exp_name.isspace():
        raise ValueError(f"`exp_name` is empty")

    date_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dir_name = f"{date_time_str}-{exp_name}"
    out_dir = os.path.join(out_dir, dir_name)
    os.makedirs(out_dir, mode=0o775, exist_ok=True)

    return out_dir

def setup_logger(out_dir: str, dry_run: bool = False):
    """Setup the logger"""
    logger_stderr_handler = logging.StreamHandler(sys.stderr)
    logger_handlers = [logger_stderr_handler]

    if not dry_run:
        log_file_path = os.path.join(out_dir, "out.log")
        logger_file_handler = logging.FileHandler(
            log_file_path, mode="w", encoding="utf-8")
        logger_handlers.append(logger_file_handler)

    log_format = "[%(asctime)s] (%(filename)s:%(lineno)d) " \
                 "%(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H-%M-%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format, datefmt=date_format,
                        handlers=logger_handlers, force=True)
