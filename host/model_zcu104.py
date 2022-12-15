# coding: utf-8
# model_zcu104.py

import numpy as np
import os
import pynq
import sys
import torch
import torch.nn

from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir)))

from net.model import PointNetCls

# Split the 64-bit address
def split_address(addr: int) -> Tuple[int, int]:
    mask = (1 << 32) - 1
    return addr & mask, addr >> 32

# Allocate a contiguous buffer for torch.nn.Conv1d (torch.nn.Linear)
def allocate_linear_buffer(in_dims: int, out_dims: int) \
    -> pynq.buffer.PynqBuffer:
    buf_size = in_dims * out_dims + out_dims
    return pynq.allocate(shape=(buf_size,), dtype=np.float32, cacheable=False)

# Allocate a contiguous buffer for a block with torch.nn.Conv1d
# (torch.nn.Linear) and torch.nn.BatchNorm1d
def allocate_block_buffer(in_dims: int, out_dims: int) \
    -> pynq.buffer.PynqBuffer:
    buf_size = 0
    buf_size += in_dims * out_dims + out_dims
    buf_size += out_dims * 3
    return pynq.allocate(shape=(buf_size,), dtype=np.float32, cacheable=False)

# Write the torch.nn.Conv1d parameters to the contiguous buffer
def write_conv1d_params(buf: pynq.buffer.PynqBuffer,
                        layer: torch.nn.Conv1d,
                        offset: int = 0) -> int:
    if layer.kernel_size != (1,):
        raise RuntimeError(f"Kernel size should be 1")

    weight_size = layer.out_channels * layer.in_channels
    bias_size = layer.out_channels

    buf[offset:offset+weight_size] = layer.weight.data.view(-1)
    offset += weight_size
    buf[offset:offset+bias_size] = layer.bias.data.view(-1)
    offset += bias_size

    return offset

# Write the torch.nn.Linear parameters to the contiguous buffer
def write_linear_params(buf: pynq.buffer.PynqBuffer,
                        layer: torch.nn.Linear,
                        offset: int = 0) -> int:
    weight_size = layer.out_features * layer.in_features
    bias_size = layer.out_features

    buf[offset:offset+weight_size] = layer.weight.data.view(-1)
    offset += weight_size
    buf[offset:offset+bias_size] = layer.bias.data.view(-1)
    offset += bias_size

    return offset

# Write the torch.nn.BatchNorm1d parameters to the contiguous buffer
def write_batchnorm1d_params(buf: pynq.buffer.PynqBuffer,
                             layer: torch.nn.BatchNorm1d,
                             offset: int = 0) -> int:
    dims = layer.num_features

    # `scale` is the multiplication of the weight and reciprocal of the
    # standard deviation (to reduce the on-chip memory consumption)
    std_inv = torch.sqrt(layer.running_var.data + layer.eps)
    std_inv = torch.reciprocal(std_inv)
    scale = std_inv * layer.weight.data

    buf[offset:offset+dims] = scale.data.view(-1)
    offset += dims
    buf[offset:offset+dims] = layer.bias.data.view(-1)
    offset += dims
    buf[offset:offset+dims] = layer.running_mean.data.view(-1)
    offset += dims

    return offset

# Write the block (torch.nn.Conv1d and torch.nn.BatchNorm1d) parameters
# to the contiguous buffer
def write_conv_batchnorm1d_params(buf: pynq.buffer.PynqBuffer,
                                  conv: torch.nn.Conv1d,
                                  bn: torch.nn.BatchNorm1d):
    offset = 0
    offset = write_conv1d_params(buf, conv, offset)
    offset = write_batchnorm1d_params(buf, bn, offset)

# Write the block (torch.nn.Linear and torch.nn.BatchNorm1d) parameters
# to the contiguous buffer
def write_linear_batchnorm1d_params(buf: pynq.buffer.PynqBuffer,
                                    linear: torch.nn.Linear,
                                    bn: torch.nn.BatchNorm1d):
    offset = 0
    offset = write_linear_params(buf, linear, offset)
    offset = write_batchnorm1d_params(buf, bn, offset)

class PointNetClsZCU104(torch.nn.Module):
    # Operation modes (refer to hls/src/op_modes.hpp)
    MODE_INIT_WEIGHTS = 100
    MODE_INFERENCE = 101

    def __init__(self, model_cpu: PointNetCls,
                 overlay_path: str, num_points: int):
        super().__init__()

        # Load an overlay
        self.overlay = self.load_overlay(overlay_path)
        # Get the IP core module
        self.net_ip: pynq.DefaultIP = self.overlay.PointNetClsTop
        # Get the control registers of the IP core
        self.registers = self.net_ip.register_map

        # Check the data width of the AXI master interface
        net_ip_params = self.overlay.ip_dict["PointNetClsTop"]["parameters"]
        self.axi_m_addr_width = int(net_ip_params["C_M_AXI_GMEM0_ADDR_WIDTH"])
        self.axi_m_data_width = int(net_ip_params["C_M_AXI_GMEM0_DATA_WIDTH"])

        # Allocate buffers for PointNet feature extraction network
        self.buf_feat_params1 = allocate_block_buffer(3, 64)
        self.buf_feat_params2 = allocate_block_buffer(64, 64)
        self.buf_feat_params3 = allocate_block_buffer(64, 64)
        self.buf_feat_params4 = allocate_block_buffer(64, 128)
        self.buf_feat_params5 = allocate_block_buffer(128, 1024)

        # Allocate buffers for classification network
        self.buf_cls_params1 = allocate_block_buffer(1024, 512)
        self.buf_cls_params2 = allocate_block_buffer(512, 256)
        self.buf_cls_params3 = allocate_linear_buffer(256, 40)

        # Allocate a buffer for point cloud
        self.num_points = num_points
        if self.axi_m_data_width == 32:
            self.buf_point_cloud: pynq.buffer.PynqBuffer = pynq.allocate(
                shape=(self.num_points, 3), dtype=np.float32, cacheable=False)
        elif self.axi_m_data_width == 64:
            self.buf_point_cloud: pynq.buffer.PynqBuffer = pynq.allocate(
                shape=(self.num_points, 4), dtype=np.float32, cacheable=False)
        else:
            raise RuntimeError(f"Unexpected data width for AXI master")

        # Allocate a buffer for output logits
        self.buf_out_logits: pynq.buffer.PynqBuffer = pynq.allocate(
            shape=(40,), dtype=np.float32, cacheable=False)

        # Copy parameters for PointNet feature extraction network
        write_conv_batchnorm1d_params(self.buf_feat_params1,
            model_cpu.feat.conv1, model_cpu.feat.bn1)
        write_conv_batchnorm1d_params(self.buf_feat_params2,
            model_cpu.feat.conv2, model_cpu.feat.bn2)
        write_conv_batchnorm1d_params(self.buf_feat_params3,
            model_cpu.feat.conv3, model_cpu.feat.bn3)
        write_conv_batchnorm1d_params(self.buf_feat_params4,
            model_cpu.feat.conv4, model_cpu.feat.bn4)
        write_conv_batchnorm1d_params(self.buf_feat_params5,
            model_cpu.feat.conv5, model_cpu.feat.bn5)

        # Copy parameters for classification network
        write_linear_batchnorm1d_params(self.buf_cls_params1,
            model_cpu.fc1, model_cpu.bn1)
        write_linear_batchnorm1d_params(self.buf_cls_params2,
            model_cpu.fc2, model_cpu.bn2)
        write_linear_params(self.buf_cls_params3, model_cpu.fc3)

        # Set the physical addresses of the buffers
        self.registers.point_cloud_1, self.registers.point_cloud_2 = \
            split_address(self.buf_point_cloud.device_address)
        self.registers.out_logits_1, self.registers.out_logits_2 = \
            split_address(self.buf_out_logits.device_address)
        self.registers.feat_params1_1, self.registers.feat_params1_2 = \
            split_address(self.buf_feat_params1.device_address)
        self.registers.feat_params2_1, self.registers.feat_params2_2 = \
            split_address(self.buf_feat_params2.device_address)
        self.registers.feat_params3_1, self.registers.feat_params3_2 = \
            split_address(self.buf_feat_params3.device_address)
        self.registers.feat_params4_1, self.registers.feat_params4_2 = \
            split_address(self.buf_feat_params4.device_address)
        self.registers.feat_params5_1, self.registers.feat_params5_2 = \
            split_address(self.buf_feat_params5.device_address)
        self.registers.cls_params1_1, self.registers.cls_params1_2 = \
            split_address(self.buf_cls_params1.device_address)
        self.registers.cls_params2_1, self.registers.cls_params2_2 = \
            split_address(self.buf_cls_params2.device_address)
        self.registers.cls_params3_1, self.registers.cls_params3_2 = \
            split_address(self.buf_cls_params3.device_address)

        # Synchronize the buffers
        self.buf_feat_params1.sync_to_device()
        self.buf_feat_params2.sync_to_device()
        self.buf_feat_params3.sync_to_device()
        self.buf_feat_params4.sync_to_device()
        self.buf_feat_params5.sync_to_device()
        self.buf_cls_params1.sync_to_device()
        self.buf_cls_params2.sync_to_device()
        self.buf_cls_params3.sync_to_device()

        # Initialize the weights (transfer the weights to the on-chip buffers)
        self.registers.op_mode = PointNetClsZCU104.MODE_INIT_WEIGHTS
        self.registers.CTRL.AP_START = 1
        self.wait_for_ip()

    def load_overlay(self, overlay_path):
        overlay = pynq.Overlay(overlay_path)

        if not overlay.is_loaded():
            raise RuntimeError(f"Unable to load overlay: {overlay_path}")

        return overlay

    def wait_for_ip(self):
        while self.registers.CTRL.AP_DONE == 0:
            pass

    def forward(self, x: torch.Tensor):
        # `x` is of size [B, N, 3]
        if x.ndim != 3 or x.shape[2] != 3:
            raise RuntimeError(f"Unexpected shape of the input: {x.shape}")

        batch_size = x.shape[0]
        num_points = x.shape[1]

        # Reallocate the buffer for point cloud if necessary
        if num_points > self.num_points:
            self.num_points = num_points
            self.buf_point_cloud.freebuffer()
            if self.axi_m_data_width == 32:
                self.buf_point_cloud: pynq.buffer.PynqBuffer = pynq.allocate(
                    shape=(self.num_points, 3),
                    dtype=np.float32, cacheable=False)
            elif self.axi_m_data_width == 64:
                self.buf_point_cloud: pynq.buffer.PynqBuffer = pynq.allocate(
                    shape=(self.num_points, 4),
                    dtype=np.float32, cacheable=False)
            else:
                raise RuntimeError(f"Unexpected data width for AXI master")
            self.registers.point_cloud_1, self.registers.point_cloud_2 = \
                split_address(self.buf_point_cloud.device_address)

        # Allocate the Tensor for output
        out = torch.empty(size=(batch_size, 40),
                          dtype=x.dtype, device=x.device)

        # Run the inference
        self.registers.op_mode = PointNetClsZCU104.MODE_INFERENCE
        self.registers.num_points = num_points

        for i in range(batch_size):
            # Copy the input point cloud
            self.buf_point_cloud[:num_points, :3] = x[i].view(-1, 3)
            self.buf_point_cloud.sync_to_device()

            # Run the inference
            self.registers.CTRL.AP_START = 1
            self.wait_for_ip()

            # Copy the output logits
            self.buf_out_logits.sync_from_device()
            out[i, :] = torch.from_numpy(self.buf_out_logits)

        return out
