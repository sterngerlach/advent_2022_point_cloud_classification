# coding: utf-8
# model.py

# Refer to the following PointNet implementation:
# https://github.com/fxia22/pointnet.pytorch
# Input transform and feature transform are excluded for simplicity

import torch
import torch.nn
import torch.nn.functional as F

class PointNetFeat(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        # `x` is of size [B, N, 3]
        N = x.shape[1]
        # `x` is of size [B, 3, N]
        x = x.transpose(1, 2)

        # `x` is of size [B, 1024, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # `x` is of size [B, 1024]
        x = torch.max(x, dim=2)[0]

        return x

class PointNetCls(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Feature extraction
        self.feat = PointNetFeat()

        # Classification network
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)

    def forward(self, x):
        # `x` is of size [B, N, 3]
        # `x` is of size [B, 1024]
        x = self.feat(x)

        # `x` is of size [B, `num_classes`]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x
