# coding: utf-8
# modelnet_transforms.py

# Refer to the following implementation:
# https://github.com/yewzijian/RPMNet/blob/master/src/data_loader/transforms.py

import numpy as np

from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from typing import Dict

class Resampler(object):
    def __init__(self, num_points: int):
        self.num_points = num_points

    @staticmethod
    def resample(points: np.ndarray, k: int):
        num_points = points.shape[0]
        if k <= num_points:
            rand_idxs = np.random.choice(num_points, k, replace=False)
            return points[rand_idxs, :]
        elif k == num_points:
            return points
        else:
            rand_idxs = np.concatenate([
                np.random.choice(num_points, num_points, replace=False),
                np.random.choice(num_points, k - num_points, replace=True)])
            return points[rand_idxs, :]

    def __call__(self, sample: Dict):
        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        sample["points"] = self.resample(sample["points"], self.num_points)
        return sample

class RandomJitter(object):
    def __init__(self, scale: float = 0.01, clip: float = 0.05):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts: np.ndarray):
        noise = np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3))
        noise = np.clip(noise, a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise
        return pts

    def __call__(self, sample: Dict):
        sample["points"] = self.jitter(sample["points"])
        return sample

class RandomTransformSE3(object):
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0,
                 random_mag: bool = False):
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.random_mag = random_mag

    def generate_transform(self):
        if self.random_mag:
            attentuation = np.random.random()
            rot_mag = attentuation * self.rot_mag
            trans_mag = attentuation * self.trans_mag
        else:
            rot_mag = self.rot_mag
            trans_mag = self.trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_matrix(rand_rot))
        axis_angle = axis_angle * rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_matrix()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1)
        rand_SE3 = rand_SE3.astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0: np.ndarray, transform: np.ndarray):
        rot = transform[:3, :3]
        trans = transform[:3, 3]
        p1 = p0[:, :3] @ rot.swapaxes(-1, -2) + trans[None, :]
        return p1

    def transform(self, tensor: np.ndarray):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample: Dict):
        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        sample["points"] = self.transform(sample["points"])
        return sample

class ShufflePoints(object):
    def __call__(self, sample: Dict):
        sample["points"] = np.random.permutation(sample["points"])
        return sample

class SetDeterministic(object):
    def __call__(self, sample: Dict):
        sample["deterministic"] = True
        return sample
