# coding: utf-8
# modelnet_dataset.py

# Refer to the following implementation:
# https://github.com/yewzijian/RPMNet/blob/master/src/data_loader/datasets.py

# Download the ModelNet40 dataset from the following URL:
# https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

import h5py
import numpy as np
import os
import sys
import torch.utils.data
import torchvision

from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir)))

from net import modelnet_transforms

def load_modelnet_category(file_name: str):
    # Check if the category file exists
    file_name = os.path.expanduser(file_name)
    if not os.path.exists(file_name) or not os.path.isfile(file_name):
        raise RuntimeError(f"Category file not found: {file_name}")

    # Load the categories to use
    with open(file_name, "r") as f:
        categories = [x.strip() for x in f]

    return categories

def get_train_set(dataset_dir: str, category_file_name: str, num_points: int,
                  rot_mag: float, trans_mag: float, random_mag: bool,
                  jitter_scale: float, jitter_clip: float):
    if category_file_name is not None:
        categories = load_modelnet_category(category_file_name)
        categories.sort()
    else:
        categories = None

    transforms = torchvision.transforms.Compose([
        modelnet_transforms.Resampler(num_points),
        modelnet_transforms.RandomTransformSE3(rot_mag, trans_mag, random_mag),
        modelnet_transforms.RandomJitter(jitter_scale, jitter_clip),
        modelnet_transforms.ShufflePoints()])
    train_set = ModelNetHdf(dataset_dir, "train", categories, transforms)

    return train_set

def get_test_set(dataset_dir: str, category_file_name: str, num_points: int):
    if category_file_name is not None:
        categories = load_modelnet_category(category_file_name)
        categories.sort()
    else:
        categories = None

    transforms = torchvision.transforms.Compose([
        modelnet_transforms.SetDeterministic(),
        modelnet_transforms.Resampler(num_points)])
    test_set = ModelNetHdf(dataset_dir, "test", categories, transforms)

    return test_set

class ModelNetHdf(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str, subset: str,
                 categories: List[str], transform=None):
        # Check the subset name
        if subset not in ["train", "test"]:
            raise RuntimeError(f"Invalid subset: {subset} " \
                               f"(expected: `train` or `test`)")

        # Check if the dataset path exists
        dataset_dir = os.path.expanduser(dataset_dir)
        if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
            raise RuntimeError(f"Dataset not found: {dataset_dir}")

        # Load the shape names
        shape_names_path = os.path.join(dataset_dir, "shape_names.txt")
        with open(shape_names_path, "r") as f:
            self.classes = [x.strip() for x in f]
            self.category_to_idx = {
                x[1]: x[0] for x in enumerate(self.classes) }
            self.idx_to_category = {
                x[0]: x[1] for x in enumerate(self.classes) }

        # Load the HDF5 file names
        h5_file_names_path = os.path.join(dataset_dir, f"{subset}_files.txt")
        with open(h5_file_names_path, "r") as f:
            h5_file_names = [x.strip() for x in f]
            h5_file_names = [x.replace("data/modelnet40_ply_hdf5_2048/", "")
                for x in h5_file_names]
            h5_file_names = [os.path.join(dataset_dir, x)
                for x in h5_file_names]

        # Load the categories to use
        if categories is not None:
            self.classes = categories
            categories_idx = [self.category_to_idx[x] for x in categories]
        else:
            categories_idx = [self.category_to_idx[x] for x in self.classes]

        # Load the HDF5 files
        self.data, self.labels = self.read_h5_files(
            h5_file_names, categories_idx)

        self.transform = transform

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = { "points": self.data[idx, :, :],
                   "label": self.labels[idx],
                   "idx": np.array(idx, dtype=np.int64) }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def read_h5_files(self, h5_file_names: List[str],
                      categories_idx: List[int]):
        all_data = []
        all_labels = []

        for file_name in h5_file_names:
            f = h5py.File(file_name, "r")
            data = f["data"][:].astype(np.float32)
            normal = f["normal"][:].astype(np.float32)
            labels = f["label"][:].flatten().astype(np.int64)

            # Filter out the unwanted categories
            mask = np.isin(labels, categories_idx).flatten()
            data = data[mask, ...]
            normal = normal[mask, ...]
            labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return all_data, all_labels
