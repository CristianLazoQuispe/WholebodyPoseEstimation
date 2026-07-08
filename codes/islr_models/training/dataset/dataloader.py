import json
import h5py
import torch
from torch.utils.data import Dataset
from .augmentations_spatial import Augmentation
import random
import os
import sys
import cv2
import copy 
#get local path of the current file
current_file_path = os.path.dirname(os.path.abspath(__file__))
# Add the src directory to the system path
src_path = os.path.join(current_file_path, '../../../../')
# Add the src directory to the system path
if src_path not in sys.path:
    sys.path.append(src_path)
from src.wholebodypose.filters.filtering_nlms import FilteringNLMS

class SimpleHDF5Dataset(Dataset):
    def __init__(self, h5_path, map_label_path=None, augmentation=False,noise_std=0.005,device="cuda",use_filtering=False):
        self.h5file = h5py.File(h5_path, "r")
        self.keys = list(self.h5file.keys())
        self.labels = [self.h5file[k]["label"][()].decode("utf-8") for k in self.keys]
        self.video_names = [self.h5file[k]["video_name"][()].decode("utf-8") for k in self.keys]
        self.noise_std = noise_std
        self.augmentation = augmentation
        self.transform = Augmentation(device=device) if augmentation else None
        self.filter    = FilteringNLMS(num_keypoints=135)  if use_filtering else None# Initialize the filter with a default number of keypoints
        if map_label_path is None:
            self.map_labels = {str(i): str(i) for i in range(len(set(self.labels)))}
        else:
            with open(map_label_path, 'r') as f:
                self.map_labels = json.load(f)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        entry = self.h5file[key]
        data = entry["data"][()]
        # if score exists, use it to filter the data
        if "score" in entry:
            score = entry["score"][()]
            if self.filter is not None:
                data = self.filter.apply(data,score)
        data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)-0.5  # [T, V, 2]

        # Undetected keypoints are the sentinel (0, 0) in the raw data, i.e. (-0.5, -0.5) after the shift.
        # Track them so noise/augmentation below don't scatter them to a fixed phantom point (which then
        # shows up as a cluster in plots and can't be masked by a near-origin test anymore).
        undetected = (data[..., 0] == -0.5) & (data[..., 1] == -0.5)  # [T, V]

        if random.random() < 0.5:
            if self.transform:
                data += (torch.randn_like(data)*self.noise_std)
                n_spatial_func  = len(self.transform.spatial_augmentations.keys())
                data = self.transform.apply_spatial(random.randint(0, n_spatial_func - 1),data)

        # Re-pin undetected keypoints to the sentinel so they stay maskable end-to-end.
        data[undetected] = -0.5

        name = self.labels[idx]
        label = self.map_labels["id_to_label"][name]
        video_name = self.video_names[idx]
        #if idx==0:
        #    print(f"Data min: {data.min()}, max: {data.max()}, mean: {data.mean()}, std: {data.std()}")
        return data, name, label, video_name
