import json
import h5py
import torch
from torch.utils.data import Dataset

class SimpleHDF5Dataset(Dataset):
    def __init__(self, h5_path, map_label_path=None, augmentation=False,noise_std=0.005):
        self.h5file = h5py.File(h5_path, "r")
        self.keys = list(self.h5file.keys())
        self.labels = [self.h5file[k]["label"][()].decode("utf-8") for k in self.keys]
        self.video_names = [self.h5file[k]["video_name"][()].decode("utf-8") for k in self.keys]
        self.noise_std = noise_std
        self.augmentation = augmentation
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

        data = torch.tensor(entry["data"][()], dtype=torch.float32).permute(0, 2, 1)-0.5  # [T, V, 2]
        data += (torch.randn_like(data)*self.noise_std) if self.augmentation else 0

        name = self.labels[idx]
        label = self.map_labels["id_to_label"][name]
        video_name = self.video_names[idx]
        return data, name, label, video_name
