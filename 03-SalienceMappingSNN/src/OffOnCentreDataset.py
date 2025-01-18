import torch
import os
from torch.utils.data import Dataset
import random

class OffOnCentreDataset(Dataset):
    def __init__(self, root_dir, device):
        super().__init__()
        self.root_dir = root_dir
        self.device = device

        # list of all scenes
        self.scenes = []
        for folder_name in os.listdir(root_dir):
            full_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(full_path):
                # e.g. "DataSet/scene1" ...
                self.scenes.append(full_path)

        self.scenes.sort()

    def __len__(self):
        return len(self.scenes)

    def _read_scene_data(self, file_path):
        spike_tensor = torch.load(file_path, map_location=self.device, weights_only=True)
        return spike_tensor

    def __getitem__(self, idx):

        scene_folder = self.scenes[idx]


        # Path
        path_input_spk = os.path.join(scene_folder, "input_spk.pt")
        path_train_spk = os.path.join(scene_folder, "train_spk.pt")

        # Load spike data

        input_spk = self._read_scene_data(path_input_spk)
        train_spk = self._read_scene_data(path_train_spk)

        return {"input": input_spk,
                "label": train_spk,
                "scene_name": os.path.basename(scene_folder)
                }

    def split_dataset(self, split_ratio):
        all_indices = list(range(len(self.scenes)))
        random.shuffle(all_indices)

        train_size = int(len(all_indices) * split_ratio[0])
        val_size = int(len(all_indices)*split_ratio[1])

        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size+val_size]
        test_indices = all_indices[train_size + val_size:]
        return train_indices, val_indices, test_indices

    def save_splits(self, file_path, train_indices, val_indices, test_indices):
        """
        Saves the split indices to a .txt file in a simple line-based format:
            train:1,2,3
            val:4,5
            test:6,7,8
        """
        with open(file_path, 'w') as f:
            # Train line
            f.write("train:" + ",".join(str(i) for i in train_indices) + "\n")
            # Val line
            f.write("val:" + ",".join(str(i) for i in val_indices) + "\n")
            # Test line
            f.write("test:" + ",".join(str(i) for i in test_indices) + "\n")

    def load_splits(self, file_path):
        """
        Loads the train/val/test indices from the .txt file created by save_splits.
        Returns train_indices, val_indices, test_indices as lists of ints.
        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Parse train
        train_line = lines[0].split(":")[1] if len(lines) > 0 else ""
        train_indices = [int(x) for x in train_line.split(",")] if train_line else []

        # Parse val
        val_line = lines[1].split(":")[1] if len(lines) > 1 else ""
        val_indices = [int(x) for x in val_line.split(",")] if val_line else []

        # Parse test
        test_line = lines[2].split(":")[1] if len(lines) > 2 else ""
        test_indices = [int(x) for x in test_line.split(",")] if test_line else []

        return train_indices, val_indices, test_indices