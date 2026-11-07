"""KaraOne imagined-speech dataset loader for REVE."""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal

from downstream_tasks.position_utils import load_positions

SRC_FREQ = 1000
TARGET_FREQ = 200
DURATION = 5  # seconds
N_TARGET = TARGET_FREQ * DURATION


class KaraOneDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None):
        super().__init__()
        self.path = path
        self.mode = mode
        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)

        self.dir = os.path.join(self.path, mode)
        self.files = sorted(
            os.path.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith(".pkl")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            d = pickle.load(f)
        data = d["X"]  # (60, ~4960-4970) at 1000Hz, raw volts -- trial length varies slightly
        label = d["y"]

        data = signal.resample(data, N_TARGET, axis=1)

        mu = data.mean(axis=1, keepdims=True)
        sigma = data.std(axis=1, keepdims=True)
        data = (data - mu) / (sigma + 1e-8)

        return {
            "sample": torch.from_numpy(data).float(),
            "label": torch.tensor(label).long(),
            "pos": self.positions,
        }

    def collate(self, batch):
        x_data = np.array([x["sample"].numpy() for x in batch])
        y_label = np.array([x["label"] for x in batch])
        N = len(batch)
        positions = self.positions.repeat(N, 1, 1)
        return {
            "sample": torch.from_numpy(x_data).float(),
            "label": torch.from_numpy(y_label).long(),
            "pos": positions,
        }