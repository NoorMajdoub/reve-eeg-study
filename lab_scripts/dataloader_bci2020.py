"""BCI Competition 2020 Track 3 loader for REVE, reading the output of
preprocess_bci2020_reve.py. Each sample: one 3 s trial, 64 channels x 600 samples
at 200 Hz, already DC-removed and per-subject z-scored. Labels 0-4 (five words).
The whole split is loaded into RAM (~0.7 GB for train).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class BCI2020Dataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None):
        super().__init__()
        self.mode = mode
        self.X = np.load(os.path.join(path, f"{mode}_X.npy"))
        self.y = np.load(os.path.join(path, f"{mode}_y.npy")).astype(np.int64)
        self.subject = np.load(os.path.join(path, f"{mode}_subject.npy"))
        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)
        assert self.X.shape[1] == self.positions.shape[0], \
            f"{self.X.shape[1]} channels in data vs {self.positions.shape[0]} positions"
        print(f"[BCI2020] {mode}: {len(self.y):,} trials, {len(np.unique(self.subject))} subjects, "
              f"shape {tuple(self.X.shape[1:])}, class counts = {np.bincount(self.y, minlength=5).tolist()}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "sample": torch.from_numpy(np.array(self.X[idx], dtype=np.float32)),
            "label": torch.tensor(self.y[idx]).long(),
            "pos": self.positions,
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([b["sample"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "pos": self.positions.repeat(len(batch), 1, 1),
        }
