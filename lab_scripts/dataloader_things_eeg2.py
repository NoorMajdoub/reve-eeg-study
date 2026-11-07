"""THINGS-EEG2 loader for REVE (option A: 200-way image classification), reading
the output of preprocess_things_eeg2_reve.py. Each sample: one 1 s epoch,
17 channels x 200 samples at 200 Hz (exactly one REVE patch per channel).
The whole split is loaded into RAM (train ~1.5 GB).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class ThingsEEG2Dataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None):
        super().__init__()
        self.mode = mode
        self.X = np.load(os.path.join(path, f"{mode}_X.npy"))
        self.y = np.load(os.path.join(path, f"{mode}_y.npy")).astype(np.int64)
        self.subject = np.load(os.path.join(path, f"{mode}_subject.npy"))
        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)
        assert self.X.shape[1] == self.positions.shape[0], \
            f"{self.X.shape[1]} channels in data vs {self.positions.shape[0]} positions"
        counts = np.bincount(self.y)
        print(f"[THINGS-EEG2] {mode}: {len(self.y):,} trials, subjects {sorted(set(self.subject.tolist()))}, "
              f"shape {tuple(self.X.shape[1:])}, {int((counts > 0).sum())} classes, "
              f"{counts.min()}-{counts.max()} trials/class")

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
