"""CAUEEG-Dementia loader for REVE, reading the output of preprocess_caueeg_reve.py.

Each sample is one 10 s window, 19 channels x 2000 samples at 200 Hz, already
per-recording z-scored and clipped in preprocessing. Labels: Normal=0, MCI=1,
Dementia=2 (inherited from the recording).
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class CAUEEGDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None):
        super().__init__()
        self.path = path
        self.mode = mode

        with open(os.path.join(path, "split.json")) as f:
            split = json.load(f)

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)

        self.seq_files, index, labels = [], [], []
        for serial in split[mode]:
            y = np.load(os.path.join(path, "labels", f"{serial}.npy"))
            fi = len(self.seq_files)
            self.seq_files.append(os.path.join(path, "seq", f"{serial}.npy"))
            index.extend((fi, i) for i in range(len(y)))
            labels.append(y)
        self.index = index
        self.labels = np.concatenate(labels).astype(np.int64)

        counts = np.bincount(self.labels, minlength=3).tolist()
        print(f"[CAUEEG] {mode}: {len(split[mode])} recordings, {len(self.index):,} windows, "
              f"class counts Normal/MCI/Dementia = {counts}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, i = self.index[idx]
        x = np.array(np.load(self.seq_files[fi], mmap_mode="r")[i], dtype=np.float32)
        return {
            "sample": torch.from_numpy(x),
            "label": torch.tensor(self.labels[idx]).long(),
            "pos": self.positions,
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([b["sample"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "pos": self.positions.repeat(len(batch), 1, 1),
        }
