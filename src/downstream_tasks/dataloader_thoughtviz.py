"""ThoughtViz (image/object-category) dataset loader for REVE."""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal

from downstream_tasks.position_utils import load_positions

SRC_FREQ = 128
TARGET_FREQ = 200
REAL_DURATION = 0.25  # seconds (32 samples @ 128Hz) -- the actual recorded signal
N_REAL = int(round(TARGET_FREQ * REAL_DURATION))  # 50

# REVE's encoder has a hard-coded patch_size=200 (baked into the pretrained
# checkpoint's patch-embedding weights) across every model size -- it cannot
# process fewer than 200 timepoints per sample, period. ThoughtViz's real
# signal is only 50 timepoints (0.25s) after resampling, so we zero-pad the
# remaining 150 timepoints. This is a KNOWN, DISCLOSED degradation: 75% of
# every input to the model is a neutral zero placeholder, not real brain
# activity. Padding was chosen over repeating/tiling the real 50 samples
# because repeating would place real recorded values into time slots where
# they never actually occurred (fabricating apparent signal); zero-padding
# just says "no data here" instead of inventing content.
N_TARGET = 200


class ThoughtVizDataset(Dataset):
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
        data = d["X"]  # (14, 32) at 128Hz
        label = d["y"]

        # Resample the real 0.25s of signal to 200Hz (32 -> 50 timepoints).
        data = signal.resample(data, N_REAL, axis=1)

        # Z-score using only the real samples' own statistics, BEFORE padding,
        # so the padded region stays exactly zero rather than being shifted
        # by whatever mean/std the real segment happened to have.
        mu = data.mean(axis=1, keepdims=True)
        sigma = data.std(axis=1, keepdims=True)
        data = (data - mu) / (sigma + 1e-8)

        # Zero-pad from 50 -> 200 timepoints to satisfy REVE's fixed
        # patch_size=200. See module docstring/comment above for why.
        n_chans = data.shape[0]
        padded = np.zeros((n_chans, N_TARGET), dtype=np.float32)
        padded[:, :N_REAL] = data
        data = padded

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