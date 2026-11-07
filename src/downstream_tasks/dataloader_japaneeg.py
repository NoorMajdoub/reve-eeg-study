"""
JapanEEG (speech-active vs rest) dataset loader for REVE.

NOTE ON POSITIONS: this dataset's true electrode geometry (a custom
left-hemisphere-dominant 128-channel subset of a 1024-channel g.Pangolin
grid) was never published anywhere we could find (paper, BIDS files, and
the dataset's GitHub repo all checked). The positions used here come from
/scratch/nourmaj/japaneeg_positions_APPROXIMATE.npy -- REAL coordinates
borrowed from a different, unrelated standard 128-channel EEG system
(Biosemi128), NOT JapanEEG's actual layout. This is a disclosed stopgap
pending a response from the dataset authors. See
extract_japaneeg_positions.py for full details.

NOTE ON PREPROCESSING: unlike every other domain in this project, the data
was already resampled to 200Hz and cut into clean non-overlapping 10s
windows during preprocessing (process_japaneeg.py) -- no further resampling
is needed here.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class JapanEEGDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None):
        super().__init__()
        self.path = path
        self.mode = mode
        # positions here is the APPROXIMATE .npy file path (see module
        # docstring) -- electrodes is unused for this dataset.
        self.positions = load_positions(positions_path=positions, electrode_names=None)

        self.dir = os.path.join(self.path, mode)
        self.files = sorted(
            os.path.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith(".pkl")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            d = pickle.load(f)
        data = d["X"].astype(np.float32)  # (128, 2000) already at 200Hz, 10s
        label = d["y"]

        # Already at REVE's target frequency and window length -- no resample.
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