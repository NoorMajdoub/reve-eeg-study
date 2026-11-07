"""CHB-MIT (binary seizure detection) dataset loader, mirroring TUEV's TUH loader interface."""

import os
import pickle
from os.path import join as pjoin

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions

SOURCE_SR = 256
TARGET_SR = 200
WINDOW_SECONDS = 10
FULL_LEN_SOURCE = SOURCE_SR * WINDOW_SECONDS  # 2560: expected length of a full segment at 256Hz
TARGET_LEN = int(round(FULL_LEN_SOURCE * (TARGET_SR / SOURCE_SR)))  # 2000: fixed output length at 200Hz


class CHBMIT(Dataset):
    def __init__(
        self,
        path: str,
        mode: str,
        positions=None,
        electrodes=None,
        scale_factor: float = 1.0,
    ):
        assert mode in ["train", "val", "test"]
        self.scale_factor = scale_factor
        self.data_path = pjoin(path, mode)
        self.file_names = sorted(
            f for f in os.listdir(self.data_path) if not f.startswith(".")
        )
        self.positions = load_positions(positions_path=positions, electrode_names=electrodes).float()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_path = pjoin(self.data_path, self.file_names[index])
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        X = data["X"]  # (n_chans, n_samples) at SOURCE_SR -- some trailing segments are shorter than FULL_LEN_SOURCE
        n = X.shape[-1]
        if n < FULL_LEN_SOURCE:
            X = np.pad(X, ((0, 0), (0, FULL_LEN_SOURCE - n)), mode="constant")
        elif n > FULL_LEN_SOURCE:
            X = X[:, :FULL_LEN_SOURCE]
        X = resample(X, TARGET_LEN, axis=-1)  # -> (n_chans, TARGET_LEN) at TARGET_SR, always fixed size
        X = torch.as_tensor(X).float()
        y = torch.as_tensor(data["y"]).long().reshape(())
        return {
            "sample": X * self.scale_factor,
            "label": y,
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([x["sample"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "pos": self.positions.unsqueeze(0).repeat(len(batch), 1, 1),
        }