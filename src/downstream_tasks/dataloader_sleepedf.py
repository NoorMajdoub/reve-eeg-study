"""Sleep-EDF (Sleep Cassette) loader for REVE, reading the output of
preprocess_sleepedf_reve.py + compute_sleepedf_stats.py.

Each sample is one 30 s epoch, 2 channels (Fpz-Cz, Pz-Oz) x 6000 samples at 200 Hz.

Normalization: per-recording, per-channel robust z-score
(x - median) / (1.4826 * MAD), then clipped to +/- clip. Per recording (a whole
night), not per epoch, so amplitude differences between stages are preserved.

Quality control (label-blind, identical for train/val/test):
  - recordings whose robust scale exceeds max_scale_uv on either channel are
    dropped (physiologically implausible scalp EEG, i.e. off-scale nights)
  - epochs whose saturation fraction exceeds max_sat_frac are dropped
    (EEG flat-lined at the amplifier limit; scored from other channels, so they
    carry no EEG information for this task)

Speed: with preload=True (default) the whole split is read sequentially into
RAM once, already normalized (train ~7 GB float32). Shuffled training then
reads from memory instead of doing small random reads from Lustre, which was
the training bottleneck (~3 it/s train vs ~57 it/s sequential eval). DataLoader
workers fork after loading, so the array is shared, not copied. Values are
identical to preload=False.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class SleepEDFDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None, clip=15.0,
                 max_sat_frac=0.5, max_scale_uv=100.0, preload=True):
        super().__init__()
        self.path = path
        self.mode = mode
        self.clip = float(clip)
        self.preload = bool(preload)

        with open(os.path.join(path, "split.json")) as f:
            split = json.load(f)
        stats_file = os.path.join(path, "robust_stats.json")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"{stats_file} missing -- run compute_sleepedf_stats.py {path} first")
        with open(stats_file) as f:
            stats = json.load(f)

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)

        kept_recs = []  # (seq_file, keep_idx, median, scale)
        labels, n_total, dropped_recs = [], 0, []
        for rec in split[mode]:
            y = np.load(os.path.join(path, "labels", f"{rec}.npy"))
            n_total += len(y)
            if max(stats[rec]["scale"]) > max_scale_uv:
                dropped_recs.append(rec)
                continue
            sat = np.load(os.path.join(path, "epoch_qc", f"{rec}.npy"))
            keep = np.flatnonzero(sat <= max_sat_frac)
            kept_recs.append((
                os.path.join(path, "seq", f"{rec}.npy"),
                keep,
                np.asarray(stats[rec]["median"], dtype=np.float32)[:, None],
                np.asarray(stats[rec]["scale"], dtype=np.float32)[:, None],
            ))
            labels.append(y[keep])
        self.labels = np.concatenate(labels).astype(np.int64)

        if self.preload:
            n = len(self.labels)
            n_ch, n_t = np.load(kept_recs[0][0], mmap_mode="r").shape[1:]
            self.data = np.empty((n, n_ch, n_t), dtype=np.float32)
            pos = 0
            for seq_file, keep, med, scale in kept_recs:
                x = np.load(seq_file)[keep].astype(np.float32)
                np.clip((x - med[None]) / scale[None], -self.clip, self.clip, out=x)
                self.data[pos:pos + len(keep)] = x
                pos += len(keep)
            assert pos == n
        else:
            self.seq_files = [r[0] for r in kept_recs]
            self.medians = [r[2] for r in kept_recs]
            self.scales = [r[3] for r in kept_recs]
            self.index = [(fi, int(i)) for fi, r in enumerate(kept_recs) for i in r[1]]

        counts = np.bincount(self.labels, minlength=5).tolist()
        mem = f", preloaded {self.data.nbytes / 1e9:.1f} GB" if self.preload else ""
        print(f"[SleepEDF] {mode}: {len(kept_recs)}/{len(split[mode])} recordings "
              f"(dropped off-scale: {dropped_recs}), {len(self.labels):,}/{n_total:,} epochs kept, "
              f"class counts W/N1/N2/N3/REM = {counts}{mem}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.preload:
            x = self.data[idx]
        else:
            fi, i = self.index[idx]
            x = np.array(np.load(self.seq_files[fi], mmap_mode="r")[i], dtype=np.float32)
            x = np.clip((x - self.medians[fi]) / self.scales[fi], -self.clip, self.clip)
        return {
            "sample": torch.from_numpy(np.array(x, dtype=np.float32)),
            "label": torch.tensor(self.labels[idx]).long(),
            "pos": self.positions,
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([b["sample"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "pos": self.positions.repeat(len(batch), 1, 1),
        }