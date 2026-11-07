"""THINGS-EEG2 preprocessing for REVE -- option A: 200-way image classification.

  - Data: the 200 test-condition images x 80 repetitions per subject
    (preprocessed_eeg_test.npy from the authors' preprocessed release:
    17 posterior channels, 100 Hz, -0.2..0.79 s, baseline-corrected, MVNN-whitened)
  - Label: image index 0..199 (identical images and order for every subject,
    so no metadata files are needed)
  - Split: subject-disjoint, from things_eeg2_split_manifest.csv
    (7 train / 2 val / 1 test subjects -> 112,000 / 32,000 / 16,000 trials)
  - Resampled 100 -> 200 Hz: 100 -> 200 samples = exactly one REVE patch
  - Per-channel z-score with TRAIN-subject statistics, clip +/-15
    (data is already whitened, so this is close to a no-op; keeps the
    convention identical to the other datasets)

Output (out_dir): {split}_X.npy float32 (N, 17, 200), {split}_y.npy int64,
                  {split}_subject.npy (str), meta.json

Usage (prep or REVE env):
  python preprocess_things_eeg2_reve.py --root /lustre10/scratch/nourmaj/things_eeg2 \
      --out_dir /lustre10/scratch/nourmaj/things_eeg2_reve_processed
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.signal import resample_poly

CLIP = 15.0


def load_subject(root, subj):
    f = glob.glob(os.path.join(root, subj, "**", "preprocessed_eeg_test.npy"), recursive=True)[0]
    d = np.load(f, allow_pickle=True).item()
    x = np.asarray(d["preprocessed_eeg_data"], dtype=np.float32)  # (200, 80, 17, 100)
    n_img, n_rep, n_ch, n_t = x.shape
    x = x.reshape(n_img * n_rep, n_ch, n_t)
    x = resample_poly(x, 2, 1, axis=2).astype(np.float32)  # 100 -> 200 samples
    y = np.repeat(np.arange(n_img), n_rep).astype(np.int64)
    return x, y, [str(c) for c in d["ch_names"]], np.asarray(d["times"]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    split = {}
    with open(os.path.join(args.root, "things_eeg2_split_manifest.csv")) as f:
        for row in csv.DictReader(f):
            split.setdefault(row["split"], []).append(row["subject"])
    assert not (set(split["train"]) & set(split["val"]) | set(split["train"]) & set(split["test"])
                | set(split["val"]) & set(split["test"])), "subject in more than one split"
    print("[info] split:", split)

    data, ch_names, times = {}, None, None
    for k in ("train", "val", "test"):
        xs, ys, ss = [], [], []
        for subj in split[k]:
            x, y, ch, t = load_subject(args.root, subj)
            if ch_names is None:
                ch_names, times = ch, t
            assert ch == ch_names, f"channel order differs for {subj}"
            xs.append(x)
            ys.append(y)
            ss.append(np.array([subj] * len(y)))
        data[k] = [np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)]

    mu = data["train"][0].mean(axis=(0, 2), keepdims=True)
    sd = data["train"][0].std(axis=(0, 2), keepdims=True) + 1e-6
    summary = {}
    for k, (X, Y, S) in data.items():
        X -= mu
        X /= sd
        np.clip(X, -CLIP, CLIP, out=X)
        np.save(os.path.join(args.out_dir, f"{k}_X.npy"), X)
        np.save(os.path.join(args.out_dir, f"{k}_y.npy"), Y)
        np.save(os.path.join(args.out_dir, f"{k}_subject.npy"), S)
        counts = np.bincount(Y, minlength=200)
        summary[k] = {
            "subjects": sorted(set(S.tolist())), "trials": int(len(Y)), "shape": list(X.shape),
            "classes": int((counts > 0).sum()), "trials_per_class": [int(counts.min()), int(counts.max())],
            "std": round(float(X.std()), 3), "frac_clipped": round(float((np.abs(X) >= CLIP).mean()), 6),
        }

    meta = {"task": "200-way image classification (test-condition images)", "channels": ch_names,
            "source_sfreq": 100, "target_sfreq": 200, "times_source": [times[0], times[-1]],
            "normalization": "per-channel z-score (train subjects), clip 15", "summary": summary}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(summary, indent=2))
    print(f"[info] channels: {ch_names}")
    print(f"[info] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
