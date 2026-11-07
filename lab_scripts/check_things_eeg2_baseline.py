"""Quick linear baseline for THINGS-EEG2 option A: 200-way classification of the
200 test-condition images (80 repetitions each per subject), subject-disjoint
split from things_eeg2_split_manifest.csv (7 train / 2 val / 1 test subjects).

Label = image index 0..199. The test images are identical and identically ordered
for all subjects, so no metadata files are needed.

Features: 17 channels x 10 time bins (mean over each 100 ms) = 170 features.
Classifier: shrinkage LDA (fast, a standard EEG decoding baseline).

Tells us whether single-trial, cross-subject 200-way decoding carries ANY signal
before spending GPU time. Chance: top-1 = 0.005, top-5 = 0.025.

Usage (REVE env, has sklearn):
  python check_things_eeg2_baseline.py /lustre10/scratch/nourmaj/things_eeg2
"""

import csv
import glob
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score


def load_subject(root, subj):
    f = glob.glob(os.path.join(root, subj, "**", "preprocessed_eeg_test.npy"), recursive=True)[0]
    d = np.load(f, allow_pickle=True).item()
    x = np.asarray(d["preprocessed_eeg_data"], dtype=np.float32)  # (200, 80, 17, 100)
    n_img, n_rep = x.shape[:2]
    x = x.reshape(n_img * n_rep, x.shape[2], 10, 10).mean(axis=3)  # 10 bins of 100 ms
    y = np.repeat(np.arange(n_img), n_rep)
    return x.reshape(len(y), -1), y


def topk(proba, y, k):
    top = np.argsort(-proba, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def main(root):
    split = {}
    with open(os.path.join(root, "things_eeg2_split_manifest.csv")) as f:
        for row in csv.DictReader(f):
            split.setdefault(row["split"], []).append(row["subject"])
    print({k: v for k, v in split.items()})

    data = {}
    for k, subs in split.items():
        parts = [load_subject(root, s) for s in subs]
        data[k] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        print(f"{k}: {data[k][0].shape[0]:,} trials, {data[k][0].shape[1]} features")

    mu, sd = data["train"][0].mean(0), data["train"][0].std(0) + 1e-6
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit((data["train"][0] - mu) / sd, data["train"][1])

    print("\nchance: top-1 0.005, top-5 0.025")
    for k in ("val", "test"):
        X, y = data[k]
        p = lda.predict_proba((X - mu) / sd)
        print(f"{k:5s} balanced acc (top-1) {balanced_accuracy_score(y, p.argmax(1)):.4f}   top-5 {topk(p, y, 5):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
