"""Sleep-EDF (Sleep Cassette) preprocessing for REVE.

Raw EDF -> per-recording .npy files + one canonical patient-disjoint split.

  - Channels: EEG Fpz-Cz, EEG Pz-Oz (in this order), converted V -> uV
  - Resampled 100 Hz -> 200 Hz on the continuous signal (before epoching)
  - No extra filtering (Sleep-EDF is already hardware filtered; keeps inputs
    identical for every model that reuses these files)
  - Labels: W=0, N1=1, N2=2, N3=3 (stage 3+4 merged), REM=4;
    'Sleep stage ?' and 'Movement time' dropped
  - Wake trimmed to 30 min before first / after last sleep epoch,
    crop end clamped to the real recording length
  - Split: 80/10/10 by SUBJECT (both nights of a subject stay together), seed 42
  - Normalization stats (per-channel mean/std) computed on TRAIN only and
    saved to norm_stats.json -- applied in the dataloader, not baked in here

Output layout (out_dir):
  seq/<rec>.npy      float32 (n_epochs, 2, 6000)
  labels/<rec>.npy   int64   (n_epochs,)
  split.json         {"train": [...], "val": [...], "test": [...], "subjects": {...}}
  norm_stats.json    {"mean": [c0, c1], "std": [c0, c1], ...}
  summary.json       per-split epoch / class counts

Usage (on a compute node, REVE venv active):
  python preprocess_sleepedf_reve.py \
      --raw_dir /lustre10/scratch/nourmaj/sleepedf/sleep-cassette/sleep-cassette \
      --out_dir /lustre10/scratch/nourmaj/sleepedf_reve_processed --workers 8
"""

import argparse
import glob
import json
import os
import random
import re
from fractions import Fraction
from multiprocessing import get_context

import mne
import numpy as np
from scipy.signal import resample_poly

mne.set_log_level("ERROR")

CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
TARGET_SF = 200
EPOCH_SEC = 30
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def subject_of(fname):
    m = re.match(r"SC4(\d{2})\d", fname)
    if m is None:
        raise ValueError(f"Unexpected filename: {fname}")
    return m.group(1)


def find_pairs(raw_dir):
    pairs = []
    for psg in sorted(glob.glob(os.path.join(raw_dir, "*-PSG.edf"))):
        base = os.path.basename(psg)
        hyps = glob.glob(os.path.join(raw_dir, base[:7] + "?-Hypnogram.edf"))
        if not hyps:
            print(f"[skip] no hypnogram for {base}")
            continue
        pairs.append((base.replace("-PSG.edf", ""), psg, hyps[0]))
    return pairs


def process_recording(job):
    rec, psg, hyp, out_dir, wake_pad_min = job

    raw = mne.io.read_raw_edf(psg, include=CHANNELS, preload=True, verbose="ERROR")
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=CHANNELS) * 1e6  # V -> uV, fixed channel order

    frac = Fraction(TARGET_SF / sf).limit_denominator(1000)
    if frac != 1:
        data = resample_poly(data, frac.numerator, frac.denominator, axis=-1)
    n_samp = data.shape[1]
    rec_sec = n_samp / TARGET_SF

    annot = mne.read_annotations(hyp)
    sleep_iv = [(o, o + d) for o, d, s in zip(annot.onset, annot.duration, annot.description)
                if STAGE_MAP.get(s, 0) != 0]
    if not sleep_iv:
        return rec, 0, None
    pad = wake_pad_min * 60
    crop_start = max(0.0, min(a for a, _ in sleep_iv) - pad)
    crop_end = min(rec_sec, max(b for _, b in sleep_iv) + pad)

    ep_len = EPOCH_SEC * TARGET_SF
    X, Y = [], []
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        lab = STAGE_MAP.get(desc)
        if lab is None:
            continue
        for k in range(int(round(dur)) // EPOCH_SEC):
            t0 = onset + k * EPOCH_SEC
            if t0 < crop_start or t0 + EPOCH_SEC > crop_end:
                continue
            s = int(round(t0 * TARGET_SF))
            e = s + ep_len
            if e > n_samp:
                continue
            X.append(data[:, s:e])
            Y.append(lab)

    if not X:
        return rec, 0, None

    X = np.stack(X).astype(np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    np.save(os.path.join(out_dir, "seq", f"{rec}.npy"), X)
    np.save(os.path.join(out_dir, "labels", f"{rec}.npy"), Y)

    x64 = X.astype(np.float64)
    stats = {
        "n_values": X.shape[0] * X.shape[2],
        "sum": x64.sum(axis=(0, 2)).tolist(),
        "sumsq": (x64 ** 2).sum(axis=(0, 2)).tolist(),
        "counts": np.bincount(Y, minlength=5).tolist(),
    }
    return rec, len(Y), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wake_pad_min", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = p.parse_args()

    os.makedirs(os.path.join(args.out_dir, "seq"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)

    pairs = find_pairs(args.raw_dir)
    print(f"[info] {len(pairs)} PSG/hypnogram pairs")

    jobs = [(rec, psg, hyp, args.out_dir, args.wake_pad_min) for rec, psg, hyp in pairs]
    results = []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_recording, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)

    stats_by_rec = {rec: st for rec, n, st in results if n > 0}
    dropped = [rec for rec, n, _ in results if n == 0]
    if dropped:
        print(f"[info] dropped {len(dropped)} recordings with no usable epochs: {dropped}")

    by_subject = {}
    for rec in sorted(stats_by_rec):
        by_subject.setdefault(subject_of(rec), []).append(rec)
    subjects = sorted(by_subject)
    random.Random(args.seed).shuffle(subjects)
    n = len(subjects)
    n_train, n_val = round(0.8 * n), round(0.1 * n)
    subj_split = {
        "train": sorted(subjects[:n_train]),
        "val": sorted(subjects[n_train:n_train + n_val]),
        "test": sorted(subjects[n_train + n_val:]),
    }
    split = {k: [r for s in v for r in by_subject[s]] for k, v in subj_split.items()}
    split["subjects"] = subj_split
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    tot_n = sum(stats_by_rec[r]["n_values"] for r in split["train"])
    tot_s = np.sum([stats_by_rec[r]["sum"] for r in split["train"]], axis=0)
    tot_sq = np.sum([stats_by_rec[r]["sumsq"] for r in split["train"]], axis=0)
    mean = tot_s / tot_n
    std = np.sqrt(tot_sq / tot_n - mean ** 2)
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump({"channels": CHANNELS, "units": "uV", "computed_on": "train",
                   "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    summary = {}
    for k in ["train", "val", "test"]:
        counts = np.sum([stats_by_rec[r]["counts"] for r in split[k]], axis=0)
        summary[k] = {
            "subjects": len(subj_split[k]),
            "recordings": len(split[k]),
            "epochs": int(counts.sum()),
            "hours": round(counts.sum() * EPOCH_SEC / 3600, 1),
            "class_counts": dict(zip(CLASS_NAMES, counts.astype(int).tolist())),
        }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[info] train channel mean (uV): {np.round(mean, 3).tolist()}  std (uV): {np.round(std, 3).tolist()}")
    print(f"[info] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
