"""CAUEEG-Dementia preprocessing for REVE.

Raw EDF (signal/edf/<serial>.edf) -> per-recording .npy windows.

  - Split: dementia-no-overlap.json (train 950 / val 82 / test 90),
    patient-disjoint. Membership is read from the json, never from folders.
  - Channels: the 19 EEG channels, average reference (EKG, Photic dropped),
    fixed order below
  - Sampling rate: native 200 Hz (resampled only if a file differs)
  - Filtering: 0.5 Hz high-pass (a 99.5 Hz low-pass is meaningless at 200 Hz)
  - Normalization: per-recording, per-channel z-score, clipped at +/-15 SD
    (REVE's pretraining convention) -- baked into the saved windows
  - Windows: 10 s non-overlapping (19 x 2000), remainder dropped;
    every window inherits the recording's label (Normal=0, MCI=1, Dementia=2)

Output layout (out_dir):
  seq/<serial>.npy     float32 (n_windows, 19, 2000)
  labels/<serial>.npy  int64   (n_windows,)
  split.json           {"train": [serials], "val": [...], "test": [...]}
  summary.json         per-split recording / window / class counts
"""

import argparse
import json
import os
from fractions import Fraction
from multiprocessing import get_context

import mne
import numpy as np
from scipy.signal import resample_poly

mne.set_log_level("ERROR")

EDF_CHANNELS = [
    "Fp1-AVG", "F3-AVG", "C3-AVG", "P3-AVG", "O1-AVG",
    "Fp2-AVG", "F4-AVG", "C4-AVG", "P4-AVG", "O2-AVG",
    "F7-AVG", "T3-AVG", "T5-AVG", "F8-AVG", "T4-AVG", "T6-AVG",
    "FZ-AVG", "CZ-AVG", "PZ-AVG",
]
TARGET_SF = 200
WINDOW_SEC = 10
CLIP = 15.0
SPLIT_KEYS = {"train": "train_split", "val": "validation_split", "test": "test_split"}
CLASS_NAMES = ["Normal", "MCI", "Dementia"]


def process_recording(job):
    serial, label, edf_dir, out_dir = job
    raw = mne.io.read_raw_edf(os.path.join(edf_dir, f"{serial}.edf"), include=EDF_CHANNELS,
                              preload=True, verbose="ERROR")
    missing = [c for c in EDF_CHANNELS if c not in raw.ch_names]
    if missing:
        return serial, 0, f"missing channels {missing}"

    raw.filter(l_freq=0.5, h_freq=None, verbose="ERROR")
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=EDF_CHANNELS) * 1e6  # V -> uV, fixed order

    if sf != TARGET_SF:
        frac = Fraction(TARGET_SF / sf).limit_denominator(1000)
        data = resample_poly(data, frac.numerator, frac.denominator, axis=-1)

    mu = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True)
    if np.any(sd < 1e-6):
        return serial, 0, "flat channel"
    data = np.clip((data - mu) / sd, -CLIP, CLIP)

    win = WINDOW_SEC * TARGET_SF
    n_win = data.shape[1] // win
    if n_win == 0:
        return serial, 0, "shorter than one window"
    X = data[:, : n_win * win].reshape(len(EDF_CHANNELS), n_win, win).transpose(1, 0, 2)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.full(n_win, label, dtype=np.int64)

    np.save(os.path.join(out_dir, "seq", f"{serial}.npy"), X)
    np.save(os.path.join(out_dir, "labels", f"{serial}.npy"), Y)
    return serial, n_win, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caueeg_dir", required=True, help="folder containing signal/edf and dementia-no-overlap.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split_json", default="dementia-no-overlap.json")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = p.parse_args()

    os.makedirs(os.path.join(args.out_dir, "seq"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)

    with open(os.path.join(args.caueeg_dir, args.split_json)) as f:
        meta = json.load(f)
    edf_dir = os.path.join(args.caueeg_dir, "signal", "edf")

    jobs, split_of = [], {}
    for split, key in SPLIT_KEYS.items():
        for entry in meta[key]:
            serial = str(entry["serial"])
            split_of[serial] = split
            jobs.append((serial, int(entry["class_label"]), edf_dir, args.out_dir))

    seen = {}
    for s, sp in split_of.items():
        seen.setdefault(s, set()).add(sp)
    assert all(len(v) == 1 for v in seen.values()), "a serial appears in more than one split"

    print(f"[info] {len(jobs)} recordings, {args.workers} workers")
    with get_context("spawn").Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process_recording, jobs), 1):
            results.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)

    failed = {s: why for s, n, why in results if n == 0}
    if failed:
        print(f"[warn] {len(failed)} recordings skipped: {failed}")
    n_by_serial = {s: n for s, n, _ in results if n > 0}
    label_of = {j[0]: j[1] for j in jobs}

    split = {sp: sorted(s for s in n_by_serial if split_of[s] == sp) for sp in SPLIT_KEYS}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({**split, "source": args.split_json}, f, indent=2)

    summary = {}
    for sp, serials in split.items():
        rec_counts = np.bincount([label_of[s] for s in serials], minlength=3)
        win_counts = np.zeros(3, dtype=int)
        for s in serials:
            win_counts[label_of[s]] += n_by_serial[s]
        summary[sp] = {
            "recordings": len(serials),
            "windows": int(win_counts.sum()),
            "recordings_per_class": dict(zip(CLASS_NAMES, rec_counts.tolist())),
            "windows_per_class": dict(zip(CLASS_NAMES, win_counts.tolist())),
        }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[info] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
