"""BCI Competition 2020 Track 3 (imagined speech, 5 words) preprocessing for REVE.

Reads the existing per-trial pkl folders (bci2020_processed/{train,val,test},
verified identical to the raw .mat data, in subject order) and writes compact
arrays for REVE.

  - Split: official competition split, within-subject:
    15 subjects x (300 train / 50 val / 50 test) = 4500 / 750 / 750
  - Subject of each trial: index // 300 (train) or index // 50 (val/test);
    verified against the raw .mat files for every subject block
  - Channels: 64, names/order from the .mat 'clab' field
  - Window: LAST 3.0 s (768 of 795 samples at 256 Hz) -> resampled to 200 Hz -> 64 x 600,
    matching REVE's own preprocessing/preprocessing_speech.py (eeg[:, :, -768:])
  - Per-trial, per-channel DC offset removed
  - Normalization: per-subject, per-channel z-score using that subject's TRAIN
    trials only, clipped at +/-15

Output (out_dir):
  {split}_X.npy        float32 (N, 64, 600)
  {split}_y.npy        int64   (N,)
  {split}_subject.npy  int64   (N,)  1..15
  meta.json            channels, rates, verification results, counts

Usage (prep env):
  python preprocess_bci2020_reve.py \
      --pkl_dir /lustre10/scratch/nourmaj/bci2020_processed \
      --raw_dir /lustre10/scratch/nourmaj/bci2020 \
      --out_dir /lustre10/scratch/nourmaj/bci2020_reve_processed
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

N_SUBJECTS = 15
PER_SUBJECT = {"train": 300, "val": 50, "test": 50}
RAW_FOLDER = {"train": "Training set", "val": "Validation set", "test": "Test set"}
SRC_SF, TARGET_SF, WINDOW_SEC = 256, 200, 3.0
CLIP = 15.0


def load_raw_epo(raw_dir, split, subject):
    path = os.path.join(raw_dir, RAW_FOLDER[split], f"Data_Sample{subject:02d}.mat")
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        key = [k for k in m if k.startswith("epo")][0]
        return m[key]
    except NotImplementedError:
        # MATLAB v7.3 (HDF5) file -- the Test set is saved this way. h5py returns
        # MATLAB arrays with reversed axes, so transpose back to MATLAB order.
        import h5py
        from types import SimpleNamespace
        with h5py.File(path, "r") as f:
            key = [k for k in f.keys() if k.startswith("epo")][0]
            g = f[key]
            x = np.transpose(g["x"][()])  # -> (time, ch, trials)
            y = np.transpose(g["y"][()]) if "y" in g else None
        return SimpleNamespace(x=x, y=y)


def raw_labels(epo):
    y = getattr(epo, "y", None)
    if y is None:
        return None
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[0] == 5:
        return np.argmax(y, axis=0)
    if y.ndim == 1 and y.size > 0:
        return y.astype(int) - (1 if y.min() == 1 else 0)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--norm", choices=["zscore", "authors"], default="zscore",
                   help="zscore: DC removal + per-subject z-score (default). "
                        "authors: replicate REVE's speech pipeline -- keep DC, divide raw uV by 1000")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clab = [str(c) for c in load_raw_epo(args.raw_dir, "train", 1).clab]
    assert len(clab) == 64, f"expected 64 channels, got {len(clab)}"

    n_crop = int(round(WINDOW_SEC * SRC_SF))  # 768
    data, verify = {}, {}
    for split, n_per in PER_SUBJECT.items():
        files = sorted(glob.glob(os.path.join(args.pkl_dir, split, "trial_*.pkl")))
        assert len(files) == N_SUBJECTS * n_per, f"{split}: {len(files)} files, expected {N_SUBJECTS * n_per}"
        X, Y = [], []
        for f in files:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            X.append(np.asarray(d["X"], dtype=np.float32))
            Y.append(int(d["y"]))
        X = np.stack(X)
        Y = np.asarray(Y, dtype=np.int64)
        S = np.repeat(np.arange(1, N_SUBJECTS + 1), n_per).astype(np.int64)

        sig_status, lab_status, unreadable = [], [], []
        for s in range(1, N_SUBJECTS + 1):
            try:
                epo = load_raw_epo(args.raw_dir, split, s)
                raw_x = np.asarray(epo.x, dtype=np.float32)  # (time, ch, trials)
            except Exception as e:  # truncated / missing raw file: cannot cross-check this block
                unreadable.append(f"Data_Sample{s:02d}.mat ({type(e).__name__})")
                sig_status.append("raw unreadable")
                lab_status.append("raw unreadable")
                continue
            i0 = (s - 1) * n_per
            ok = bool(raw_x.shape[2] == n_per and
                      np.allclose(X[i0], raw_x[:, :, 0].T, atol=1e-3) and
                      np.allclose(X[i0 + n_per - 1], raw_x[:, :, -1].T, atol=1e-3))
            sig_status.append("match" if ok else "MISMATCH")
            ry = raw_labels(epo)
            if ry is None or len(ry) != n_per:
                lab_status.append("no raw labels")
            else:
                lab_status.append("match" if np.array_equal(ry, Y[i0:i0 + n_per]) else "MISMATCH")
        verify[split] = {"signal_blocks": sig_status, "labels": lab_status, "unreadable_raw": unreadable}
        print(f"[verify] {split}: signal blocks { {k: sig_status.count(k) for k in set(sig_status)} }; "
              f"labels { {k: lab_status.count(k) for k in set(lab_status)} }")
        if unreadable:
            print(f"[warn] {split}: raw files unreadable, blocks not cross-checked: {unreadable}")
        if "MISMATCH" in sig_status:
            raise SystemExit(f"[error] {split}: processed trials are not in raw subject order -- stopping")
        if "MISMATCH" in lab_status:
            raise SystemExit(f"[error] {split}: labels differ from raw .mat -- stopping")

        X = X[:, :, -n_crop:]  # LAST 3.0 s of the 3.1 s trial, as in REVE's preprocessing_speech.py
        if args.norm == "zscore":
            X = X - X.mean(axis=2, keepdims=True)
        X = resample_poly(X, TARGET_SF, SRC_SF, axis=2).astype(np.float32)  # 768 -> 600
        data[split] = (X, Y, S)

    Xtr, _, Str = data["train"]
    sd = {s: Xtr[Str == s].std(axis=(0, 2)) for s in range(1, N_SUBJECTS + 1)}  # per-channel
    summary = {}
    for split, (X, Y, S) in data.items():
        if args.norm == "zscore":
            for s in range(1, N_SUBJECTS + 1):
                X[S == s] /= np.maximum(sd[s], 1e-6)[None, :, None]
            np.clip(X, -CLIP, CLIP, out=X)
        else:
            X /= 1000.0  # authors' LMDBDataset scale_factor for speech.yaml; DC offset kept
        np.save(os.path.join(args.out_dir, f"{split}_X.npy"), X)
        np.save(os.path.join(args.out_dir, f"{split}_y.npy"), Y)
        np.save(os.path.join(args.out_dir, f"{split}_subject.npy"), S)
        summary[split] = {
            "trials": int(len(Y)), "shape": list(X.shape),
            "class_counts": np.bincount(Y, minlength=5).tolist(),
            "subjects": int(len(np.unique(S))),
            "std_after_norm": round(float(X.std()), 3),
            "frac_clipped": round(float((np.abs(X) >= CLIP).mean()), 5),
        }

    meta = {
        "channels": clab, "source_sfreq": SRC_SF, "target_sfreq": TARGET_SF,
        "window_sec": WINDOW_SEC, "crop": "last 768 samples",
        "normalization": ("DC removal + per-subject per-channel z-score (train stats), clip 15"
                          if args.norm == "zscore" else "authors: raw uV / 1000, DC kept"),
        "split": "official within-subject: 300/50/50 per subject", "verification": verify, "summary": summary,
        "train_subject_channel_std_uV": {s: np.round(v, 2).tolist() for s, v in sd.items()},
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(summary, indent=2))
    print(f"[info] channels: {clab}")
    print(f"[info] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
