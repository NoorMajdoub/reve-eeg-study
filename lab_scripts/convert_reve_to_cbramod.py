"""Convert REVE-processed datasets into CBraMod's one-pkl-per-window format,
so both models train/evaluate on literally the same windows, subjects and QC.

Output: <dst>/{train,val,test}/<rec>_<idx>.pkl, each {'X': float32 array, 'y': int}
plus <dst>/split.json (copied) and <dst>/conversion_summary.json.

  sleepedf: X = (2, 6000) raw uV at 200 Hz (Fpz-Cz, Pz-Oz). Same QC as the
            REVE loader: recordings with robust scale > max_scale_uv dropped,
            epochs with saturation fraction > max_sat_frac dropped.
            CBraMod's Dataset resample(6000) is then a no-op and its own
            per-epoch z-score is applied as before.
  caueeg:   X = (19, 2000), 10 s windows from dementia-no-overlap.json
            (950/82/90). Values are 0.5 Hz high-passed and per-recording
            z-scored (+/-15 SD clip) as stored for REVE; CBraMod's own
            per-window z-score is applied on top as before.

Usage:
  python convert_reve_to_cbramod.py --task sleepedf \
      --src /lustre10/scratch/nourmaj/sleepedf_reve_processed \
      --dst /lustre10/scratch/nourmaj/sleepedf_cbramod_from_reve
  python convert_reve_to_cbramod.py --task caueeg \
      --src /lustre10/scratch/nourmaj/caueeg_reve_processed \
      --dst /lustre10/scratch/nourmaj/caueeg_cbramod_from_reve
"""

import argparse
import json
import os
import pickle
import shutil
from multiprocessing import get_context

import numpy as np

N_CLASSES = {"sleepedf": 5, "caueeg": 3}


def convert_recording(job):
    rec, split, src, dst, keep_idx = job
    X = np.load(os.path.join(src, "seq", f"{rec}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(src, "labels", f"{rec}.npy"))
    out_dir = os.path.join(dst, split)
    idx = range(len(Y)) if keep_idx is None else keep_idx
    counts = np.zeros(max(N_CLASSES.values()), dtype=np.int64)
    n = 0
    for i in idx:
        x = np.ascontiguousarray(X[i], dtype=np.float32)
        y = int(Y[i])
        with open(os.path.join(out_dir, f"{rec}_{i:05d}.pkl"), "wb") as f:
            pickle.dump({"X": x, "y": y}, f, protocol=pickle.HIGHEST_PROTOCOL)
        counts[y] += 1
        n += 1
    return split, rec, n, counts.tolist()


def build_jobs(task, src, dst, max_sat_frac, max_scale_uv):
    with open(os.path.join(src, "split.json")) as f:
        split = json.load(f)
    jobs, dropped = [], []
    stats = None
    if task == "sleepedf":
        with open(os.path.join(src, "robust_stats.json")) as f:
            stats = json.load(f)
    for k in ("train", "val", "test"):
        for rec in split[k]:
            keep = None
            if task == "sleepedf":
                if max(stats[rec]["scale"]) > max_scale_uv:
                    dropped.append(rec)
                    continue
                sat = np.load(os.path.join(src, "epoch_qc", f"{rec}.npy"))
                keep = np.flatnonzero(sat <= max_sat_frac).tolist()
            jobs.append((rec, k, src, dst, keep))
    return jobs, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["sleepedf", "caueeg"])
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--max_sat_frac", type=float, default=0.5)
    p.add_argument("--max_scale_uv", type=float, default=100.0)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = p.parse_args()

    if os.path.exists(args.dst) and any(os.scandir(args.dst)):
        raise SystemExit(f"{args.dst} is not empty -- remove it or choose another --dst")
    for k in ("train", "val", "test"):
        os.makedirs(os.path.join(args.dst, k), exist_ok=True)

    jobs, dropped = build_jobs(args.task, args.src, args.dst, args.max_sat_frac, args.max_scale_uv)
    print(f"[info] {args.task}: {len(jobs)} recordings to convert, dropped recordings: {dropped}")

    summary = {k: {"recordings": 0, "windows": 0, "class_counts": [0] * N_CLASSES[args.task]}
               for k in ("train", "val", "test")}
    with get_context("spawn").Pool(args.workers) as pool:
        for i, (k, rec, n, counts) in enumerate(pool.imap_unordered(convert_recording, jobs), 1):
            s = summary[k]
            s["recordings"] += 1
            s["windows"] += n
            s["class_counts"] = [a + b for a, b in zip(s["class_counts"], counts[: N_CLASSES[args.task]])]
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} recordings done", flush=True)

    shutil.copy(os.path.join(args.src, "split.json"), os.path.join(args.dst, "split.json"))
    summary["dropped_recordings"] = dropped
    summary["qc"] = ({"max_sat_frac": args.max_sat_frac, "max_scale_uv": args.max_scale_uv}
                     if args.task == "sleepedf" else "none (clean no-overlap split)")
    with open(os.path.join(args.dst, "conversion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[info] done -> {args.dst}")


if __name__ == "__main__":
    main()
