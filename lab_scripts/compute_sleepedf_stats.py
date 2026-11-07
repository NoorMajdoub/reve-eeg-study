"""Per-recording robust normalization stats + per-epoch quality control for the
REVE Sleep-EDF data.

For every recording in split.json (train, val and test):
  - per-channel median and robust scale (1.4826 * MAD) -> robust_stats.json
  - per-epoch saturation fraction: share of samples with
    |x - median| / scale >= 15 (i.e. that would be clipped)  -> epoch_qc/<rec>.npy

Everything is computed from each recording's own signal only -- no labels and
nothing shared across recordings -- so it is safe to use for val/test.
The loader then applies two label-blind QC rules (thresholds in sleepedf.yaml):
  - drop epochs whose saturation fraction > max_sat_frac
  - drop whole recordings whose robust scale > max_scale_uv on either channel

Usage:
  python compute_sleepedf_stats.py /lustre10/scratch/nourmaj/sleepedf_reve_processed
"""

import json
import os
import sys

import numpy as np

CLIP = 15.0
REPORT_MAX_SCALE_UV = 100.0
REPORT_MAX_SAT_FRAC = 0.5


def main(path):
    with open(os.path.join(path, "split.json")) as f:
        split = json.load(f)
    split_of = {r: k for k in ("train", "val", "test") for r in split[k]}
    os.makedirs(os.path.join(path, "epoch_qc"), exist_ok=True)

    stats = {}
    for i, rec in enumerate(split_of, 1):
        x = np.load(os.path.join(path, "seq", f"{rec}.npy"), mmap_mode="r")
        x = np.asarray(x, dtype=np.float32)
        flat = x[::4].transpose(1, 0, 2).reshape(x.shape[1], -1)
        med = np.median(flat, axis=1)
        mad = np.maximum(np.median(np.abs(flat - med[:, None]), axis=1) * 1.4826, 1e-3)
        z = np.abs((x - med[None, :, None]) / mad[None, :, None])
        sat = (z >= CLIP).mean(axis=(1, 2)).astype(np.float32)
        np.save(os.path.join(path, "epoch_qc", f"{rec}.npy"), sat)
        stats[rec] = {"median": med.tolist(), "scale": mad.tolist()}
        if i % 20 == 0 or i == len(split_of):
            print(f"  {i}/{len(split_of)}", flush=True)

    with open(os.path.join(path, "robust_stats.json"), "w") as f:
        json.dump(stats, f)

    print(f"\n[QC preview] recordings with robust scale > {REPORT_MAX_SCALE_UV} uV (would be dropped):")
    bad_recs = {r for r, s in stats.items() if max(s["scale"]) > REPORT_MAX_SCALE_UV}
    for r in sorted(bad_recs):
        print(f"  {r}  split={split_of[r]}  scale_uV={np.round(stats[r]['scale'], 1).tolist()}")
    if not bad_recs:
        print("  none")

    print(f"\n[QC preview] after dropping those recordings and epochs with saturation > {REPORT_MAX_SAT_FRAC}:")
    for k in ("train", "val", "test"):
        tot = kept = 0
        n_rec = 0
        for r in split[k]:
            sat = np.load(os.path.join(path, "epoch_qc", f"{r}.npy"))
            tot += len(sat)
            if r in bad_recs:
                continue
            n_rec += 1
            kept += int((sat <= REPORT_MAX_SAT_FRAC).sum())
        print(f"  {k}: recordings {n_rec}/{len(split[k])}, epochs kept {kept}/{tot} ({100 * kept / tot:.1f}%)")

    scales = np.array([v["scale"] for v in stats.values()])
    print(f"\n[info] scale (uV) across recordings: median {np.median(scales):.1f}, "
          f"p90 {np.percentile(scales, 90):.1f}, max {scales.max():.1f}")
    print(f"[info] wrote robust_stats.json and epoch_qc/ in {path}")


if __name__ == "__main__":
    main(sys.argv[1])
