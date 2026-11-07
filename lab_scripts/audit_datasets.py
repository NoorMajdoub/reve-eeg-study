"""Audit every processed dataset folder used in the study (read-only, CPU).

For each dataset folder and each split (train/val/test) it reports:
  - number of windows (files)
  - label distribution (reads every file's 'y')
  - sample array shape / dtype / value range (from a few files) -> channels x samples
  - subject IDs guessed from file names, number of subjects per split,
    and whether any subject appears in more than one split (leakage check)
  - a few example file names (so the subject-ID rule can be checked by eye)

Writes a Markdown report and a JSON file next to each other.
Nothing is modified or deleted.

Usage (any env with numpy):
  python audit_datasets.py                          # all folders in DATASETS below
  python audit_datasets.py --only chb-mit-processed-80h ds004504_processed
  python audit_datasets.py --max_files 2000         # quick pass: label counts from a sample
"""
import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = "/scratch/nourmaj"
DATASETS = [
    "chb-mit-processed-80h", "ds004504_processed", "karaone_processed",
    "japaneeg_processed_80h", "dream_processed", "yang2025_processed",
    "isruc_biot_processed", "isruc_processed_80h", "thoughtviz_processed",
    "caueeg_reve_processed", "sleepedf_reve_processed", "stress_reve_processed",
]
SPLITS = ["train", "val", "test"]


def subject_of(name):
    """Best-effort subject ID from a file name: leading token before the first '_', '-' or '.'.
    The report prints example names so this rule can be checked/adjusted."""
    base = os.path.basename(name)
    m = re.match(r"([A-Za-z]*\d+|[A-Za-z]+-?\d+|[^_\-.]+)", base)
    return m.group(1) if m else base


def read_label(path):
    try:
        with open(path, "rb") as f:
            s = pickle.load(f)
        y = s["y"] if isinstance(s, dict) and "y" in s else s.get("label") if isinstance(s, dict) else None
        if hasattr(y, "item"):
            y = y.item() if np.size(y) == 1 else tuple(np.ravel(y).tolist())
        return y
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def describe_sample(path):
    try:
        with open(path, "rb") as f:
            s = pickle.load(f)
        if isinstance(s, dict):
            X = s.get("X", s.get("sample", s.get("data")))
            keys = list(s.keys())
        else:
            X, keys = s, ["<non-dict>"]
        X = np.asarray(X)
        return {"keys": keys, "shape": list(X.shape), "dtype": str(X.dtype),
                "min": float(np.nanmin(X)), "max": float(np.nanmax(X)),
                "abs_p95": float(np.nanpercentile(np.abs(X), 95))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def audit_split(folder, split, max_files, workers):
    d = os.path.join(folder, split)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if not f.startswith("."))
    exts = Counter(os.path.splitext(f)[1] or "<none>" for f in files)
    out = {"n_files": len(files), "extensions": dict(exts), "examples": files[:5]}
    pkl = [f for f in files if f.endswith(".pkl")]
    if pkl:
        use = pkl if not max_files or len(pkl) <= max_files else \
            [pkl[i] for i in np.linspace(0, len(pkl) - 1, max_files).astype(int)]
        with ProcessPoolExecutor(workers) as ex:
            labels = list(ex.map(read_label, [os.path.join(d, f) for f in use], chunksize=64))
        out["labels_counted_from"] = len(use)
        out["label_counts"] = {str(k): v for k, v in sorted(Counter(labels).items(), key=lambda kv: str(kv[0]))}
        out["sample"] = describe_sample(os.path.join(d, pkl[0]))
    subj = Counter(subject_of(f) for f in files)
    out["n_subjects_guess"] = len(subj)
    out["subjects_guess"] = sorted(subj)[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max_files", type=int, default=0, help="0 = read every file's label")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset_audit"))
    a = ap.parse_args()

    report, md = {}, ["# Dataset audit\n"]
    for name in (a.only or DATASETS):
        folder = os.path.join(a.root, name)
        if not os.path.isdir(folder):
            md.append(f"## {name}\n\n**not found** at `{folder}`\n")
            continue
        top = sorted(os.listdir(folder))
        info = {"top_level": top[:50]}
        for sp in SPLITS:
            info[sp] = audit_split(folder, sp, a.max_files, a.workers)
        # leakage: subjects present in more than one split
        sets = {sp: set(info[sp]["subjects_guess"]) for sp in SPLITS if info.get(sp)}
        overlap = {f"{x}&{y}": sorted(sets[x] & sets[y])[:20]
                   for i, x in enumerate(sets) for y in list(sets)[i + 1:] if sets[x] & sets[y]}
        info["subject_overlap_guess"] = overlap
        report[name] = info

        md.append(f"## {name}\n")
        md.append(f"top level: `{', '.join(top[:20])}`\n")
        md.append("| split | windows | labels | subjects (guess) | sample shape | dtype | abs p95 |")
        md.append("|---|---|---|---|---|---|---|")
        for sp in SPLITS:
            s = info.get(sp)
            if not s:
                md.append(f"| {sp} | — | — | — | — | — | — |")
                continue
            smp = s.get("sample", {})
            md.append(f"| {sp} | {s['n_files']:,} | {s.get('label_counts', '—')} | {s['n_subjects_guess']} | "
                      f"{smp.get('shape', smp.get('error', '—'))} | {smp.get('dtype', '—')} | "
                      f"{smp.get('abs_p95', float('nan')):.3g} |")
        for sp in SPLITS:
            if info.get(sp):
                md.append(f"\n- {sp} example files: `{info[sp]['examples']}`")
        md.append(f"\n- sample keys: `{(info.get('train') or {}).get('sample', {}).get('keys')}`")
        md.append(f"- **subject overlap between splits (guess):** {overlap if overlap else 'none found'}\n")
        print(f"[audit] done {name}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "dataset_audit.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(a.out, "dataset_audit.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[audit] wrote {a.out}/dataset_audit.md and dataset_audit.json")


if __name__ == "__main__":
    main()
