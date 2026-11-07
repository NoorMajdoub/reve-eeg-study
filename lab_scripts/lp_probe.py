"""STEP 2 of the linear-probing study: fit the linear probe on saved embeddings (CPU).

For every .npz written by the extraction step, and every embedding type in it (head/mean/flat):
  1. standardize features (mean/std from TRAIN only, applied to val and test)
  2. for each C in the grid: train a logistic regression on TRAIN, score balanced acc on VAL
  3. keep the C with the best VAL score -> report its TEST balanced accuracy
Then print a summary: mean +- std over seeds for each task / arm / fraction / pooling.

Logistic regression = a linear classification head (linear layer + softmax + cross-entropy),
solved exactly instead of trained for epochs. class_weight='balanced' avoids majority-class
collapse on imbalanced sets. Same setup as NeuroAtlas (App. C).

Works for any model's embeddings as long as the .npz uses the same keys.

Usage:
  python lp_probe.py                                   # all files in the feature folder
  python lp_probe.py --filter isruc caueeg             # only some tasks
  python lp_probe.py --feat_dir /path/to/other_model_feats --tag cbramod
"""

import argparse
import glob
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

C_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100]
NAME_RE = re.compile(r"(?P<task>.+)_(?P<arm>pretrained|random)_seed(?P<seed>\d+)_(?P<frac>\d+)pct\.npz$")


def fit_one(C, Xtr, ytr, Xva, yva):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)  # lbfgs solver
        clf.fit(Xtr, ytr)
    return C, balanced_accuracy_score(yva, clf.predict(Xva)), clf


def probe(Xtr, ytr, Xva, yva, Xte, yte, n_jobs):
    scaler = StandardScaler().fit(Xtr)  # train statistics only
    Xtr, Xva, Xte = (scaler.transform(X.astype(np.float32)) for X in (Xtr, Xva, Xte))
    fits = Parallel(n_jobs=n_jobs)(delayed(fit_one)(C, Xtr, ytr, Xva, yva) for C in C_GRID)
    best_C, best_val, clf = max(fits, key=lambda r: r[1])  # chosen on VAL only
    pred = clf.predict(Xte)
    return {"best_C": best_C, "val_bacc": best_val,
            "test_bacc": balanced_accuracy_score(yte, pred),
            "test_kappa": cohen_kappa_score(yte, pred),
            "test_f1_macro": f1_score(yte, pred, average="macro"),
            "val_by_C": {str(C): v for C, v, _ in fits},
            "test_pred_classes": int(len(np.unique(pred)))}  # 1 = collapsed to one class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="/scratch/nourmaj/reve_lp_feats")
    ap.add_argument("--out", default=None, help="results json (default: <feat_dir>/lp_results.json)")
    ap.add_argument("--filter", nargs="*", default=None)
    ap.add_argument("--pooling", nargs="*", default=["head", "mean", "flat"])
    ap.add_argument("--n_jobs", type=int, default=6)
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.feat_dir, "lp_results.json")

    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    files = sorted(glob.glob(os.path.join(a.feat_dir, "*.npz")))
    if a.filter:
        files = [f for f in files if any(s in os.path.basename(f) for s in a.filter)]

    for f in files:
        m = NAME_RE.search(os.path.basename(f))
        if not m:
            continue
        d = np.load(f)
        for pool in a.pooling:
            key = f"{os.path.basename(f)[:-4]}|{pool}"
            if key in results or f"X_train_{pool}" not in d:
                continue
            ytr = np.asarray(d["y_train"]).ravel()
            if len(np.unique(ytr)) < 2:
                print(f"{key:55s} SKIPPED: training set has a single class (counts {np.unique(ytr, return_counts=True)})")
                continue
            try:
                r = probe(d[f"X_train_{pool}"], ytr, d[f"X_val_{pool}"], np.asarray(d["y_val"]).ravel(),
                          d[f"X_test_{pool}"], np.asarray(d["y_test"]).ravel(), a.n_jobs)
            except Exception as e:  # keep going: one bad file must not stop the whole run
                print(f"{key:55s} ERROR: {type(e).__name__}: {e}")
                continue
            r.update({k: m[k] for k in ("task", "arm")}, seed=int(m["seed"]), fraction=int(m["frac"]),
                     pooling=pool, n_train=int(len(d["y_train"])), dim=int(d[f"X_train_{pool}"].shape[1]))
            results[key] = r
            print(f"{key:55s} C={r['best_C']:<6g} val {r['val_bacc']:.4f} test {r['test_bacc']:.4f}"
                  + ("  [COLLAPSED: 1 class predicted]" if r["test_pred_classes"] == 1 else ""), flush=True)
            json.dump(results, open(out_path, "w"), indent=1)  # save after every fit (resumable)

    # Summary table: mean +- sample std over seeds
    groups = defaultdict(list)
    for r in results.values():
        groups[(r["task"], r["pooling"], r["fraction"], r["arm"])].append(r["test_bacc"] * 100)
    print(f"\n{'task':12s} {'pool':5s} {'frac':>4s}  {'pretrained':>16s}  {'random (3 seeds)':>18s}  {'delta':>6s}")
    for task, pool, frac in sorted({k[:3] for k in groups}):
        pt, rd = groups.get((task, pool, frac, "pretrained"), []), groups.get((task, pool, frac, "random"), [])
        fmt = lambda v: f"{np.mean(v):6.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:5.2f} (n={len(v)})" if v else "—"
        delta = f"{np.mean(pt) - np.mean(rd):+6.2f}" if pt and rd else ""
        print(f"{task:12s} {pool:5s} {frac:>3d}%  {fmt(pt):>16s}  {fmt(rd):>18s}  {delta}")
    print(f"\n[probe] results saved to {out_path}")


if __name__ == "__main__":
    main()
