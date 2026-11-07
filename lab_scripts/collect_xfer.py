"""Summarise REVE cross-dataset linear-probe results (reve_xfer_results/*.json):
mean +/- sample std (ddof=1) of test balanced accuracy per (backbone, target),
with the pretrained and random backbones shown next to each source for reference.

Usage: python collect_xfer.py
"""

import glob
import json
import os
import statistics as st
from collections import defaultdict

ROOT = "/lustre10/scratch/nourmaj/reve_xfer_results"


def fmt(vals):
    if not vals:
        return "        -        "
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):6.2f} ± {sd:5.2f} (n={len(vals)})"


def main():
    res = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "*.json")):
        d = json.load(open(f))
        res[(d["source"], d["target"])].append(100 * d["best_test_balanced_acc"])

    targets = sorted({t for _, t in res})
    sources = sorted({s for s, _ in res if s not in ("pretrained", "random")})
    print(f"{'target':12s} | {'pretrained (frozen)':>22s} | {'random (frozen)':>22s} | "
          + " | ".join(f"{'from ' + s:>22s}" for s in sources))
    for t in targets:
        cells = [fmt(res.get(("pretrained", t), [])), fmt(res.get(("random", t), []))]
        cells += [fmt(res.get((s, t), [])) for s in sources]
        print(f"{t:12s} | " + " | ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
