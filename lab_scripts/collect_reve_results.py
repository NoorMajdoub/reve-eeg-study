"""Collect REVE production results across seeds -> mean +/- sample std (ddof=1).

Scans production logs (reve_<task>_<arm>_<jobid>.out) for:
  - "Output directory: .../<name>"   (e.g. caueeg_pretrained_seed2)
  - the last "Stage ft | ..." line   -> best test / best val (balanced accuracy)
  - "Model saved to"                 -> run finished
Groups by output-directory name with the _seedN suffix removed. If the same
directory appears in several logs (re-runs), the newest job id wins.
Unfinished runs are listed and excluded from the statistics.

Also prints every Optuna *_best.json for reference (--optuna).

Usage (any env with python3):
  python collect_reve_results.py                  # production table
  python collect_reve_results.py --optuna         # + Optuna best values
  python collect_reve_results.py --filter sleepedf caueeg
"""

import argparse
import glob
import json
import os
import re
import statistics as st

ROOT = "/lustre10/scratch/nourmaj"
LINE = re.compile(r"best test: ([0-9.]+),\s*best val: ([0-9.]+)")


def parse(path):
    out_dir, last, done = None, None, False
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("Output directory:"):
                out_dir = os.path.basename(line.split(":", 1)[1].strip().rstrip("/"))
            elif line.startswith("Stage ft |"):
                last = line
            elif line.startswith("Model saved to"):
                done = True
    if out_dir is None or last is None:
        return None
    m = LINE.search(last)
    epoch = re.search(r"Val \(\s*(\d+)", last)
    return {
        "dir": out_dir, "test": float(m.group(1)), "val": float(m.group(2)),
        "epochs": int(epoch.group(1)) + 1 if epoch else None, "done": done,
        "job": int(re.search(r"_(\d+)\.out$", path).group(1)), "log": os.path.basename(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", nargs="*", default=None, help="only groups containing these substrings")
    ap.add_argument("--optuna", action="store_true")
    args = ap.parse_args()

    runs = {}
    for path in glob.glob(os.path.join(ROOT, "reve_*_*.out")):
        if "optuna" in path or "smoke" in path or "diag" in path:
            continue
        r = parse(path)
        if r and (r["dir"] not in runs or r["job"] > runs[r["dir"]]["job"]):
            runs[r["dir"]] = r

    groups, unfinished = {}, []
    for r in runs.values():
        g = re.sub(r"_seed\d+$", "", r["dir"])
        if args.filter and not any(s in g for s in args.filter):
            continue
        if not r["done"]:
            unfinished.append(r)
            continue
        groups.setdefault(g, []).append(r)

    print(f"{'group':38s} {'n':>2s}  {'test mean ± std (%)':>22s}  {'val mean (%)':>12s}  per-run test (%)")
    for g in sorted(groups):
        rs = sorted(groups[g], key=lambda r: r["dir"])
        t = [100 * r["test"] for r in rs]
        v = [100 * r["val"] for r in rs]
        sd = st.stdev(t) if len(t) > 1 else 0.0
        seeds = ", ".join(f"{r['dir'].rsplit('_', 1)[-1]}={100 * r['test']:.2f}" for r in rs)
        print(f"{g:38s} {len(t):2d}  {st.mean(t):10.2f}% ± {sd:5.2f}%   {st.mean(v):10.2f}%   {seeds}")

    if unfinished:
        print("\n[unfinished / still running -- excluded]")
        for r in sorted(unfinished, key=lambda r: r["dir"]):
            print(f"  {r['dir']:38s} {r['log']}  (last epoch {r['epochs']}, best test so far {100 * r['test']:.2f}%)")

    if args.optuna:
        print("\n[Optuna best (val balanced acc)]")
        for f in sorted(glob.glob(os.path.join(ROOT, "reve_optuna_results", "*_best.json"))):
            d = json.load(open(f))
            if args.filter and not any(s in os.path.basename(f) for s in args.filter):
                continue
            p = d["best_params"]
            print(f"  {os.path.basename(f)[:-10]:38s} {100 * d['best_val_balanced_acc']:6.2f}%  "
                  f"lr={p['lr']:.2e} clip={p['grad_clip']} warmup={p['warmup_ratio']:.3f} trials={d['n_trials_run']}")


if __name__ == "__main__":
    main()
