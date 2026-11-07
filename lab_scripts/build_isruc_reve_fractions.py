"""Build REVE-format ISRUC fraction folders (isruc_processed_80h_{50,25}pct) that
contain EXACTLY the same training epochs as BIOT's isruc_biot_processed_{50,25}pct.

  REVE  : <root>/seq/ISRUC-group1-<n>/ISRUC-group1-<subj>-<seq>.npy  (20, 6, 6000)
          <root>/labels/ISRUC-group1-<n>/<same name>.npy              (20,)
          loader split by folder index: 1-80 train, 81-90 val, 91-100 test
  BIOT  : <biot>/<split>/ISRUC-group1-<subj>-<seq>_e<k>.pkl          one 30 s epoch

For each fraction:
  - train subject folders (1-80): each sequence file is rewritten with only the
    epochs BIOT kept for that sequence (sequences with none kept are omitted;
    every subject folder is still created so the loader's 1..100 loop works)
  - val/test subject folders (81-100): symlinked unchanged
Checks: full REVE train == full BIOT train (stems + epoch count), every kept BIOT
epoch maps to a REVE train sequence, final counts == BIOT counts.

Usage:
  python build_isruc_reve_fractions.py --reve_root /lustre10/scratch/nourmaj/isruc_processed_80h \
      --biot_prefix /lustre10/scratch/nourmaj/isruc_biot_processed --fractions 50 25
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np

PAT = re.compile(r"^(ISRUC-group1-\d+-\d+)_e(\d+)\.pkl$")


def biot_epochs(split_dir):
    kept = defaultdict(set)
    for f in os.listdir(split_dir):
        m = PAT.match(f)
        if m:
            kept[m.group(1)].add(int(m.group(2)))
    return kept


def reve_files(root, lo, hi):
    out = {}  # stem -> (subject folder, filename)
    for n in range(lo, hi + 1):
        folder = f"ISRUC-group1-{n}"
        d = os.path.join(root, "seq", folder)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            out[f[:-4]] = (folder, f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reve_root", required=True)
    p.add_argument("--biot_prefix", required=True)
    p.add_argument("--fractions", nargs="+", default=["50", "25"])
    args = p.parse_args()

    tr = reve_files(args.reve_root, 1, 80)
    n_val = sum(np.load(os.path.join(args.reve_root, "labels", fo, fi)).shape[0]
                for fo, fi in reve_files(args.reve_root, 81, 90).values())
    n_test = sum(np.load(os.path.join(args.reve_root, "labels", fo, fi)).shape[0]
                 for fo, fi in reve_files(args.reve_root, 91, 100).values())
    n_train = sum(np.load(os.path.join(args.reve_root, "labels", fo, fi)).shape[0] for fo, fi in tr.values())
    print(f"[REVE full] train seqs {len(tr)} / epochs {n_train} | val epochs {n_val} | test epochs {n_test}")

    full = biot_epochs(os.path.join(args.biot_prefix, "train"))
    n_full = sum(len(v) for v in full.values())
    print(f"[BIOT full] train seqs {len(full)} / epochs {n_full}")
    assert set(full) == set(tr), "REVE train sequences != BIOT train sequences"
    assert n_full == n_train, "REVE and BIOT full train epoch counts differ"
    for split, n in (("val", n_val), ("test", n_test)):
        nb = len(os.listdir(os.path.join(args.biot_prefix, split)))
        assert nb == n, f"{split}: REVE {n} epochs vs BIOT {nb}"
    print("[check] full train/val/test match BIOT exactly")

    for frac in args.fractions:
        out = f"{args.reve_root}_{frac}pct"
        kept = biot_epochs(os.path.join(f"{args.biot_prefix}_{frac}pct", "train"))
        missing = set(kept) - set(tr)
        assert not missing, f"{frac}%: {len(missing)} BIOT sequences not in REVE train, e.g. {sorted(missing)[:3]}"
        for part in ("seq", "labels"):
            os.makedirs(os.path.join(out, part), exist_ok=True)
            for n in range(1, 81):
                os.makedirs(os.path.join(out, part, f"ISRUC-group1-{n}"), exist_ok=True)
            for n in range(81, 101):
                dst = os.path.join(out, part, f"ISRUC-group1-{n}")
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(args.reve_root, part, f"ISRUC-group1-{n}"), dst)

        n_ep, n_seq = 0, 0
        for stem, idx in kept.items():
            folder, fname = tr[stem]
            idx = sorted(idx)
            x = np.load(os.path.join(args.reve_root, "seq", folder, fname))[idx]
            y = np.load(os.path.join(args.reve_root, "labels", folder, fname))[idx]
            np.save(os.path.join(out, "seq", folder, fname), x)
            np.save(os.path.join(out, "labels", folder, fname), y)
            n_ep += len(idx)
            n_seq += 1
        n_biot = sum(len(v) for v in kept.values())
        assert n_ep == n_biot
        print(f"[{frac}%] wrote {n_seq} train sequences / {n_ep} epochs (BIOT: {n_biot}); "
              f"val {n_val} + test {n_test} linked unchanged -> {out}")


if __name__ == "__main__":
    main()
