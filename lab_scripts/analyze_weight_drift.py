"""How far did fine-tuning move REVE away from its pretrained checkpoint?

For every pretrained-arm production run (reve_outputs/*pretrained*seed*/model_best.pth),
compares the fine-tuned encoder weights with the original Hugging Face checkpoint and
reports relative drift ||W_ft - W_pt|| / ||W_pt||:
  - overall (all encoder weights)
  - patch embedding / positional-encoding module
  - per transformer block (early, middle, late)

Interpretation: a large pretrained-vs-random gain with LOW drift means the pretrained
features were reused almost as-is; HIGH drift means fine-tuning rewrote the backbone
and pretraining mainly provided a good starting point.

Usage (REVE env, from anywhere):
  python analyze_weight_drift.py                       # all pretrained runs
  python analyze_weight_drift.py --filter sleepedf caueeg
Writes weight_drift.csv next to the script's working directory.
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import torch

REVE_SRC = "/lustre10/scratch/nourmaj/REVE/src"
sys.path.insert(0, REVE_SRC)
from models.encoder import REVE  # noqa: E402

OUT_ROOT = "/lustre10/scratch/nourmaj/reve_outputs"
HF_CACHE = "/lustre10/scratch/nourmaj/hf_cache"


def rel(a, b):
    return (torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", nargs="*", default=None)
    args = ap.parse_args()

    encoder, _ = REVE.from_pretrained("brain-bzh/reve-base", cache_dir=HF_CACHE)
    pt = {k: v.float().cpu() for k, v in encoder.state_dict().items() if v.is_floating_point()}

    runs = sorted(glob.glob(os.path.join(OUT_ROOT, "*pretrained*", "model_best.pth")))
    if args.filter:
        runs = [r for r in runs if any(s in r for s in args.filter)]
    rows = []
    for path in runs:
        name = os.path.basename(os.path.dirname(path))
        sd = torch.load(path, map_location="cpu")
        ft = {k[len("encoder."):]: v.float() for k, v in sd.items() if k.startswith("encoder.")}
        common = [k for k in pt if k in ft and ft[k].shape == pt[k].shape]
        if not common:
            print(f"[skip] {name}: no matching encoder weights")
            continue

        def group_drift(keys):
            if not keys:
                return float("nan")
            a = torch.cat([ft[k].flatten() for k in keys])
            b = torch.cat([pt[k].flatten() for k in keys])
            return rel(a, b)

        blocks = defaultdict(list)
        for k in common:
            m = re.search(r"\.(\d+)\.", k)
            if m:
                blocks[int(m.group(1))].append(k)
        order = sorted(blocks)
        n = len(order)
        early = [k for i in order[: n // 3] for k in blocks[i]]
        mid = [k for i in order[n // 3: 2 * n // 3] for k in blocks[i]]
        late = [k for i in order[2 * n // 3:] for k in blocks[i]]
        embed = [k for k in common if not re.search(r"\.(\d+)\.", k)]

        row = {
            "run": name,
            "group": re.sub(r"_seed\d+$", "", name),
            "overall": group_drift(common),
            "embedding_pos": group_drift(embed),
            "early_blocks": group_drift(early),
            "middle_blocks": group_drift(mid),
            "late_blocks": group_drift(late),
            "n_tensors": len(common),
        }
        rows.append(row)
        print(f"{name:40s} overall {row['overall']:.4f} | embed/pos {row['embedding_pos']:.4f} | "
              f"early {row['early_blocks']:.4f} mid {row['middle_blocks']:.4f} late {row['late_blocks']:.4f}")

    if rows:
        with open("weight_drift.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[info] wrote weight_drift.csv ({len(rows)} runs)")


if __name__ == "__main__":
    main()
