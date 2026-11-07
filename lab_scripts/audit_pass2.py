"""Dataset audit, pass 2 (read-only, CPU).

1. Amplitude statistics over MANY windows per split (pkl folders): per-window |x| 95th percentile,
   reported as median and 5-95% range across windows -> detects units (V / uV / normalized) and
   scale mismatches between splits.
2. Better subject/session parsing (per-dataset rules), subject overlap between splits.
3. Folders in "seq/labels" layout (seq/<rec>.npy, labels/<rec>.npy, optional split.json).
4. LMDB folders (data.mdb): number of entries, record structure, label counts, split keys.

Usage:
  python audit_pass2.py                 # everything below
  python audit_pass2.py --only isruc_biot_processed isruc_processed_80h
Writes /scratch/nourmaj/dataset_audit/dataset_audit_pass2.md (+ .json). Modifies nothing.
"""
import argparse
import json
import os
import pickle
import random
import re
from collections import Counter

import numpy as np

ROOT = "/scratch/nourmaj"
PKL_DATASETS = {
    # folder: regex whose group(1) is the subject (and session, where it matters)
    "chb-mit-processed-80h": r"^(chb\d+)",
    "ds004504_processed": r"^(sub-\d+)",
    "karaone_processed": r"^([A-Z]+\d+)",
    "japaneeg_processed_80h": r"^(sub-\d+_ses-\d+)",       # subject + session
    "dream_processed": r"^(s\d+)",
    "yang2025_processed": r"^(sub\d+)",
    "isruc_biot_processed": r"^ISRUC-(group\d+-\d+)",
    "thoughtviz_processed": r"^([a-z]+)",
}
SEQ_DATASETS = ["isruc_processed_80h", "caueeg_reve_processed", "sleepedf_reve_processed"]
LMDB_DATASETS = ["stress_reve_processed"]
SPLITS = ["train", "val", "test"]


def amp_stats(arrs):
    p95 = np.array([np.percentile(np.abs(a), 95) for a in arrs])
    return {"n": len(p95), "abs_p95_median": float(np.median(p95)),
            "abs_p95_5pct": float(np.percentile(p95, 5)), "abs_p95_95pct": float(np.percentile(p95, 95)),
            "frac_flat": float(np.mean(p95 < 1e-12))}


def audit_pkl(name, rx, n_amp, md, rep):
    folder = os.path.join(ROOT, name)
    if not os.path.isdir(folder):
        md.append(f"## {name}\nnot found\n"); return
    info, subj = {}, {}
    md.append(f"## {name}\n\n| split | subjects | windows | abs p95 median [5–95%] across {n_amp} windows | flat |")
    md.append("|---|---|---|---|---|")
    for sp in SPLITS:
        d = os.path.join(folder, sp)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".pkl"))
        ids = Counter((re.match(rx, f).group(1) if re.match(rx, f) else "?") for f in files)
        subj[sp] = set(ids)
        pick = random.Random(0).sample(files, min(n_amp, len(files)))
        arrs = []
        for f in pick:
            with open(os.path.join(d, f), "rb") as fh:
                arrs.append(np.asarray(pickle.load(fh)["X"], dtype=np.float64))
        st = amp_stats(arrs)
        info[sp] = {"n_windows": len(files), "n_subjects": len(ids), "subjects": sorted(ids), "amp": st}
        md.append(f"| {sp} | {len(ids)} | {len(files):,} | {st['abs_p95_median']:.3g} "
                  f"[{st['abs_p95_5pct']:.3g} – {st['abs_p95_95pct']:.3g}] | {st['frac_flat']:.1%} |")
    ov = {f"{a}&{b}": sorted(subj[a] & subj[b]) for i, a in enumerate(subj) for b in list(subj)[i + 1:] if subj[a] & subj[b]}
    info["overlap"] = ov
    md.append(f"\n- subject/session IDs: " + "; ".join(f"{sp}: {', '.join(info[sp]['subjects'][:15])}"
                                                      + (' …' if len(info[sp]['subjects']) > 15 else '')
                                                      for sp in SPLITS if sp in info))
    md.append(f"- **overlap between splits:** {ov if ov else 'none'}\n")
    rep[name] = info


def audit_seq(name, n_amp, md, rep):
    folder = os.path.join(ROOT, name)
    if not os.path.isdir(folder):
        md.append(f"## {name}\nnot found\n"); return
    info = {}
    md.append(f"## {name} (seq/labels layout)\n")
    # structure
    tree = []
    for dp, dn, fn in os.walk(folder):
        depth = dp[len(folder):].count(os.sep)
        if depth <= 2:
            exts = Counter(os.path.splitext(f)[1] for f in fn)
            tree.append(f"`{os.path.relpath(dp, folder)}`: {len(dn)} dirs, {dict(exts)}; e.g. {sorted(fn)[:3]}")
    md += ["- " + t for t in tree[:15]]
    # split.json / summary.json
    for js in ["split.json", "summary.json"]:
        p = os.path.join(folder, js)
        if os.path.exists(p):
            obj = json.load(open(p))
            if isinstance(obj, dict):
                short = {k: (f"list[{len(v)}] e.g. {v[:3]}" if isinstance(v, list) else v) for k, v in obj.items()}
            else:
                short = str(obj)[:500]
            info[js] = short
            md.append(f"- **{js}:** `{json.dumps(short, default=str)[:1500]}`")
    # per-split label counts + amplitudes if seq/<rec>.npy & labels/<rec>.npy and split.json lists recordings
    seq_dir, lab_dir = os.path.join(folder, "seq"), os.path.join(folder, "labels")
    split = json.load(open(os.path.join(folder, "split.json"))) if os.path.exists(os.path.join(folder, "split.json")) else None
    recs_by_split = {}
    if isinstance(split, dict):
        for sp in SPLITS:
            for k in (sp, f"{sp}_split", f"{sp}_recordings", f"{sp}_subjects"):
                if isinstance(split.get(k), list):
                    recs_by_split[sp] = split[k]; break
    elif os.path.isdir(seq_dir) and all(os.path.isdir(os.path.join(seq_dir, s)) for s in SPLITS):
        recs_by_split = {sp: [os.path.join(sp, f) for f in sorted(os.listdir(os.path.join(seq_dir, sp)))] for sp in SPLITS}
    if recs_by_split and os.path.isdir(lab_dir):
        md.append(f"\n| split | recordings | epochs | label counts | seq shape | abs p95 median [5–95%] |")
        md.append("|---|---|---|---|---|---|")
        for sp, recs in recs_by_split.items():
            labs, shapes, arrs = Counter(), set(), []
            rng = random.Random(0)
            for r in recs:
                stem = os.path.splitext(str(r))[0]
                lp = next((p for p in [os.path.join(lab_dir, stem + ".npy"), os.path.join(lab_dir, os.path.basename(stem) + ".npy")] if os.path.exists(p)), None)
                sp_ = next((p for p in [os.path.join(seq_dir, stem + ".npy"), os.path.join(seq_dir, os.path.basename(stem) + ".npy")] if os.path.exists(p)), None)
                if lp:
                    labs.update(np.load(lp).ravel().tolist())
                if sp_ and len(arrs) < n_amp:
                    x = np.load(sp_, mmap_mode="r")
                    shapes.add(tuple(x.shape[1:]) if x.ndim >= 3 else tuple(x.shape))
                    for i in rng.sample(range(len(x)), min(5, len(x))):
                        arrs.append(np.asarray(x[i], dtype=np.float64))
            st = amp_stats(arrs) if arrs else {}
            info[sp] = {"recordings": len(recs), "labels": dict(labs), "seq_shapes": [list(s) for s in shapes], "amp": st}
            md.append(f"| {sp} | {len(recs)} | {sum(labs.values()):,} | {dict(sorted(labs.items()))} | {sorted(shapes)[:2]} | "
                      + (f"{st['abs_p95_median']:.3g} [{st['abs_p95_5pct']:.3g} – {st['abs_p95_95pct']:.3g}]" if st else "—") + " |")
        sets = {sp: set(map(str, r)) for sp, r in recs_by_split.items()}
        ov = {f"{a}&{b}": sorted(sets[a] & sets[b])[:10] for i, a in enumerate(sets) for b in list(sets)[i + 1:] if sets[a] & sets[b]}
        md.append(f"\n- **recording overlap between splits:** {ov if ov else 'none'}")
    else:
        md.append("\n- could not map recordings to splits automatically (see structure above)")
    md.append("")
    rep[name] = info


def audit_lmdb(name, n_amp, md, rep):
    folder = os.path.join(ROOT, name)
    md.append(f"## {name} (LMDB)\n")
    try:
        import lmdb
    except ImportError:
        md.append("- `lmdb` not installed in this env (use the reve-gpu env)\n"); return
    env = lmdb.open(folder, readonly=True, lock=False, readahead=False, meminit=False)
    info = {"entries": env.stat()["entries"]}
    with env.begin() as txn:
        keys_obj = txn.get(b"__keys__")
        if keys_obj is not None:
            ko = pickle.loads(keys_obj)
            info["__keys__"] = {k: len(v) for k, v in ko.items()} if isinstance(ko, dict) else str(type(ko))
            md.append(f"- `__keys__`: {info['__keys__']}")
            split_keys = ko if isinstance(ko, dict) else {}
        else:
            split_keys = {}
        cur = txn.cursor()
        first = [(k, v) for (k, v), _ in zip(cur, range(3)) if k != b"__keys__"]
        for k, v in first[:1]:
            rec = pickle.loads(v)
            desc = {kk: (list(np.asarray(vv).shape) if hasattr(vv, "__len__") and not isinstance(vv, str) else vv)
                    for kk, vv in (rec.items() if isinstance(rec, dict) else [("value", rec)])}
            md.append(f"- first key `{k[:80]}` → {desc}")
        for sp, ks in split_keys.items():
            labs, arrs = Counter(), []
            for kk in random.Random(0).sample(list(ks), min(n_amp, len(ks))):
                rec = pickle.loads(txn.get(kk.encode() if isinstance(kk, str) else kk))
                y = rec.get("label", rec.get("y")) if isinstance(rec, dict) else None
                labs[str(np.asarray(y).item() if y is not None and np.size(y) == 1 else y)] += 1
                x = rec.get("sample", rec.get("X")) if isinstance(rec, dict) else None
                if x is not None:
                    arrs.append(np.asarray(x, dtype=np.float64))
            st = amp_stats(arrs) if arrs else {}
            info[sp] = {"n": len(ks), "labels_in_sample": dict(labs), "amp": st}
            md.append(f"- **{sp}:** {len(ks):,} records; labels (sample of {min(n_amp, len(ks))}): {dict(labs)}; "
                      + (f"abs p95 median {st['abs_p95_median']:.3g}" if st else ""))
    md.append("")
    rep[name] = info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--n_amp", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset_audit"))
    a = ap.parse_args()
    want = lambda n: a.only is None or n in a.only
    md, rep = ["# Dataset audit — pass 2\n"], {}
    for n, rx in PKL_DATASETS.items():
        if want(n): audit_pkl(n, rx, a.n_amp, md, rep); print("[pass2] done", n, flush=True)
    for n in SEQ_DATASETS:
        if want(n): audit_seq(n, a.n_amp, md, rep); print("[pass2] done", n, flush=True)
    for n in LMDB_DATASETS:
        if want(n): audit_lmdb(n, a.n_amp, md, rep); print("[pass2] done", n, flush=True)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "dataset_audit_pass2.md"), "w").write("\n".join(md))
    json.dump(rep, open(os.path.join(a.out, "dataset_audit_pass2.json"), "w"), indent=1, default=str)
    print(f"[pass2] wrote {a.out}/dataset_audit_pass2.md")


if __name__ == "__main__":
    main()
