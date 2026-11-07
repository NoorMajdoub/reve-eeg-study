import re, glob, os

DATASETS = ["chbmit","clinical","dream","japaneeg","karaone","thoughtviz","yang2025"]
ARMS = ["pretrained","random"]

def latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    # pick the file with the highest job id (last _NNNNNNN.out)
    def jobid(f):
        m = re.search(r"_(\d+)\.out$", f)
        return int(m.group(1)) if m else -1
    return sorted(files, key=jobid)[-1]

def parse_log(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    val_lines = [l for l in lines if l.startswith("Stage ft | Val")]
    if not val_lines:
        return {"status": "NO_TRAINING_LINES"}
    last = val_lines[-1]
    m = re.search(r"best test:\s*([\d.]+),best val:\s*([\d.]+),val:\s*([\d.]+),\s*LR\s*([\d.]+).*train_loss:\s*([\d.]+)", last)
    if not m:
        return {"status": "PARSE_FAIL"}
    best_test, best_val, val, lr, train_loss = m.groups()
    collapsed = (float(best_val) <= 0.51 and float(train_loss) > 3.0)
    finished_full = "patience limit" not in "".join(lines)
    return {
        "status": "COLLAPSED" if collapsed else "OK",
        "best_test": best_test,
        "best_val": best_val,
        "last_train_loss": train_loss,
        "ran_full_50_epochs": finished_full,
        "n_val_epochs_logged": len(val_lines),
    }

rows = []
for ds in DATASETS:
    for arm in ARMS:
        combos = [
            ("25pct_seed42", f"reve_{ds}_{arm}_25pct_[0-9]*.out"),
            ("25pct_seed2",  f"reve_{ds}_{arm}_25pct_seed2_*.out"),
            ("50pct",        f"reve_{ds}_{arm}_50pct_*.out"),
        ]
        for label, pattern in combos:
            f = latest_file(pattern)
            result = parse_log(f)
            row = {"dataset": ds, "arm": arm, "split": label, "file": f or "MISSING"}
            row.update(result or {"status": "MISSING"})
            rows.append(row)

# print a simple table
header = ["dataset","arm","split","status","best_val","best_test","last_train_loss","ran_full_50_epochs"]
print("\t".join(header))
for r in rows:
    print("\t".join(str(r.get(h,"")) for h in header))
