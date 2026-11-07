"""Quick linear baselines on BCI2020 Track 3 to locate where decodable
information lives (and whether DC removal could explain the gap to the
published ~56%).

Features, per trial and per channel, on the LAST 3 s (768 samples at 256 Hz) of
the RAW data (bci2020_processed pkl files, uV, DC offset included):
  - dc:      channel mean                      (slow drift / offset only)
  - logvar:  log variance after mean removal  (signal power)
  - dc+logvar

Classifier: StandardScaler + multinomial logistic regression.
  - pooled: one model on all 15 subjects' train trials -> val / test
  - per-subject: one model per subject (300 train) -> that subject's val / test

Interpretation:
  - dc features well above chance (20%)  -> class information sits in slow
    offsets/drift; removing DC (as our preprocessing does) throws it away,
    and the published number may partly reflect it
  - only logvar above chance              -> genuine oscillatory information
  - everything at chance                  -> neither explains the gap

Usage (REVE env, which has sklearn):
  python check_bci2020_baselines.py /lustre10/scratch/nourmaj/bci2020_processed
"""

import glob
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PER_SUBJECT = {"train": 300, "val": 50, "test": 50}


def load(path, split):
    files = sorted(glob.glob(os.path.join(path, split, "trial_*.pkl")))

    def one(f):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        return np.asarray(d["X"], dtype=np.float32)[:, -768:], int(d["y"])

    with ThreadPoolExecutor(32) as ex:
        items = list(ex.map(one, files))
    X = np.stack([x for x, _ in items])
    y = np.array([lab for _, lab in items])
    s = np.repeat(np.arange(15), PER_SUBJECT[split])
    return X, y, s


def feats(X):
    dc = X.mean(axis=2)
    lv = np.log(X.var(axis=2) + 1e-6)
    return {"dc": dc, "logvar": lv, "dc+logvar": np.concatenate([dc, lv], axis=1)}


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def main(path):
    data = {sp: load(path, sp) for sp in PER_SUBJECT}
    F = {sp: feats(data[sp][0]) for sp in PER_SUBJECT}
    print("chance = 0.200 (5 balanced classes)\n")
    print(f"{'features':12s} {'pooled val':>10s} {'pooled test':>11s} {'per-subj val':>13s} {'per-subj test':>14s}")
    for name in ["dc", "logvar", "dc+logvar"]:
        m = clf().fit(F["train"][name], data["train"][1])
        pv = balanced_accuracy_score(data["val"][1], m.predict(F["val"][name]))
        pt = balanced_accuracy_score(data["test"][1], m.predict(F["test"][name]))
        sv, st = [], []
        for s in range(15):
            tr = data["train"][2] == s
            ms = clf().fit(F["train"][name][tr], data["train"][1][tr])
            for sp, acc in (("val", sv), ("test", st)):
                k = data[sp][2] == s
                acc.append(balanced_accuracy_score(data[sp][1][k], ms.predict(F[sp][name][k])))
        print(f"{name:12s} {pv:10.3f} {pt:11.3f} {np.mean(sv):13.3f} {np.mean(st):14.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
