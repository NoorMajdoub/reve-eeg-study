"""
REVE-format ISRUC dataset properties report.

Adapted from dataset_properties_report.py (the BIOT-format checker) for
REVE's isruc_processed_80h layout: per-subject seq/ and labels/ folders of
.npy files (see downstream_tasks/dataloader_isruc.py), which is a totally
different storage format from BIOT's per-trial .pkl files -- the original
script cannot read this data at all.

Reproduces dataloader_isruc.py's own train/val/test split logic exactly
(subject index < 80 -> train, < 90 -> val, else test; subjects are
1-indexed ISRUC-group1-1 .. ISRUC-group1-100) so the reported counts match
what REVE will actually load at train time.

Also actively checks, per subject, that the seq/ and labels/ file counts
match and that each seq/label .npy pair has matching shapes along axis 0 --
this is the same kind of check that would have caught our transfer being
incomplete (folders present, .npy files missing) much earlier, since it
flags 0 usable trials or file-count mismatches explicitly instead of
silently producing an empty dataset.

Needs only numpy -- no torch/hydra/REVE imports required, so it can run
standalone anywhere, including directly on a login node.

Run directly on the cluster: python3 reve_isruc_properties_report.py
"""
import os

import numpy as np

ROOT = "/scratch/nourmaj/isruc_processed_80h"
# Matches the sampling rate already confirmed for the BIOT-format copy of
# ISRUC (isruc_biot_processed: 6000 samples/trial @ 200Hz = 30.000s trials,
# consistent with configs/task/isruc.yaml's duration: 30). Not independently
# re-derived here beyond that cross-check -- flagged if it looks wrong.
SAMPLING_RATE_HZ = 200

SPLIT1 = 80  # subjects [0, SPLIT1) -> train
SPLIT2 = 90  # subjects [SPLIT1, SPLIT2) -> val, [SPLIT2, 100) -> test


def subject_split(subject_num):
    idx = subject_num - 1  # subject_num is 1-indexed (ISRUC-group1-1 .. -100)
    if idx < SPLIT1:
        return "train"
    elif idx < SPLIT2:
        return "val"
    else:
        return "test"


def main():
    seq_dir = os.path.join(ROOT, "seq")
    label_dir = os.path.join(ROOT, "labels")

    split_trials = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": {}, "val": {}, "test": {}}
    trial_duration_sec = None
    issues = []

    for subject_num in range(1, 101):
        subj_seq_dir = os.path.join(seq_dir, f"ISRUC-group1-{subject_num}")
        subj_label_dir = os.path.join(label_dir, f"ISRUC-group1-{subject_num}")

        if not os.path.isdir(subj_seq_dir):
            issues.append(f"subject {subject_num}: MISSING seq dir {subj_seq_dir}")
            continue
        if not os.path.isdir(subj_label_dir):
            issues.append(f"subject {subject_num}: MISSING labels dir {subj_label_dir}")
            continue

        split = subject_split(subject_num)
        seq_files = sorted(os.listdir(subj_seq_dir))
        label_files = sorted(os.listdir(subj_label_dir))

        if len(seq_files) == 0 or len(label_files) == 0:
            issues.append(
                f"subject {subject_num}: EMPTY -- {len(seq_files)} seq files, {len(label_files)} label files"
            )
            continue
        if len(seq_files) != len(label_files):
            issues.append(
                f"subject {subject_num}: file count mismatch -- {len(seq_files)} seq files vs {len(label_files)} label files"
            )

        for seq_fname, label_fname in zip(seq_files, label_files):
            seq_path = os.path.join(subj_seq_dir, seq_fname)
            label_path = os.path.join(subj_label_dir, label_fname)
            seqs = np.load(seq_path)
            labels = np.load(label_path)

            if seqs.shape[0] != labels.shape[0]:
                issues.append(
                    f"subject {subject_num} {seq_fname}/{label_fname}: shape mismatch -- seqs {seqs.shape} vs labels {labels.shape}"
                )
                continue

            if trial_duration_sec is None:
                n_timepoints = seqs.shape[-1]
                trial_duration_sec = n_timepoints / SAMPLING_RATE_HZ
                print(
                    f"Detected per-trial shape = {seqs.shape[1:]}, assuming "
                    f"sampling_rate={SAMPLING_RATE_HZ}Hz -> trial duration = {trial_duration_sec:.3f}s"
                )
                print()

            split_trials[split] += seqs.shape[0]
            for lbl in labels.reshape(-1):
                label_counts[split][int(lbl)] = label_counts[split].get(int(lbl), 0) + 1

    if trial_duration_sec is None:
        print("!!! No usable trials found at all -- dataset is effectively empty. Nothing to report.")
    else:
        for split in ["train", "val", "test"]:
            n = split_trials[split]
            hours = n * trial_duration_sec / 3600
            print(f">>> {split.upper()} split: {n} samples / {hours:.2f} hours <<<")
            print(f"    label counts: {dict(sorted(label_counts[split].items()))}")

        total = sum(split_trials.values())
        total_hours = total * trial_duration_sec / 3600
        print(f"\nTOTAL (train+val+test): {total} samples / {total_hours:.2f} hours")

    if issues:
        print(f"\n!!! {len(issues)} issue(s) found:")
        for msg in issues[:30]:
            print(f"  - {msg}")
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more")
    else:
        print("\nNo issues found -- all 100 subjects present, seq/label counts and shapes line up cleanly.")


if __name__ == "__main__":
    main()