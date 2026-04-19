"""
Phase 5 — leave-one-session-out cross-validation.

Reads every labeled CSV in recordings/, filters to ready=1 rows, then
for each session in turn: holds it out, trains a random forest on the
rest, and measures accuracy on the held-out session. Averages across
folds. This is the honest generalization metric for this dataset —
unlike a random row-level split, it cannot leak session-identity into
the test set.

Run: python3 train.py
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = ["knee_angle", "trunk_lean", "foot_offset", "cadence_spm"]
LABEL_COL = "label"
READY_COL = "ready"


def load_recordings(folder):
    """Concatenate every labeled session CSV into one DataFrame."""
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, "session_*.csv"))):
        df = pd.read_csv(path)
        if LABEL_COL not in df.columns:
            continue
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise SystemExit("No labeled session files found in recordings/.")
    return pd.concat(frames, ignore_index=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    df = load_recordings(os.path.join(here, "recordings"))

    # Quality filter: only trustworthy frames with all features present.
    df = df[df[READY_COL] == 1]
    df = df.dropna(subset=FEATURES + [LABEL_COL])

    sessions = sorted(df["source_file"].unique())
    print(f"Loaded {len(df)} ready rows across {len(sessions)} sessions.")
    print("\nSessions and their labels:")
    for s in sessions:
        rows = df[df["source_file"] == s]
        print(f"  {s:40s}  label={rows[LABEL_COL].iloc[0]:16s}  rows={len(rows)}")

    print(f"\nLabel distribution:")
    print(df[LABEL_COL].value_counts().to_string())

    # --- Leave-one-session-out cross-validation --------------------------
    print(f"\n{'=' * 72}")
    print(f"Leave-one-session-out cross-validation ({len(sessions)} folds)")
    print(f"{'=' * 72}")
    print(
        f"{'held-out session':42s} {'true':16s} {'rows':>5s}  "
        f"{'accuracy':>10s}"
    )

    fold_results = []
    all_true = []
    all_pred = []

    for held_out in sessions:
        train_df = df[df["source_file"] != held_out]
        test_df = df[df["source_file"] == held_out]

        clf = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
        )
        clf.fit(train_df[FEATURES], train_df[LABEL_COL])
        preds = clf.predict(test_df[FEATURES])

        acc = (preds == test_df[LABEL_COL]).mean()
        true_label = test_df[LABEL_COL].iloc[0]
        fold_results.append(
            {"session": held_out, "label": true_label, "acc": acc}
        )
        all_true.extend(test_df[LABEL_COL].tolist())
        all_pred.extend(preds.tolist())

        print(
            f"  {held_out:40s} {true_label:16s} {len(test_df):>5d}  {acc:>10.3f}"
        )

    print(f"{'=' * 72}")
    accs = [r["acc"] for r in fold_results]
    print(f"Mean accuracy across folds:   {np.mean(accs):.3f}")
    print(f"Median:                       {np.median(accs):.3f}")
    print(f"Min / Max:                    {min(accs):.3f} / {max(accs):.3f}")

    # Per-label mean accuracy — shows which classes generalize.
    print("\nPer-label mean accuracy:")
    for label in sorted(df[LABEL_COL].unique()):
        label_accs = [r["acc"] for r in fold_results if r["label"] == label]
        print(f"  {label:16s}  n={len(label_accs)}   mean={np.mean(label_accs):.3f}")

    # --- Aggregate confusion matrix across all held-out predictions -------
    print("\n=== Aggregate confusion matrix (all folds concatenated) ===")
    labels = sorted(df[LABEL_COL].unique())
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    header = " " * 17 + " ".join(f"{c:>16}" for c in labels)
    print(header)
    for i, label in enumerate(labels):
        row = " ".join(f"{cm[i, j]:>16d}" for j in range(len(labels)))
        print(f"{label:>16} {row}")

    print("\n=== Classification report (aggregated) ===")
    print(classification_report(all_true, all_pred, digits=3))

    # --- Feature importances from a model trained on all data ------------
    clf_final = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
    )
    clf_final.fit(df[FEATURES], df[LABEL_COL])
    print("\n=== Feature importances (model trained on all data) ===")
    importances = sorted(
        zip(FEATURES, clf_final.feature_importances_), key=lambda kv: -kv[1]
    )
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {name:14s}  {imp:.3f}  {bar}")


if __name__ == "__main__":
    main()
