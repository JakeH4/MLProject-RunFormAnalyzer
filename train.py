"""
Phase 5 — baseline random-forest classifier for running form.

Reads every labeled CSV in recordings/, filters to ready=1 rows, splits
into train/test, fits a random forest, and reports accuracy, a confusion
matrix, and feature importances.

Run: python3 train.py
"""

import glob
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURES = ["knee_angle", "trunk_lean", "foot_offset", "cadence_spm"]
LABEL_COL = "label"
READY_COL = "ready"


def load_recordings(folder):
    """Concatenate every labeled session CSV into one DataFrame.

    Pre-labeling-feature sessions (no `label` column) are skipped. A
    `source_file` column is added so later code can see which session each
    row came from.
    """
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
    print(f"Loaded {len(df)} rows from {df['source_file'].nunique()} labeled sessions.")

    # Only use frames where hip/knee/ankle were confidently tracked AND where
    # every feature is present (cadence is missing during the first ~1s while
    # the rolling buffer fills).
    df = df[df[READY_COL] == 1]
    df = df.dropna(subset=FEATURES + [LABEL_COL])
    print(f"After filtering to ready=1 with all features: {len(df)} rows.")

    print("\nLabel distribution:")
    print(df[LABEL_COL].value_counts().to_string())

    X = df[FEATURES]
    y = df[LABEL_COL]

    # Stratified random split holds each label's share constant in both
    # train and test sets. random_state=42 makes it reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\nTrain: {len(X_train)} rows   Test: {len(X_test)} rows")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Test-set performance ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion matrix (rows = true, columns = predicted):")
    labels = sorted(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header = " " * 17 + " ".join(f"{c:>16}" for c in labels)
    print(header)
    for i, label in enumerate(labels):
        row = " ".join(f"{cm[i, j]:>16d}" for j in range(len(labels)))
        print(f"{label:>16} {row}")

    print("\nFeature importances (how often the forest split on each feature):")
    importances = sorted(
        zip(FEATURES, clf.feature_importances_), key=lambda kv: -kv[1]
    )
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {name:14s}  {imp:.3f}  {bar}")

    print(
        "\n[!] Caveat: with only one session per label, the train/test split "
        "is row-level within sessions. Reported accuracy is an upper bound; "
        "true generalization needs more sessions per class."
    )


if __name__ == "__main__":
    main()
