"""
Option B — per-frame classifier that includes BOTH:
  - the original per-frame features (knee_angle, trunk_lean, foot_offset, cadence_spm), AND
  - for every frame, the features at the most recent foot-contact
    (direction-corrected foot_offset_at_contact, knee_angle_at_contact,
    trunk_lean_at_contact, cadence_at_contact).

Each frame still becomes one training sample, so we keep Phase 5's
sample count, but the model also sees the last-contact context that
made Phase 6 per-sample more informative.

Run: python3 train_combined.py
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from train_phase6 import detect_contacts, MIN_CADENCE, MIN_DIRECTION_DX

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(HERE, "recordings")

FRAME_FEATS = ["knee_angle", "trunk_lean", "foot_offset", "cadence_spm"]
CONTACT_FEATS = [
    "foot_offset_at_contact", "knee_angle_at_contact",
    "trunk_lean_at_contact", "cadence_at_contact",
]
COMBINED_FEATS = FRAME_FEATS + CONTACT_FEATS


def add_contact_features(df):
    """Attach last-contact features to every frame in a session."""
    df = df.sort_values("frame_index").reset_index(drop=True)
    n = len(df)
    ankle_y = df["ankle_y"].to_numpy()
    hip_x = df["hip_x"].to_numpy()
    foot_offset = df["foot_offset"].to_numpy()
    knee_angle = df["knee_angle"].to_numpy()
    trunk_lean = df["trunk_lean"].to_numpy()
    cadence = df["cadence_spm"].to_numpy()

    peaks = set(detect_contacts(ankle_y))

    last_fo = last_ka = last_tl = last_cad = np.nan
    contact_fo = np.full(n, np.nan)
    contact_ka = np.full(n, np.nan)
    contact_tl = np.full(n, np.nan)
    contact_cad = np.full(n, np.nan)

    for i in range(n):
        if i in peaks:
            lo = max(0, i - 5)
            hi = min(n - 1, i + 5)
            dx = hip_x[hi] - hip_x[lo]
            cad = cadence[i]
            # Only accept this peak as a real running contact.
            if (abs(dx) > MIN_DIRECTION_DX and not pd.isna(cad) and cad >= MIN_CADENCE):
                ds = 1.0 if dx > 0 else -1.0
                last_fo = foot_offset[i] * ds
                last_ka = knee_angle[i]
                last_tl = trunk_lean[i]
                last_cad = cad
        contact_fo[i] = last_fo
        contact_ka[i] = last_ka
        contact_tl[i] = last_tl
        contact_cad[i] = last_cad

    df["foot_offset_at_contact"] = contact_fo
    df["knee_angle_at_contact"] = contact_ka
    df["trunk_lean_at_contact"] = contact_tl
    df["cadence_at_contact"] = contact_cad
    return df


def load_with_contact_features():
    frames = []
    for path in sorted(glob.glob(os.path.join(RECORDINGS_DIR, "session_*.csv"))):
        df = pd.read_csv(path)
        if "ankle_y" not in df.columns or "label" not in df.columns:
            continue
        df = add_contact_features(df)
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise SystemExit(
            "No Phase 6-schema sessions found. Run formAnalyzer.py to record some."
        )
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_with_contact_features()
    df = df[df["ready"] == 1]
    # Drop rows before we've seen the first valid contact (contact features NaN).
    df = df.dropna(subset=COMBINED_FEATS + ["label"]).reset_index(drop=True)

    sessions = sorted(df["source_file"].unique())
    print(f"Loaded {len(df)} usable rows from {len(sessions)} sessions.\n")
    print("Rows per session:")
    for s in sessions:
        sub = df[df["source_file"] == s]
        print(f"  {s:40s}  label={sub['label'].iloc[0]:16s}  rows={len(sub)}")

    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    print(f"\n{'=' * 78}")
    print(f"Combined-features LOSO CV  ({len(sessions)} folds)")
    print(f"{'=' * 78}")
    print(f"  {'held-out session':40s} {'true':16s} {'rows':>6s} {'accuracy':>10s}")

    fold_results = []
    all_true, all_pred = [], []

    for held in sessions:
        train = df[df["source_file"] != held]
        test = df[df["source_file"] == held]

        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(train[COMBINED_FEATS], train["label"])
        preds = clf.predict(test[COMBINED_FEATS])

        acc = (preds == test["label"]).mean()
        label = test["label"].iloc[0]
        fold_results.append(
            {"session": held, "label": label, "acc": acc, "n": len(test)}
        )
        all_true.extend(test["label"].tolist())
        all_pred.extend(preds.tolist())
        print(f"  {held:40s} {label:16s} {len(test):>6d} {acc:>10.3f}")

    print(f"{'=' * 78}")
    accs = [r["acc"] for r in fold_results]
    print(f"Mean accuracy:  {np.mean(accs):.3f}")
    print(f"Median:         {np.median(accs):.3f}")
    print(f"Min / Max:      {min(accs):.3f} / {max(accs):.3f}")

    print("\nPer-label mean accuracy:")
    for label in sorted(df["label"].unique()):
        la = [r["acc"] for r in fold_results if r["label"] == label]
        print(f"  {label:16s}  n={len(la)}   mean={np.mean(la):.3f}")

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    print("\n=== Confusion matrix ===")
    print(" " * 17 + " ".join(f"{c:>16}" for c in labels))
    for i, label in enumerate(labels):
        row = " ".join(f"{cm[i, j]:>16d}" for j in range(len(labels)))
        print(f"{label:>16} {row}")

    print("\n=== Classification report ===")
    print(classification_report(all_true, all_pred, digits=3))

    clf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_final.fit(df[COMBINED_FEATS], df["label"])
    print("=== Feature importances ===")
    for name, imp in sorted(
        zip(COMBINED_FEATS, clf_final.feature_importances_), key=lambda kv: -kv[1]
    ):
        bar = "#" * int(imp * 50)
        print(f"  {name:26s}  {imp:.3f}  {bar}")


if __name__ == "__main__":
    main()
