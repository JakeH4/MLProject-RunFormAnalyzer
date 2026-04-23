"""
Phase 6 — stride-level classifier with contact-aligned features.

For each labeled session in recordings/:
  1. Detect foot-contact frames as local maxima in ankle_y.
  2. At each contact, record four features (direction-corrected
     foot_offset, knee_angle, trunk_lean, cadence_spm).
  3. Filter out peaks that happen while walking (cadence < 80 spm).

Each stride is one labeled sample; strides inherit the session label.
Train-test uses leave-one-session-out so strides from the same
recording never land in both train and test.

Run: python3 train_phase6.py
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(HERE, "recordings")

FEATURES = [
    "foot_offset_at_contact",
    "knee_angle_at_contact",
    "trunk_lean_at_contact",
    "cadence_at_contact",
]
PEAK_WINDOW = 3       # frames on each side a candidate must exceed
MIN_CADENCE = 80.0    # below this, assume walking and skip the contact
MIN_DIRECTION_DX = 1e-4  # tiny hip motion -> treat as stationary, skip


def detect_contacts(ankle_y, window=PEAK_WINDOW):
    """Return indices where ankle_y is a local maximum over +/- window frames."""
    peaks = []
    n = len(ankle_y)
    for i in range(window, n - window):
        y = ankle_y[i]
        if y > max(ankle_y[i - window:i]) and y > max(ankle_y[i + 1:i + 1 + window]):
            peaks.append(i)
    return peaks


def extract_strides(session_df):
    """Turn one session's per-frame DataFrame into a list of stride samples."""
    df = session_df[session_df["ready"] == 1].reset_index(drop=True)
    needed = ["ankle_y", "hip_x", "foot_offset", "knee_angle",
              "trunk_lean", "cadence_spm", "label"]
    if not all(c in df.columns for c in needed) or len(df) < 20:
        return []

    ankle_y = df["ankle_y"].to_numpy()
    hip_x = df["hip_x"].to_numpy()
    foot_offset = df["foot_offset"].to_numpy()
    knee_angle = df["knee_angle"].to_numpy()
    trunk_lean = df["trunk_lean"].to_numpy()
    cadence = df["cadence_spm"].to_numpy()
    label = df["label"].iloc[0]

    strides = []
    for p in detect_contacts(ankle_y):
        # Direction of travel from hip_x motion over +/- 5 frames around the peak.
        lo = max(0, p - 5)
        hi = min(len(df) - 1, p + 5)
        dx = hip_x[hi] - hip_x[lo]
        if abs(dx) < MIN_DIRECTION_DX:
            continue
        direction_sign = 1.0 if dx > 0 else -1.0

        # Only keep actively-running contacts.
        cad = cadence[p]
        if pd.isna(cad) or cad < MIN_CADENCE:
            continue

        # Skip if any feature is missing.
        fo = foot_offset[p]
        ka = knee_angle[p]
        tl = trunk_lean[p]
        if any(pd.isna(v) for v in (fo, ka, tl)):
            continue

        strides.append({
            "foot_offset_at_contact": float(fo) * direction_sign,
            "knee_angle_at_contact": float(ka),
            "trunk_lean_at_contact": float(tl),
            "cadence_at_contact": float(cad),
            "label": label,
            "direction_sign": direction_sign,
        })
    return strides


def load_all_strides():
    """Walk recordings/, extract strides from every session CSV that has the new schema."""
    all_strides = []
    for path in sorted(glob.glob(os.path.join(RECORDINGS_DIR, "session_*.csv"))):
        df = pd.read_csv(path)
        if "ankle_y" not in df.columns or "label" not in df.columns:
            continue  # pre-schema session; can't run Phase 6 on it
        df = df.sort_values("frame_index").reset_index(drop=True)
        strides = extract_strides(df)
        for s in strides:
            s["source_file"] = os.path.basename(path)
        all_strides.extend(strides)
    return all_strides


def main():
    strides = load_all_strides()
    if not strides:
        raise SystemExit(
            "No strides extracted. Make sure recordings/ has fresh sessions "
            "with hip_x, ankle_y, etc. populated."
        )

    df = pd.DataFrame(strides)
    print(f"Extracted {len(df)} strides from {df['source_file'].nunique()} sessions.\n")
    print("Strides per session:")
    for sess, sub in df.groupby("source_file"):
        label = sub["label"].iloc[0]
        print(f"  {sess:40s}  label={label:16s}  strides={len(sub)}")

    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    X = df[FEATURES]
    y = df["label"]
    sessions = sorted(df["source_file"].unique())

    print(f"\n{'=' * 78}")
    print(f"Leave-one-session-out CV  ({len(sessions)} folds)")
    print(f"{'=' * 78}")
    print(
        f"  {'held-out session':40s} {'true':16s} {'strides':>8s} {'accuracy':>10s}"
    )

    fold_results = []
    all_true, all_pred = [], []

    for held_out in sessions:
        train_mask = df["source_file"] != held_out
        test_mask = ~train_mask

        clf = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
        )
        clf.fit(X[train_mask], y[train_mask])
        preds = clf.predict(X[test_mask])
        true = y[test_mask]

        acc = (preds == true).mean()
        label = true.iloc[0]
        fold_results.append(
            {"session": held_out, "label": label, "acc": acc, "n": len(true)}
        )
        all_true.extend(true.tolist())
        all_pred.extend(preds.tolist())
        print(
            f"  {held_out:40s} {label:16s} {len(true):>8d} {acc:>10.3f}"
        )

    print(f"{'=' * 78}")
    accs = [r["acc"] for r in fold_results]
    print(f"Mean accuracy across folds:   {np.mean(accs):.3f}")
    print(f"Median:                       {np.median(accs):.3f}")
    print(f"Min / Max:                    {min(accs):.3f} / {max(accs):.3f}")

    print("\nPer-label mean accuracy:")
    for label in sorted(df["label"].unique()):
        label_accs = [r["acc"] for r in fold_results if r["label"] == label]
        print(f"  {label:16s}  n={len(label_accs)}   mean={np.mean(label_accs):.3f}")

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    print("\n=== Aggregate confusion matrix (rows=true, cols=predicted) ===")
    print(" " * 17 + " ".join(f"{c:>16}" for c in labels))
    for i, label in enumerate(labels):
        row = " ".join(f"{cm[i, j]:>16d}" for j in range(len(labels)))
        print(f"{label:>16} {row}")

    print("\n=== Classification report (aggregated) ===")
    print(classification_report(all_true, all_pred, digits=3))

    clf_final = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1,
    )
    clf_final.fit(X, y)
    print("=== Feature importances (model trained on all strides) ===")
    imps = sorted(
        zip(FEATURES, clf_final.feature_importances_), key=lambda kv: -kv[1]
    )
    for name, imp in imps:
        bar = "#" * int(imp * 50)
        print(f"  {name:26s}  {imp:.3f}  {bar}")


if __name__ == "__main__":
    main()
